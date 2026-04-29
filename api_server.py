"""
파일 이름 : api_server.py
기능 : 랜섬웨어 패밀리 분류 AI 서버 (FastAPI)
      백엔드 Spring Boot의 AiService와 통신하여 실시간 파일 이벤트 분석을 수행한다.
작성 날짜 : 2025/12/17/
작성자 : 시스템 (수정: 2026/01/18)
- XGBoost JSON(model_xgb.json) 로드 지원 추가
- classes.json 기반 클래스명 매핑 추가 (LabelEncoder 없어도 benign/ransomware 판정 가능)
- XGBoost/LightGBM/Sklearn predict_proba 호환 유지
"""

from __future__ import annotations

import os
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from fastapi import Body, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LightGBM 또는 XGBoost 모델 지원
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

app = FastAPI(
    title="Ransomware Family Classifier API Server",
    description="백엔드 Spring Boot와 통신하는 랜섬웨어 패밀리 분류 서버",
    version="1.0.1",
)

# CORS 설정 (백엔드에서 호출 가능하도록)
_cors_origins_raw = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8080")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Config & Model Loading
# ============================
CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./artifacts")

model: Any = None
label_encoder: Any = None
feature_list: List[str] = []
class_names: List[str] = []  # classes.json 기반 (예: ["benign","ransomware"])


def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일 로드"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] Config 파일 로드 실패: {e}, 기본값 사용")
        return {}


def _default_feature_list() -> List[str]:
    return [
        "file_read_count",
        "file_write_count",
        "file_delete_count",
        "file_rename_count",
        "file_encrypt_like_count",
        "changed_files_count",
        "random_extension_flag",
        "entropy_diff_mean",
        "file_size_diff_mean",
    ]


def load_artifacts(artifacts_dir: str) -> None:
    """
    artifacts 디렉토리에서 모델, label_encoder, features, classes 로드

    지원:
    - XGBoost JSON: model_xgb.json  (untitled21.py에서 model.save_model로 저장한 형태)
    - PKL: model_lgbm.pkl / model_xgb.pkl / model.pkl
    - label_encoder.pkl (옵션)
    - features.json (옵션)
    - classes.json  (옵션, label_encoder 없을 때 클래스명 매핑)
    """
    global model, label_encoder, feature_list, class_names

    artifacts_dir = artifacts_dir or "./artifacts"

    # ----------------------------
    # 1) Model load (XGBoost JSON 우선)
    # ----------------------------
    model = None

    xgb_json = os.path.join(artifacts_dir, "model_xgb.json")
    if os.path.exists(xgb_json):
        if not HAS_XGB:
            raise RuntimeError("xgboost가 설치되어 있지 않습니다. `python3 -m pip install xgboost`")
        try:
            clf = xgb.XGBClassifier()
            clf.load_model(xgb_json)
            model = clf
            print(f"[INFO] XGBoost JSON 모델 로드 성공: {xgb_json}")
        except Exception as e:
            raise RuntimeError(f"XGBoost JSON 모델 로드 실패: {e}")

    # ----------------------------
    # 2) PKL model fallback
    # ----------------------------
    if model is None:
        model_paths = [
            os.path.join(artifacts_dir, "model_lgbm.pkl"),
            os.path.join(artifacts_dir, "model_xgb.pkl"),
            os.path.join(artifacts_dir, "model.pkl"),
        ]
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    print(f"[INFO] PKL 모델 로드 성공: {model_path}")
                    break
                except Exception as e:
                    print(f"[WARN] PKL 모델 로드 실패 ({model_path}): {e}")

    if model is None:
        raise FileNotFoundError(
            "모델 파일을 찾을 수 없습니다. 다음 경로 확인: "
            f"{xgb_json}, {os.path.join(artifacts_dir,'model_lgbm.pkl')}, "
            f"{os.path.join(artifacts_dir,'model_xgb.pkl')}, {os.path.join(artifacts_dir,'model.pkl')}"
        )

    # ----------------------------
    # 3) Label Encoder load (optional)
    # ----------------------------
    label_encoder = None
    le_path = os.path.join(artifacts_dir, "label_encoder.pkl")
    if os.path.exists(le_path):
        try:
            with open(le_path, "rb") as f:
                label_encoder = pickle.load(f)
            print(f"[INFO] Label Encoder 로드 성공: {le_path}")
        except Exception as e:
            print(f"[WARN] Label Encoder 로드 실패: {e}")

    # ----------------------------
    # 4) Features list load (optional)
    # ----------------------------
    feature_list = []
    feats_path = os.path.join(artifacts_dir, "features.json")
    if os.path.exists(feats_path):
        try:
            with open(feats_path, "r", encoding="utf-8") as f:
                feature_list = json.load(f) or []
            print(f"[INFO] Features 리스트 로드 성공: {len(feature_list)}개")
        except Exception as e:
            print(f"[WARN] Features 리스트 로드 실패: {e}")

    if not feature_list:
        feature_list = _default_feature_list()
        print(f"[WARN] 기본 피처 리스트 사용: {feature_list}")

    # ----------------------------
    # 5) Classes list load (optional)
    # ----------------------------
    class_names = []
    classes_path = os.path.join(artifacts_dir, "classes.json")
    if os.path.exists(classes_path):
        try:
            with open(classes_path, "r", encoding="utf-8") as f:
                class_names = json.load(f) or []
            if class_names:
                print(f"[INFO] classes 로드 성공: {class_names}")
        except Exception as e:
            print(f"[WARN] classes.json 로드 실패: {e}")

    # fallback: label_encoder가 있으면 그걸 우선 사용, 없으면 이진 분류 기본
    if not class_names and label_encoder is None:
        class_names = ["benign", "ransomware"]
        print(f"[WARN] classes 기본값 사용: {class_names}")


# 애플리케이션 시작 시 아티팩트 로드
try:
    cfg = load_config(CONFIG_PATH)
    if isinstance(cfg, dict) and "output_dir" in cfg and cfg["output_dir"]:
        OUTPUT_DIR = cfg["output_dir"]
    load_artifacts(OUTPUT_DIR)
except Exception as e:
    print(f"[ERROR] 아티팩트 로드 실패: {e}")
    print("[WARN] 서버는 시작되지만 모델이 없어 예측이 실패할 수 있습니다.")


# ============================
# Request/Response Schemas
# ============================
class PredictRequest(BaseModel):
    """
    /predict 엔드포인트 요청 형식
    {
        "features": {...},  # 또는 바디에 직접 피처 넣기
        "topk": 5
    }
    """
    features: Optional[Dict[str, Any]] = None
    topk: int = 5

    class Config:
        extra = "allow"


class PredictResponseItem(BaseModel):
    """패밀리 분류 결과 항목"""
    family: str
    prob: float


class PredictResponse(BaseModel):
    """패밀리 분류 응답"""
    topk: List[PredictResponseItem]
    message: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """
    /api/analyze 엔드포인트 응답 형식
    백엔드 AiResponse DTO와 호환되도록 설계
    """
    status: Optional[str] = "ok"
    label: Optional[str] = None  # SAFE / WARNING / DANGER / UNKNOWN
    score: Optional[float] = None  # 0~1 위험도
    detail: Optional[str] = None  # "top_family=LockBit, top_prob=0.92" 형식
    message: Optional[str] = None
    topk: Optional[List[PredictResponseItem]] = None


# ============================
# Helper Functions
# ============================
def _to_float_or_int(v: Any) -> Any:
    """문자열 숫자를 숫자로 변환"""
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except Exception:
            return v
    return v


def _normalize_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """피처 값 정규화 (문자열 -> 숫자)"""
    return {k: _to_float_or_int(v) for k, v in raw.items() if v is not None}


def _build_row_from_features(features: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    모델이 기대하는 피처 리스트 기준으로 row 생성
    반환: (row_dict, matched_keys, missing_keys)
    """
    matched = [c for c in feature_list if c in features]
    missing = [c for c in feature_list if c not in features]
    row = {c: features.get(c, 0) for c in feature_list}  # 누락은 0으로 채움 (카운트 피처의 자연스러운 결측값)
    return row, matched, missing


def _class_name(i: int) -> str:
    """클래스 인덱스를 사람이 읽을 수 있는 이름으로 매핑"""
    if label_encoder is not None and hasattr(label_encoder, "classes_") and i < len(label_encoder.classes_):
        return str(label_encoder.classes_[i])
    if class_names and i < len(class_names):
        return str(class_names[i])
    return f"Class_{i}"


def _predict_proba_any(X: pd.DataFrame) -> np.ndarray:
    """
    다양한 모델 타입에 대해 확률 배열 반환.
    반환 shape: (1, C)
    """
    if model is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")

    # LightGBM Booster
    if HAS_LGB and isinstance(model, getattr(lgb, "Booster", ())):
        proba = model.predict(
            X.values,
            num_iteration=model.best_iteration if hasattr(model, "best_iteration") else None,
        )
        proba = np.array(proba)
        # binary면 (N,)로 나올 수 있어 -> (N,2)로 변환
        if proba.ndim == 1:
            p1 = proba.reshape(-1, 1)
            p0 = 1.0 - p1
            proba = np.concatenate([p0, p1], axis=1)
        return proba

    # sklearn 스타일 (XGBClassifier 포함)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.array(proba)
        return proba

    # fallback: predict가 확률을 줄 수도 있음
    if hasattr(model, "predict"):
        proba = model.predict(X)
        proba = np.array(proba)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        return proba

    raise RuntimeError("모델 예측 메서드를 찾을 수 없습니다.")


def _predict_topk(features: Dict[str, Any], topk: int = 5) -> List[PredictResponseItem]:
    """
    모델로 예측하고 top-k 결과 반환
    """
    # 피처 정규화
    features = _normalize_features(features)

    # 모델 입력 형식으로 변환 (누락 피처는 _build_row_from_features에서 이미 -1로 채워짐)
    row, matched, missing = _build_row_from_features(features)
    X = pd.DataFrame([row])

    # 확률 예측
    proba = _predict_proba_any(X)
    proba = np.array(proba)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)

    # topk 추출
    topk = max(1, min(int(topk), proba.shape[1]))
    idxs = np.argsort(-proba[0])[:topk]

    items: List[PredictResponseItem] = []
    for i in idxs:
        items.append(PredictResponseItem(family=_class_name(int(i)), prob=float(proba[0, i])))

    return items


# ============================
# API Endpoints
# ============================
@app.get("/health")
def health(response: Response):
    """헬스체크 — 모델 미로드 시 503 반환"""
    if model is None:
        response.status_code = 503
    return {
        "ok": model is not None,
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "feature_count": len(feature_list),
        "classes": class_names if class_names else (list(label_encoder.classes_) if label_encoder is not None and hasattr(label_encoder, "classes_") else None),
        "model_type": type(model).__name__ if model is not None else None,
        "output_dir": OUTPUT_DIR,
    }


@app.get("/debug/feats")
def debug_feats():
    """디버깅: 모델이 기대하는 피처 리스트 확인"""
    return {
        "n_model_feats": len(feature_list),
        "feats": feature_list,
        "model_type": type(model).__name__ if model else None,
        "classes": class_names if class_names else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    패밀리 분류 예측 (원본 형식)
    백엔드 FamilyPredictRequest와 호환
    """
    # features 추출
    features = req.features
    if features is None:
        raw = req.model_dump()
        raw.pop("features", None)
        raw.pop("topk", None)
        features = raw

    # 예측
    items = _predict_topk(features, req.topk)

    # 메시지 생성
    row, matched, missing = _build_row_from_features(_normalize_features(features))
    msg = None
    if missing:
        msg = f"Missing {len(missing)} features filled with 0."

    return PredictResponse(topk=items, message=msg)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def api_analyze(payload: Dict[str, Any] = Body(...)):
    """
    ✅ 백엔드 AiService.requestAnalysis()가 호출하는 엔드포인트
    AiPayload (9개 피처)를 받아서 AiResponse 형식으로 응답
    """
    try:
        # 피처 정규화
        features = _normalize_features(payload)

        # 예측 수행 (이진분류면 사실상 2개)
        items = _predict_topk(features, topk=5)

        if not items:
            return AnalyzeResponse(
                status="error",
                label="UNKNOWN",
                message="예측 결과가 없습니다.",
            )

        # top-1 결과 추출 (detail 문자열용)
        top1 = items[0]
        top_family = top1.family
        top_prob = top1.prob

        # 라벨 결정
        # 다중 클래스에서도 정확하도록 benign 이외 모든 클래스 확률을 합산해 위험도(score)를 계산한다.
        # 이진 분류(benign/ransomware)에서는 기존 방식과 동일한 결과를 낸다.
        if not top_family:
            label = "UNKNOWN"
            score = None
        else:
            ransomware_score = float(sum(
                item.prob for item in items if item.family.lower() != "benign"
            ))
            score = max(0.0, min(1.0, ransomware_score))
            if score >= 0.70:
                label = "DANGER"
            elif score >= 0.50:
                label = "WARNING"
            else:
                label = "SAFE"

        # detail 문자열 생성 (백엔드가 파싱하는 형식)
        detail_parts = [f"top_family={top_family}"]
        detail_parts.append(f"top_prob={top_prob:.4f}")
        detail = ", ".join(detail_parts)

        # 메시지
        row, matched, missing = _build_row_from_features(features)
        msg_parts = []
        if missing:
            msg_parts.append(f"Missing {len(missing)} features filled with 0.")
        if matched:
            msg_parts.append(f"Matched {len(matched)} features.")
        message = " ".join(msg_parts) if msg_parts else None

        return AnalyzeResponse(
            status="ok",
            label=label,
            score=score,
            detail=detail,
            message=message,
            topk=items,
        )

    except Exception as e:
        import traceback
        error_msg = f"예측 중 오류 발생: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(traceback.format_exc())
        return AnalyzeResponse(
            status="error",
            label="UNKNOWN",
            message=error_msg,
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
