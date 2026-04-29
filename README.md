# WatchService AI Server

랜섬웨어 실시간 탐지 시스템의 AI 분석 서버입니다.  
Spring Boot 백엔드로부터 파일 행위 피처를 받아 XGBoost 모델로 랜섬웨어 여부를 판정합니다.

## 기술 스택

- Python 3.9+
- FastAPI
- XGBoost
- uvicorn

## 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버 및 모델 로드 상태 확인 |
| POST | `/api/analyze` | 행위 피처 분석 → SAFE / WARNING / DANGER 판정 |
| POST | `/predict` | 랜섬웨어 패밀리 분류 (top-k) |

## 입력 피처 (9개)

`/api/analyze` 에 전달하는 JSON 필드:

| 필드 | 설명 |
|------|------|
| `fileReadCount` | 파일 읽기 횟수 |
| `fileWriteCount` | 파일 쓰기 횟수 |
| `fileDeleteCount` | 파일 삭제 횟수 |
| `fileRenameCount` | 파일 이름 변경 횟수 |
| `fileEncryptLikeCount` | 암호화 의심 파일 수 |
| `changedFilesCount` | 변경된 고유 파일 수 |
| `randomExtensionFlag` | 의심 확장자 플래그 (0 or 1) |
| `entropyDiffMean` | 엔트로피 변화 평균 |
| `fileSizeDiffMean` | 파일 크기 변화 평균 (bytes) |

## 로컬 실행

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn api_server:app --host 0.0.0.0 --port 8001
```

## Railway 배포

이 레포는 Railway에 직접 연결해서 배포합니다.  
`railway.toml` 과 `Procfile` 이 배포 설정을 담고 있습니다.

배포 완료 후 헬스체크:
```bash
curl https://[Railway URL]/health
# {"ok":true,"model_loaded":true,...}
```

## 파일 구조

```
├── api_server.py        # FastAPI 서버 본체
├── requirements.txt     # Python 의존성
├── Procfile             # Railway 실행 명령
├── railway.toml         # Railway 배포 설정
├── configs/
│   └── config.yaml
└── artifacts/
    ├── model_xgb.json   # XGBoost 모델 (필수)
    ├── features.json    # 피처 목록
    └── classes.json     # 클래스 레이블
```
