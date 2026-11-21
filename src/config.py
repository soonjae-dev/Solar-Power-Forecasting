"""경로와 대회 상수."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 데이터는 작업 폴더 밖에 둔다(외부 AI 서비스 업로드 금지 규정). OIBC_DATA_ROOT로 바꿀 수 있다.
DATA_ROOT = Path(os.environ.get("OIBC_DATA_ROOT", ROOT.parent / "OIBC_2026_data"))
DATA_DIR = DATA_ROOT / "raw"                # 원본. 읽기 전용
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
EXP_DIR = ROOT / "experiments"              # metrics.json만 둔다
PRED_DIR = DATA_ROOT / "predictions"        # 발전소별 예측
SUB_DIR = DATA_ROOT / "submissions"
CACHE_DIR = DATA_ROOT / os.environ.get("OIBC_CACHE_NAME", "cache")   # 다른 특징 구성을 시험할 때 이름을 바꾼다

TZ = "Asia/Seoul"
FIRST_TEST_CUTOFF = "2025-05-31 17:00"   # 평가 첫 대상일(2025-06-01)의 예측시점. 제출용 학습은 여기까지만 쓴다
MIN_GROUP_KW = 20_000.0
PENALTY_RATE = 0.10
N_SITES = 253
N_TEST_HOURS = 8_760
