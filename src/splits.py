"""시간순 검증 구간 (노트북 16-A).

우측닫힘이라 대상 구간은 시작일 01시 ~ 종료 경계일 00시이고,
학습은 첫 대상일의 전일 17시까지만 쓴다.
"""
import pandas as pd

from .config import TZ

_FOLDS = {
    "summer_support": ("2024-06-01", "2024-09-01"),   # 205개소만 정답 완전
    "autumn_support": ("2024-09-01", "2024-12-01"),   # 252개소
    "development":    ("2025-01-01", "2025-03-01"),
    "selection":      ("2025-03-01", "2025-05-01"),
    "final_holdout":  ("2025-05-01", "2025-06-01"),   # 선택이 끝난 뒤 한 번만 채점
}
# 패턴이 다른 해에도 나타나는지 볼 때만 쓰는 과거 구간(src/diagnose_cloudy). _FOLDS를 도는 감사에는 넣지 않는다.
_HISTORY = {
    "autumn_2022": ("2022-09-01", "2022-12-01"),
    "janfeb_2023": ("2023-01-01", "2023-03-01"),
    "spring_2023": ("2023-03-01", "2023-05-01"),
    "summer_2023": ("2023-06-01", "2023-09-01"),
    "autumn_2023": ("2023-09-01", "2023-12-01"),
    "janfeb_2024": ("2024-01-01", "2024-03-01"),
    "spring_2024": ("2024-03-01", "2024-05-01"),
    "fold1_static": ("2024-06-01", "2025-05-01"),     # 한 번 학습한 모델로 11개월(src/diagnose_static). 5월 holdout은 넣지 않는다
    "min_fold2":    ("2025-01-01", "2025-05-01"),     # 팀원 설계와 같은 조건(src/diagnose_fold2). 학습은 첫날 00시까지
    # 팀원 배정 탐색용 6개월 확장창 OOF 블록(팀원 보고서와 같은 구간)
    "oof6m_1": ("2022-01-01", "2022-07-01"),
    "oof6m_2": ("2022-07-01", "2023-01-01"),
    "oof6m_3": ("2023-01-01", "2023-07-01"),
    "oof6m_4": ("2023-07-01", "2024-01-01"),
    "oof6m_5": ("2024-01-01", "2024-06-01"),
}


def fold(name: str) -> dict:
    start_day, stop_day = (pd.Timestamp(d, tz=TZ) for d in {**_FOLDS, **_HISTORY}[name])
    return dict(name=name,
                cutoff=start_day if name == "min_fold2" else start_day - pd.Timedelta(hours=7),
                val_start=start_day + pd.Timedelta(hours=1),
                val_end=stop_day)
