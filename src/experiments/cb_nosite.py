"""식별 입력 없는 공유 CatBoost, p55. 집계만 저장한다."""
import json
import traceback

from ..cb_onehot import PARAMS, numeric_input, run_candidate
from ..config import EXP_DIR

SPEC = dict(rule="14.42", params=PARAMS, seeds=[2026, 2027, 2028], features=143,
            excluded_features=["site"])
PATH = EXP_DIR/"p55_cb_nosite"/"metrics.json"


def main():
    run_candidate(spec=SPEC, path=PATH, drop_site=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
