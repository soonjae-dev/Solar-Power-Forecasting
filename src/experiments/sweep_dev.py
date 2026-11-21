"""개발(2025-01~02)·선택(2025-03~04) 구간에서 한 번에 한 가지씩 바꿔 본다.

    python -m src.experiments.sweep_dev 1|2|...

1차 결과(개발 구간, 참조 배정 페널티): baseline 29.9M → 전체 특징 27.2M, 용량 가중 27.0M,
L1은 28.0M으로 오히려 나빴다. 행 순서만 달라도 약 1% 흔들리므로 그 이하 차이는 잡음이다.
"""
import sys

from ..train import run

FOLDS = ("development", "selection")


def both(name, **kw):
    rows = [run(name, fold=f, **kw) for f in FOLDS]
    print(f"  => {name:<26} 합계 ref {sum(r['penalty_ref'] for r in rows):>12,}  "
          f"lb {sum(r['penalty_lb'] for r in rows):>12,}", flush=True)


SWEEPS = {
    "1": lambda: [
        run("a0_baseline", feats="baseline"),
        run("a1_site", feats="baseline_site"),
        run("a2_full_nosite", feats="full_nosite"),
        run("a3_full", feats="full"),
        run("b1_full_cap", feats="full", weight="cap"),
        run("b2_full_l1", feats="full", objective="l1"),
        run("b3_full_l1_cap", feats="full", objective="l1", weight="cap"),
        run("b4_full_l1_capsmp", feats="full", objective="l1", weight="capsmp"),
        run("b5_full_huber_cap", feats="full", objective="huber", alpha=0.1, weight="cap"),
    ],
    "2": lambda: [
        both("c0_baseline", feats="baseline"),
        both("c1_full_cap", weight="cap"),
        both("c1_full_cap_seed7", weight="cap", seed=7),
        both("c2_nosite_cap", feats="full_nosite", weight="cap"),
        both("c3_from2021", weight="cap", train_start="2021-01-01"),
        both("c4_from2022", weight="cap", train_start="2022-06-01"),
        both("c5_from2024", weight="cap", train_start="2024-01-01"),
        both("c6_slow", weight="cap", n_estimators=1500, learning_rate=0.03),
        both("c7_leaves31", weight="cap", num_leaves=31, min_child_samples=100),
        both("c8_capsmp", weight="capsmp"),
    ],
    # 2차 결론: 2021년부터 전부 학습하면 두 구간 합계 59.6M → 57.3M. 시드만 바꿔도 0.6M 흔들린다.
    # 느린 학습률·작은 트리·SMP 가중·2024년부터 학습은 이득이 없었다.
    "3": lambda: [
        both("d0_nbr", weight="cap", train_start="2021-01-01"),                 # 인접 관측소 특징 추가 후
        both("d1_kt", weight="cap", train_start="2021-01-01", target="kt"),
        both("d2_leaves255", weight="cap", train_start="2021-01-01", num_leaves=255, min_child_samples=100),
        both("d3_noweight", train_start="2021-01-01"),
    ],
}

if __name__ == "__main__":
    SWEEPS[sys.argv[1]]()
