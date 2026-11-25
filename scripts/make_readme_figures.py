"""README 그림 생성. 공개된 집계값(README·docs 표의 서버 점수, 오라클 분해)만 쓴다.

    python scripts/make_readme_figures.py

docs/images/ 에 영문(*_en.png)·국문(*_ko.png) 그림을 저장한다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"

INK, MUTED, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
BLUE, ORANGE, BLUE_LIGHT = "#257de0", "#f47230", "#b6d5f8"
KO_FONTS = ["Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"]

# 서버 정산금 S (원). README '서버 점수 추이' 표와 같다.
BASE = 6_418_399_860
BEST = [
    ("v3", "84 features\nkt target · 3 seeds", "특징 84개\n청천 지수 · 3시드", 6_446_717_423),
    ("v5", "+ GFS irradiance\n& cloud (20)", "+ GFS 일사·운량\n20열", 6_455_854_022),
    ("v6", "+ GFS cloud\nlayers & PW (40)", "+ GFS 운량 층·\n가강수량 40열", 6_460_890_524),
    ("split", "+ teammate's\naggregate split", "+ 팀원\n집합 배정", 6_462_937_912),
    ("D", "+ CatBoost blend\n& snow model", "+ CatBoost 결합\n& 눈 모델", 6_464_297_778),
    ("#6", "3-model\nequal blend", "3모델\n등분 결합", 6_464_511_199),
    ("#9", "+ LDAPS in\nv6 component", "+ v6 성분에\nLDAPS", 6_465_154_213),
]
# 서버 S 차이(만 원). docs/experiments.md 1·2절.
HURT = [
    ("Forecast level ×0.984", "예측 수준 ×0.984", -251),
    ("CatBoost alone", "CatBoost 단독", -141),
    ("LDAPS in all components", "전 성분 LDAPS", -97),
    ("Median-shape scaling", "중앙값 모양 배율", -78),
    ("Drop outage days", "정지일 학습 제외", -50),
]
# 정답 수준을 알았을 때의 페널티 감소(%, 2024-06 ~ 2025-04). 기술 보고서 Figure 1과 같다.
ORACLE = [("Constant", "상수", 1.2), ("Monthly", "월별", 4.0),
          ("Month × hour", "월 × 시각", 4.7), ("Daily total", "일 총량", 47.5)]

TEXT = {
    "en": dict(
        prog_title="Server settlement gain over the official baseline",
        prog_y="Cumulative gain (M KRW)", prog_unit=lambda v: f"+{v:.1f}M",
        total="Final: +46.8M KRW (rank 134 / 385)",
        lvs_title="Server result of changes that passed local validation",
        lvs_x="Server settlement change (M KRW)", adopted="Adopted (server gain)",
        hurt="Rejected after server loss", lvs_unit=lambda v: f"{v:+.2f}M",
        dec_title="Penalty reduction if the true level were known (oracle)",
        dec_sub="Diagnostic on Jun 2024 – Apr 2025: rescale the forecast to the true total at each resolution",
        dec_x="Penalty reduction (%)",
    ),
    "ko": dict(
        prog_title="공식 기준선 대비 서버 정산금 개선 추이",
        prog_y="누적 개선 (만 원)", prog_unit=lambda v: f"+{v:,.0f}",
        total="최종 +4,675만 원 (385팀 중 134위)",
        lvs_title="로컬 검증 통과 변경의 서버 결과",
        lvs_x="서버 정산금 변화 (만 원)", adopted="채택 (서버 개선)",
        hurt="서버 손해로 기각됨", lvs_unit=lambda v: f"{v:+,.0f}",
        dec_title="정답 수준을 알 때의 페널티 감소 폭 (오라클 진단)",
        dec_sub="2024-06~2025-04 구간 진단, 해상도별로 예측을 실제 총량에 맞춰 배율 조정",
        dec_x="페널티 감소 (%)",
    ),
}


def setup(lang):
    family = ["DejaVu Sans"]
    if lang == "ko":
        dirs = ["/System/Library/Fonts", "/Library/Fonts", str(Path.home() / "Library/Fonts"),
                "/usr/share/fonts", "/usr/local/share/fonts", "C:/Windows/Fonts"]
        files = [p for d in dirs if Path(d).is_dir() for p in Path(d).rglob("*") if p.suffix.lower() in (".ttf", ".otf", ".ttc")]
        for path in files:
            if any(k in path.name for k in ("AppleSDGothicNeo", "AppleGothic", "NotoSansCJK-Regular", "NanumGothic.", "malgun")):
                try:
                    font_manager.fontManager.addfont(path)
                except (OSError, RuntimeError):
                    pass
        have = {f.name for f in font_manager.fontManager.ttflist}
        family = [f for f in KO_FONTS if f in have][:1] + family
    plt.rcParams.update({
        "font.family": family, "font.size": 10, "axes.unicode_minus": False,
        "axes.edgecolor": AXIS, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, name, lang):
    fig.savefig(OUT / f"{name}_{lang}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fig_progress(lang):
    t = TEXT[lang]
    scale = 1e6 if lang == "en" else 1e4
    cum = [(s - BASE) / scale for *_, s in BEST]
    prev = [0.0] + cum[:-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = range(len(BEST))
    ax.bar(x, [c - p for c, p in zip(cum, prev)], bottom=prev, color=BLUE, width=0.6)
    for i in range(len(BEST) - 1):
        ax.plot([i + 0.3, i + 0.7], [cum[i], cum[i]], color=AXIS, linewidth=1, linestyle=":")
    for i, (c, p) in enumerate(zip(cum, prev)):
        ax.text(i, c + cum[-1] * 0.015, t["prog_unit"](c - p), ha="center", va="bottom", color=INK, fontsize=9)
    labels = [f"{k}\n{en if lang == 'en' else ko}" for k, en, ko, _ in BEST]
    ax.set_xticks(list(x), labels, fontsize=8)
    ax.set_ylabel(t["prog_y"])
    ax.set_ylim(0, cum[-1] * 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_title(t["prog_title"], pad=26)
    ax.text(0, 1.03, t["total"], transform=ax.transAxes, color=MUTED, fontsize=9.5)
    save(fig, "server_progress", lang)


def fig_local_vs_server(lang):
    t = TEXT[lang]
    scale = 100 if lang == "en" else 1  # 만 원 → M KRW
    gains = [(k, en, ko, (s - p) / 1e4) for (k, en, ko, s), p in zip(BEST, [BASE] + [b[3] for b in BEST[:-1]])][1:]
    rows = [(f"{k}: " + (en if lang == "en" else ko).replace("\n", " "), round(v)) for k, en, ko, v in gains]
    rows.sort(key=lambda r: r[1])
    hurt = sorted(((en if lang == "en" else ko), v) for en, ko, v in HURT)
    allrows = sorted(hurt, key=lambda r: r[1]) + rows  # barh는 아래부터: 큰 손해가 맨 아래, 큰 개선이 맨 위
    names = [r[0] for r in allrows]
    vals = [r[1] / scale for r in allrows]
    cols = [ORANGE if v < 0 else BLUE for v in vals]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = range(len(allrows))
    ax.barh(y, vals, color=cols, height=0.62, edgecolor="white", linewidth=2)
    span = max(vals) - min(vals)
    for i, v in enumerate(vals):
        ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02), i, t["lvs_unit"](v),
                ha="left" if v >= 0 else "right", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(list(y), names, fontsize=8.5)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_xlim(min(vals) - span * 0.16, max(vals) + span * 0.14)
    ax.set_xlabel(t["lvs_x"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+,.0f}" if v else "0"))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=BLUE, label=t["adopted"]), Patch(color=ORANGE, label=t["hurt"])],
              frameon=False, loc="lower right", fontsize=8.5)
    ax.set_title(t["lvs_title"], pad=12)
    save(fig, "local_vs_server", lang)


def fig_decomp(lang):
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    names = [en if lang == "en" else ko for en, ko, _ in ORACLE]
    vals = [v for *_, v in ORACLE]
    ax.barh(names, vals, color=[BLUE_LIGHT] * 3 + [BLUE], height=0.6, edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"−{v}%", va="center", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 55)
    ax.set_xlabel(t["dec_x"])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(t["dec_title"], pad=24)
    ax.text(0, 1.04, t["dec_sub"], transform=ax.transAxes, color=MUTED, fontsize=8.5)
    save(fig, "penalty_decomposition", lang)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "ko"):
        setup(lang)
        fig_progress(lang)
        fig_local_vs_server(lang)
        fig_decomp(lang)
    print("saved:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
