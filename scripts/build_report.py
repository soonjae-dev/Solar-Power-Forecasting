"""Build docs/OIBC2026_technical_report.pdf from aggregate results only.

    pip install reportlab==4.2.5 matplotlib==3.9.4
    python scripts/build_report.py

Inputs: server scores (hard-coded below, from the team's submission ledger) and
experiments/p92_median_level/metrics.json. No competition data is read.
"""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"docs"/"OIBC2026_technical_report.pdf"
INK, MUTED, GRID, ACCENT, NEG = "#1f2933", "#6b7280", "#e5e7eb", "#2563eb", "#dc2626"
BASE = 6_418_399_860
BEST = [("Official\nbaseline", BASE), ("v3\nfeatures", 6_446_717_423), ("v5\n+GFS 20", 6_455_854_022),
        ("v6\n+GFS 40", 6_460_890_524), ("+split", 6_462_937_912), ("D\n+CatBoost", 6_464_297_778),
        ("#6\n3-model", 6_464_511_199), ("#9\n+LDAPS", 6_465_154_213)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def fig_png(fig, width_mm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_mm*mm, height=width_mm*mm*h/w)


def fig_progress():
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    names = [n for n, _ in BEST]
    gain = [(s-BASE)/1e6 for _, s in BEST]
    ax.bar(names, gain, color=[GRID]+[ACCENT]*(len(gain)-1), width=0.62)
    for i, g in enumerate(gain[1:], 1):
        ax.text(i, g+0.8, f"+{g:.1f}", ha="center", va="bottom", color=INK, fontsize=7.5)
    ax.set_ylabel("S − baseline (M KRW)")
    ax.set_ylim(0, max(gain)*1.18)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig_png(fig, 165)


def fig_decomp():
    fig, ax = plt.subplots(figsize=(4.2, 1.7))
    lv = ["Constant", "Monthly", "Month × hour", "Daily total"]
    val = [1.2, 4.0, 4.7, 47.5]
    ax.barh(lv, val, color=[GRID, GRID, GRID, ACCENT], height=0.55)
    for y, v in enumerate(val):
        ax.text(v+0.8, y, f"−{v}%", va="center", color=INK, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 56)
    ax.set_xlabel("Penalty reduction if this level were known (oracle)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig_png(fig, 105)


def fig_level():
    d = json.loads((ROOT/"experiments"/"p92_median_level"/"metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    shades = ["#9ca3af", "#6b7280", "#374151", ACCENT]
    labels = ["2022-01–05", "2022-06–2023-05", "2023-06–2024-05", "2024-06–2025-04 (main local year)"]
    for (name, v), col, lab in zip(d["blocks"].items(), shades, labels):
        c = [r["c"] for r in v["curve"]]
        y = [r["rule_pct"] for r in v["curve"]]
        ax.plot(c, y, color=col, linewidth=1.6, label=lab)
        if v["server_like_c"] is not None:
            ax.plot(v["server_like_c"], v["rule_at_server_like_pct"], "o", color=col, markersize=4.5)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axhline(0.57, color=NEG, linewidth=0.9, linestyle="--")
    ax.text(1.098, 0.75, "server result +0.57%", color=NEG, ha="right", fontsize=7.5)
    ax.set_xlabel("Base forecast multiplied by c before applying the correction")
    ax.set_ylabel("Penalty change of correction (%)")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    return fig_png(fig, 140)


def table(rows, widths, head=True):
    t = Table(rows, colWidths=[w*mm for w in widths])
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.3), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
             ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(MUTED)), ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor(MUTED)),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]
    if head:
        style.append(("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.3))
    t.setStyle(TableStyle(style))
    return t


def main():
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.3, leading=12.6, textColor=colors.HexColor(INK), spaceAfter=4)
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=3)
    cap = ParagraphStyle("c", parent=body, fontSize=8, leading=10, textColor=colors.HexColor(MUTED), alignment=TA_CENTER, spaceAfter=8)
    title = ParagraphStyle("t", parent=body, fontName="Helvetica-Bold", fontSize=15, leading=19, alignment=TA_CENTER, spaceAfter=2)
    sub = ParagraphStyle("s", parent=body, fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor(MUTED), spaceAfter=10)
    P = lambda s, st=body: Paragraph(s, st)
    bl = lambda items: [P("•&nbsp;&nbsp;"+x, ParagraphStyle("li", parent=body, leftIndent=9, firstLineIndent=-9, spaceAfter=2)) for x in items]

    s = [P("Why Local Gains Reversed on the Server:<br/>Lessons from Day-Ahead PV Forecasting for Aggregated Resources", title),
         P("Technical report · 2026 POSTECH OIBC Big Data Challenge · [Author name] · November 2026", sub)]

    s += [P("Abstract", h1), P(
        "We forecast hourly generation of two aggregates of 253 PV plants (42.7 MW) one day ahead, scored by an SMP-weighted "
        "absolute error on aggregate sums. A shared LightGBM model on a clear-sky-index target, enriched with public NWP "
        "(NOAA GFS and KMA LDAPS) and blended with CatBoost, reduced the official baseline penalty by about 25% "
        "(+46.8 M KRW in settlement). Every one of ~90 experiments was pre-registered with a fixed acceptance rule. "
        "The main finding is methodological. Five changes that passed local validation lost on the server, and three of them "
        "share one cause: out-of-sample evidence was concentrated in a single year in which the model over-forecast by 1.3%, "
        "so any change that lowered forecasts looked beneficial locally. We show that even a total-preserving correction carries "
        "this level effect, and we propose a level-sensitivity check for future validation.")]

    s += [P("1. Problem", h1), P(
        "Plants are assigned once to G1 or G2 (each ≥ 20 MW). At 17:00 KST on D−1 we submit hourly forecasts F<sub>g</sub>(t) for each "
        "aggregate over a full year (8,760 h). The settlement is S = Σ SMP·A − 0.1·Σ SMP·k·|F<sub>g</sub> − A<sub>g</sub>|, with k = 3 "
        "between 11:00 and 15:00. Only the second term (the <i>penalty</i>) depends on our choices. Two structural facts guided the work:"),
        *bl(["<b>Composition has a hard floor.</b> With plant forecasts fixed, |E<sub>1</sub>| + |E<sub>2</sub>| ≥ |E<sub>1</sub>+E<sub>2</sub>|, "
             "so no split can beat 0.1·Σ w·|total error|. The reference split was only 2.6% above this bound, and an in-sample optimum only 1.4% better.",
             "<b>The penalty is dominated by daily totals.</b> If each day's total were known, the penalty would fall by 47.5%. "
             "Knowing monthly levels would remove only 4% (Fig. 1). This pointed us to better day-ahead irradiance information. In hindsight, "
             "model structure that captures within-day dependence also mattered (Section 6)."])]
    s += [KeepTogether([fig_decomp(), P("Figure 1. Penalty decomposition with oracle scaling at different resolutions (June 2024 – April 2025).", cap)])]

    s += [P("2. Data, features and model", h1), *bl([
        "<b>Inputs.</b> Two vendors' day-ahead forecasts (no irradiance variable; cloud forecasts correlate at only 0.73), solar geometry and plant metadata. "
        "Generation history runs from 2021, but 22% of capacity came online after June 2024.",
        "<b>Target.</b> Capacity factor divided by Haurwitz clear-sky irradiance, weighted by capacity × clear-sky irradiance. "
        "One model is shared by all plants, because short histories make per-plant models unable to see unseen seasons.",
        "<b>External NWP.</b> GFS (00 UTC run of D−1): 6 variables × 10 fixed 2° regions. LDAPS (1.5 km): shortwave × 10 regions. "
        "Each file's publication time was audited against the deadline. Five candidate sources were screened by their correlation with daily totals "
        "<i>before</i> any costly download (LDAPS 0.912, GFS 0.904, MSM 0.859, ECMWF 0.857).",
        "<b>Final blend.</b> An equal-weight average of LightGBM + GFS + LDAPS (with a snow-specific model on snow-forecast hours), "
        "CatBoost with plant ID, and CatBoost without plant ID."])]

    s += [KeepTogether([P("3. Validation protocol", h1), P(
        "All folds are time-ordered and use only data available at each forecast time. Four seasonal folds (summer 2024 – spring 2025) "
        "and a static 11-month fold (one model trained to May 2024) were scored with a local implementation of the official formula, "
        "with 7-day block-bootstrap intervals. A change was accepted only if the 4-season sum improved by ≥ 1%, at least three seasons improved, "
        "no season worsened by more than 3%, and the static fold did not worsen. Rules were written before results. Server submissions were "
        "fixed in advance and never tuned to the leaderboard.")])]

    s += [P("4. Results", h1)]
    s += [KeepTogether([fig_progress(), P("Figure 2. Server settlement gain over the official baseline for each new team best.", cap)])]
    s += [table([["Change", "Local validation", "Server (ΔS)"],
                 ["Feature set, full history, clear-sky target (v3)", "−16% (two dev folds)", "+28.3 M"],
                 ["+ GFS irradiance & cloud, 60 columns (v5, v6)", "passed, all folds", "+14.2 M"],
                 ["Residual-based composition", "static fold improved", "+2.0 M"],
                 ["+ CatBoost blend & snow model (D, #6)", "−2.9% (4 seasons)", "+1.6 M"],
                 ["+ LDAPS in LightGBM component (#9)", "−1.1% (4 seasons)", "+0.6 M"],
                 ["CatBoost alone (#4)", "large gain", "−1.4 M"],
                 ["Forecast level × 0.984 (#8)", "gain", "−2.5 M"],
                 ["Total-preserving median-shape correction (#13)", "−1.1 to −1.7%", "−0.8 M"]], [85, 45, 30]),
          Spacer(1, 4), P(
        "About 90 further experiments were rejected under the pre-registered rule. These include alternative NWP (ECMWF, JMA MSM, GEFS mean/spread/aerosol, "
        "finer LDAPS grids), L1/Huber losses, hyper-parameter search, per-source models, a day-level residual TCN, holiday features and "
        "regime-specific corrections. The final rank was outside the top 100. The best score sits 1.9% (penalty) below rank 100 and 6.5% below rank 6.")]

    s += [P("5. Why local gains reversed", h1), *bl([
        "<b>Level bias of the validation year.</b> In June 2024 – April 2025 the base model's total was 1.013× actual, "
        "while CatBoost's was 0.997×. Much of CatBoost's local advantage was simply a lower level. On the evaluation year the two levels "
        "coincided (ratio 0.998), and lowering the level by 1.6% (#8) cost 2.5 M KRW.",
        "<b>Residual model effect.</b> After rescaling every model to its oracle level per fold, CatBoost still led locally (−3.1%). "
        "Combined with its degradation for recent cutoffs, this suggests it fitted that particular year.",
        "<b>Hidden level effect of a ‘neutral’ correction.</b> The median-shape correction kept the total unchanged, but it lowered "
        "clear-sky hours, which carry most of the weight, and raised cloudy ones. We multiplied each local block's base forecast by c "
        "before applying it (Fig. 3). In two of three blocks the gain turned into a loss once the base was under-forecast like the "
        "evaluation year (+1.7%, +0.7%), bracketing the server's +0.57%. A check across six different models had passed, but all "
        "six were measured in the same year. We verified transfer across models, not across years."])]
    s += [KeepTogether([fig_level(), P("Figure 3. Effect of the total-preserving correction as a function of the base forecast level, per held-out year block. "
                                       "Dots mark the level at which a 1.6% down-scaling costs +1.8%, as observed on the server.", cap)])]

    s += [P("6. Comparison with verified top teams", h1), *bl([
        "<b>Source.</b> After the competition the organisers published per-submission error rates (hours with capacity factor ≥ 10%; "
        "formula not disclosed) for the 17 teams asked to submit code, and method summaries for the two teams whose code had been verified. "
        "Ranks 6–17 converged at 3.0–3.3%. The top three dropped abruptly in a single late submission and were under review, so we exclude them.",
        "<b>Not information alone.</b> Both verified teams used three external weather sources with publication-time audits, as we did, "
        "and an ensemble mixing ML and DL models. We used tree models only. Assuming the error rate scales with the penalty, ours is about 3.6%, "
        "12–18% above the converged group. The gap likely comes from model structure as well as information; the notice does not let us separate the two.",
        "<b>Composition matters for rank.</b> Some teams with higher error ranked above teams with lower error, by 1–2%. This matches our estimate of the "
        "composition margin (−1.3% to −2.7% vs. the reference split): too small to explain the gap to first place, but worth dozens of ranks in a dense leaderboard. "
        "One verified team searched compositions with cross-validated errors from several training periods, which addresses why our single-window search did not transfer."])]

    s += [P("7. Lessons", h1), *bl([
        "Split out-of-sample validation across several yearly blocks. One year's level bias can contaminate every decision.",
        "Before accepting a post-hoc correction or model swap, re-measure it after shifting the base forecast level by ±4%.",
        "Total-neutral is not penalty-neutral: changes concentrated in high-weight hours carry a level effect.",
        "Large gains came from new information (features, NWP; +42 M KRW). Blending, tuning and corrections were worth ≤ 1.6 M KRW and were directionally unstable.",
        "Better information is necessary but not sufficient. Design for day-level structure (daily total, then hourly profile) early, "
        "and add DL models that differ from trees to the ensemble.",
        "Validate composition search on out-of-sample errors from several periods, not one window."])]

    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=17*mm, bottomMargin=17*mm,
                      title="OIBC 2026 technical report", author="[Author name]").build(s)
    print(OUT)


if __name__ == "__main__":
    main()
