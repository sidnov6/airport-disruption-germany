"""Figure panels for the micropublication.

Run from the repository root after the notebooks have written outputs/:

    python figures/make_figures.py

Panels are saved without titles or panel letters (the micropublication guidelines
ask for those to be given in the text).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
DATA = ROOT / "data" / "processed" / "airport_day.parquet"

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
LOGISTIC = "#2a78d6"   # categorical slot 1
LIGHTGBM = "#eb6834"   # slot 2
HISTORY = "#1baf7a"    # slot 3
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#f0efec"
RED = "#e34948"
MASKED = "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
})

WEATHER = {"Thunderstorm": "thunder_hours", "Low visibility": "low_vis_hours_lt1000",
           "Strong gust": "gust_hours_ge15", "Snow": "snow_hours", "Freezing": "freezing_hours"}
MIN_EVENTS = 10


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load_predictions():
    lr = pd.read_csv(ROOT / "outputs" / "03_logistic_regression" / "test_predictions_2024_2025.csv")
    gb = pd.read_csv(ROOT / "outputs" / "04_lightgbm_classification" / "test_predictions_2024_2025.csv")
    assert (lr["actual"].to_numpy() == gb["actual"].to_numpy()).all()
    return lr, gb


def panel_pr_curves(lr, gb):
    y = lr["actual"]
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    for name, score, color in [("Logistic regression", lr["probability"], LOGISTIC),
                               ("LightGBM", gb["probability"], LIGHTGBM),
                               ("Previous 28-day rate", lr["benchmark_28d"], HISTORY)]:
        p, r, _ = precision_recall_curve(y, score)
        ax.plot(r, p, color=color, linewidth=2, label=f"{name} (AP {average_precision_score(y, score):.2f})")
    ax.axhline(y.mean(), color="#8a8984", linestyle="--", linewidth=1, label=f"Base rate ({y.mean():.3f})")
    ax.set_xlabel("Recall (share of disrupted days flagged)")
    ax.set_ylabel("Precision (share of flags that were disrupted)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    save(fig, "panelB_precision_recall_2024_2025.png")


def panel_airport_ap(lr, gb):
    rows = []
    for icao, g in lr.groupby("airport_icao"):
        n = int(g["actual"].sum())
        if n < 20:
            continue
        g2 = gb[gb["airport_icao"] == icao]
        rows.append({"airport": f"{g['airport_name'].iloc[0]}\n({n} days)", "n": n,
                     "Logistic regression": average_precision_score(g["actual"], g["probability"]),
                     "LightGBM": average_precision_score(g2["actual"], g2["probability"]),
                     "Previous 28-day rate": average_precision_score(g["actual"], g["benchmark_28d"])})
    t = pd.DataFrame(rows).sort_values("n", ascending=False)
    x = np.arange(len(t))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k, (col, color) in enumerate([("Logistic regression", LOGISTIC), ("LightGBM", LIGHTGBM),
                                      ("Previous 28-day rate", HISTORY)]):
        ax.bar(x + (k - 1) * w, t[col], width=w - 0.03, color=color, label=col)
    ax.set_xticks(x, t["airport"], fontsize=9)
    ax.set_ylabel("Average precision, 2024-2025")
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    save(fig, "extra_average_precision_by_airport.png")


def panel_weather_risk(df):
    ratios = pd.DataFrame(index=sorted(df["airport_name"].unique()), columns=list(WEATHER), dtype=float)
    counts = ratios.copy()
    for airport, g in df.groupby("airport_name"):
        for name, col in WEATHER.items():
            obs = g[g[col].notna()]
            flag = obs[col] > 0
            y = obs["disrupted_day"].astype(float)
            with_w, without_w = y[flag].mean(), y[~flag].mean()
            counts.loc[airport, name] = y[flag].sum()
            if y[flag].sum() >= MIN_EVENTS and without_w > 0:
                ratios.loc[airport, name] = with_w / without_w
    order = ratios.notna().sum(axis=1).sort_values(ascending=False).index
    ratios, counts = ratios.loc[order], counts.loc[order]
    log_ratio = np.log2(ratios.astype(float))
    cmap = LinearSegmentedColormap.from_list("div", [LOGISTIC, NEUTRAL, RED]).with_extremes(bad=MASKED)
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=2.5)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    im = ax.imshow(np.ma.masked_invalid(log_ratio.values), cmap=cmap, norm=norm, aspect="auto")
    for i in range(ratios.shape[0]):
        for j in range(ratios.shape[1]):
            v = ratios.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1f}×", ha="center", va="center", fontsize=6.2, color=INK)
            else:
                ax.text(j, i, f"n={int(counts.iloc[i, j])}", ha="center", va="center", fontsize=5.2, color=INK_2)
    ax.set_xticks(range(ratios.shape[1]), ["Thunder-\nstorm", "Low\nvisibility", "Strong\ngust", "Snow", "Freezing"], fontsize=6.2)
    ax.set_yticks(range(ratios.shape[0]), ratios.index, fontsize=6.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    ticks = [0.5, 1, 2, 4]
    cb.set_ticks(np.log2(ticks), labels=[f"{t:g}×" for t in ticks])
    cb.set_label("Risk ratio (with / without condition)", fontsize=6.2)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    save(fig, "panelC_weather_risk_ratio_by_airport.png")


def panel_traffic_weather(df):
    d = df.copy()
    observed = d[list(WEATHER.values())].notna().all(axis=1)
    d = d[observed & d["disrupted_day"].notna()]
    median = d.groupby(["airport_icao", "year"])["arrivals"].transform("median")
    d["traffic"] = pd.cut(d["arrivals"] / median, [-np.inf, 0.8, 1.0, 1.2, np.inf],
                          labels=["<80%", "80-100%", "100-120%", ">120%"])
    hazards = sum((d[c] > 0).astype(int) for c in WEATHER.values())
    d["hazards"] = pd.cut(hazards, [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    rate = d.groupby(["hazards", "traffic"], observed=True)["disrupted_day"].mean().unstack() * 100
    n = d.groupby(["hazards", "traffic"], observed=True).size().unstack()
    shown = rate.where(n >= 50)
    cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP).with_extremes(bad=MASKED)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    im = ax.imshow(np.ma.masked_invalid(shown.values.astype(float)), cmap=cmap, vmin=0, vmax=30, aspect="auto")
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            v = shown.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1f}%\nn={int(n.iloc[i, j]):,}", ha="center", va="center", fontsize=5.8,
                        color="white" if v > 18 else INK)
            else:
                ax.text(j, i, f"n={int(n.iloc[i, j])}", ha="center", va="center", fontsize=5.4, color=INK_2)
    ax.set_xticks(range(shown.shape[1]), shown.columns, fontsize=6.2)
    ax.set_yticks(range(shown.shape[0]), shown.index, fontsize=6.2)
    ax.set_xlabel("Arrivals relative to the airport's median day in the same year", fontsize=6.4)
    ax.set_ylabel("Adverse weather conditions", fontsize=6.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Disrupted airport-days (%)", fontsize=6.2)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    save(fig, "panelD_disruption_by_traffic_and_weather.png")


def panel_pipeline():
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, head, body, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=fill, edgecolor="#8a8984", linewidth=0.8))
        ax.text(x + w / 2, y + h - 2.2, head, ha="center", va="top", fontsize=7.2, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=5.9, color=INK_2, linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK_2, linewidth=1))

    box(1, 22, 22, 16, "EUROCONTROL", "apt_dly_2014..2025.csv.bz2\n12 yearly files, all airports\narrivals, ATFM delay,\n16 delay causes", "#cde2fb")
    box(1, 2, 22, 16, "DWD open data", "hourly station records\n6 products x 12 stations\ntemp, wind, gust, rain,\nvisibility, weather codes", "#cde2fb")
    box(28, 22, 21, 16, "Clean traffic", "keep 12 airports\nblank delay = 0\ncauses sum to total\n38 missing days kept", NEUTRAL)
    box(28, 2, 21, 16, "Clean weather", "-999 = missing, UTC checked\nhourly -> daily (>= 20 h)\nWMO code groups:\nfog, thunder, snow, freezing", NEUTRAL)
    box(54, 8, 20, 24, "airport_day table", "12 airports x 4,383 days\n= 52,596 rows, 79 columns\n\ntarget: >= 2% of arrivals\ndelayed > 15 min\n\nhistory features use\nearlier days only", "#ffe0d0")
    box(79, 22, 20, 16, "BI notebooks", "KPIs as ratios of sums\nweather risk ratios\ntraffic x weather", NEUTRAL)
    box(79, 2, 20, 16, "Models", "train 2014-2021\nselect 2022, calibrate 2023\ntest 2024-2025 (once)\n37 contract predictors", NEUTRAL)
    for y in (30, 10):
        arrow(23.8, y, 27.2, y)
    arrow(49.8, 30, 53.2, 24)
    arrow(49.8, 10, 53.2, 16)
    arrow(74.8, 24, 78.2, 30)
    arrow(74.8, 16, 78.2, 10)
    save(fig, "panelE_data_pipeline.png")


AIRPORT_LABELS = {  # display name, label offset in points (dx, dy)
    "EDDF": ("Frankfurt", (17, -4)), "EDDM": ("Munich", (15, -11)), "EDDL": ("Düsseldorf", (-8, 4)),
    "EDDH": ("Hamburg", (9, 3)), "EDDK": ("Cologne-Bonn", (-8, -9)), "EDDS": ("Stuttgart", (-8, -7)),
    "EDDP": ("Leipzig-Halle", (8, 3)), "EDDV": ("Hannover", (7, -7)), "EDDN": ("Nuremberg", (7, 3)),
    "EDDW": ("Bremen", (-7, 4)), "EDDC": ("Dresden", (7, -6)), "EDDG": ("Münster-Osnabrück", (-7, 6)),
}


def panel_map(df):
    import json
    import matplotlib.patheffects as pe
    from matplotlib.patches import Polygon
    import yaml

    geo = json.loads((ROOT / "data" / "geo" / "germany_states_neighbours.geojson").read_text())
    airports = pd.DataFrame(yaml.safe_load((ROOT / "pipeline" / "config.yaml").read_text())["airports"])
    stats = df.groupby("airport_icao").agg(arrivals=("arrivals", "sum"), rate=("disrupted_day", "mean"))
    airports = airports.join(stats, on="icao")

    fig, ax = plt.subplots(figsize=(3.6, 4.4))
    ax.set_facecolor("#eef3f8")  # sea
    for f in geo["features"]:
        state = f["properties"]["layer"] == "state"
        for poly in f["geometry"]["coordinates"]:
            ax.add_patch(Polygon(poly[0], closed=True, facecolor="#ffffff" if state else "#ecebe6",
                                 edgecolor="#c9c8c2" if state else "#ffffff", linewidth=0.5 if state else 0.8,
                                 zorder=2 if state else 1))
    ax.set_xlim(5.5, 15.4)
    ax.set_ylim(47.15, 55.15)
    ax.set_aspect(1 / np.cos(np.radians(51.2)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ramp = LinearSegmentedColormap.from_list("disruption", ["#fbe3d6", "#f4a07a", "#eb6834", "#b8461a", "#7a2a0b"])
    size = lambda a: 640 * a / 2.6e6  # noqa: E731  marker area proportional to arrivals
    sc = ax.scatter(airports["lon"], airports["lat"], s=size(airports["arrivals"]), c=airports["rate"] * 100,
                    cmap=ramp, vmin=0, vmax=20, edgecolors="#ffffff", linewidths=0.8, zorder=4)
    halo = [pe.withStroke(linewidth=2.2, foreground="#ffffff")]
    for _, a in airports.iterrows():
        name, (dx, dy) = AIRPORT_LABELS[a["icao"]]
        ha = "left" if dx >= 0 else "right"
        ax.annotate(f"{name}\n{a['rate'] * 100:.1f}%", (a["lon"], a["lat"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=6.6, color=INK,
                    linespacing=1.05, path_effects=halo, zorder=5)

    # nested size legend: circles share a bottom edge
    fig.canvas.draw()
    pts_per_deg = (ax.transData.transform((0, 1)) - ax.transData.transform((0, 0)))[1] * 72 / fig.dpi
    x0, base = 14.1, 53.9
    for val, lab in [(2.5e6, "2.5 M"), (1.0e6, "1 M"), (0.25e6, "0.25 M")]:
        r = np.sqrt(size(val)) / 2 / pts_per_deg  # scatter s is diameter squared in points
        ax.scatter(x0, base + r, s=size(val), facecolor="none", edgecolor=INK_2, linewidth=0.7, zorder=4)
        ax.plot([x0, x0 + 0.5], [base + 2 * r] * 2, color=INK_2, linewidth=0.5, zorder=4)
        ax.text(x0 + 0.55, base + 2 * r, lab, va="center", fontsize=5.6, color=INK_2, zorder=5)
    ax.text(x0 + 0.2, 54.95, "Arrivals 2014-2025", ha="center", fontsize=5.8, color=INK_2, zorder=5)
    cax = ax.inset_axes([0.06, 0.05, 0.36, 0.022])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 5, 10, 15, 20], labels=["0", "5", "10", "15", "20%"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.8, length=2, colors=INK_2)
    cax.set_title("Disrupted airport-days, 2014-2025", fontsize=5.8, color=INK_2, loc="left", pad=2)
    ax.text(0.99, 0.005, "Boundaries: Natural Earth", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=4.6, color="#8a8984", zorder=5)
    save(fig, "panelA_airport_map.png")


def main():
    lr, gb = load_predictions()
    df = pd.read_parquet(DATA)
    df["disrupted_day"] = df["disrupted_day"].astype("float64")
    for c in ["arrivals", *WEATHER.values()]:
        df[c] = df[c].astype("float64")
    panel_map(df)
    panel_pr_curves(lr, gb)
    panel_airport_ap(lr, gb)
    panel_weather_risk(df)
    panel_traffic_weather(df)
    panel_pipeline()


if __name__ == "__main__":
    main()
