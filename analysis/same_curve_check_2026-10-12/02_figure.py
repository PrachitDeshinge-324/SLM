"""Figure for 01_same_curve.py: reads outputs/01_same_curve.json and writes outputs/same_curve.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(HERE, "outputs", "01_same_curve.json")))

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE16, BLUE4 = "#184f95", "#5598e7"   # one hue, two steps: 16-bit -> 4-bit
ACCENT = "#2a78d6"                     # the contrast the figure is about; the yardsticks stay grey
NAMES = {"Qwen3.5-0.8B": "Qwen3.5-0.8B", "Qwen3.5-2B": "Qwen3.5-2B", "DS-R1-1.5B": "DeepSeek-R1-1.5B",
         "gemma-3-1b-it": "Gemma-3-1B", "Llama-1B": "Llama-3.2-1B", "Llama-3B": "Llama-3.2-3B"}
# label offsets in points from each model's 16-bit dot (three models share the top-right corner)
OFFSET = {"Qwen3.5-0.8B": (12, -4, "left"), "gemma-3-1b-it": (-10, 10, "right"), "Llama-1B": (22, -40, "left"),
          "Qwen3.5-2B": (-14, 14, "right"), "DS-R1-1.5B": (-2, 24, "center"), "Llama-3B": (16, -16, "left")}

plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.linewidth": 0.8})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 6.3), dpi=200, gridspec_kw={"width_ratios": [1, 1.12], "wspace": 0.16})
fig.patch.set_facecolor(SURFACE)
for a in (ax, bx):
    a.set_facecolor(SURFACE)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(length=0)
    a.set_axisbelow(True)


def heading(a, title, subtitle):
    a.text(0, 1.13, title, transform=a.transAxes, fontsize=12.5, fontweight="bold", color=INK, va="bottom")
    a.text(0, 1.04, subtitle, transform=a.transAxes, fontsize=9.5, color=INK2, va="bottom", linespacing=1.35)


# ---------------------------------------------------------------- left: along the curve
heading(ax, "4-bit models land on the line the 16-bit models trace",
        "GSM8K, six models, 8 samples per question. Fewer confident labels at 4-bit,\nbut only as many fewer as the drop in accuracy implies.")
x0, x1 = 24, 78
line = lambda x: 100 * (F["line"]["intercept"] + F["line"]["slope"] * x / 100)
ax.plot([x0, x1], [line(x0), line(x1)], color=AXIS, lw=1.2, solid_capstyle="round", zorder=1)
ax.annotate("line through the six\n16-bit models", (59, line(59)), xytext=(22, -30), textcoords="offset points",
            fontsize=9, color=MUTED, ha="left", va="top", arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})
for c in F["curve"]:
    a16, a4 = c["16bit"], c["4bit"]
    ax.plot([100 * a16["x"], 100 * a4["x"]], [100 * a16["yield"], 100 * a4["yield"]], color=BLUE4, lw=2, solid_capstyle="round", zorder=2)
for prec, color, label in (("4bit", BLUE4, "4-bit"), ("16bit", BLUE16, "16-bit")):
    ax.scatter([100 * c[prec]["x"] for c in F["curve"]], [100 * c[prec]["yield"] for c in F["curve"]],
               s=62, color=color, edgecolors=SURFACE, linewidths=1.6, zorder=3, label=label)
for c in F["curve"]:
    dx, dy, ha = OFFSET[c["model"]]
    ax.annotate(NAMES[c["model"]], (100 * c["16bit"]["x"], 100 * c["16bit"]["yield"]), xytext=(dx, dy), textcoords="offset points",
                fontsize=9.5, color=INK, ha=ha, va="center",
                arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8, "shrinkA": 1, "shrinkB": 5} if abs(dy) > 12 else None)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2, handletextpad=0.3, borderaxespad=0.6)
ax.set_xlim(x0, x1)
ax.set_ylim(14, 80)
ax.grid(color=GRID, lw=0.8)
ax.set_xlabel("Single-sample accuracy (%)", labelpad=8)
ax.set_ylabel("Questions with a confident label (%)\nat least 6 of 8 samples agree", labelpad=8)

# ---------------------------------------------------------------- right: off its own curve?
heading(bx, "4-bit rarely moves the calibration curve; switching model does",
        "Change in how often the voted answer is right at the same agreement level,\nin points, with 95% intervals. Zero means the calibration curve did not move.")
GROUPS = [("4bit", "Same model, 4-bit against 16-bit", ACCENT),
          ("rerun", "Same model and precision, run twice (noise floor)", MUTED),
          ("models", "Two different models, both at 16-bit", MUTED)]
y = 0.0
for key, title, color in GROUPS:
    rows = sorted(F["offcurve"][key], key=lambda r: -r["est"])
    bx.text(-39.5, y + 0.15, f"{title}  ·  {len(rows)} {'cells' if key == '4bit' else 'pairs'}", fontsize=9.5, fontweight="bold", color=INK, va="center")
    y -= 1.25
    for r in rows:
        bx.plot([r["lo"], r["hi"]], [y, y], color=color, lw=1.6, solid_capstyle="round", zorder=2, alpha=0.75)
        bx.scatter([r["est"]], [y], s=46, color=color, edgecolors=SURFACE, linewidths=1.4, zorder=3)
        if key == "4bit" and "gemma" in r["label"]:
            bx.text(r["lo"] - 1.5, y, "Gemma-3-1B, whose answer\nformat changed at 4-bit", fontsize=9, color=INK2, va="center", ha="right", linespacing=1.3)
        y -= 1.0
    y -= 1.9
bx.axvline(0, color=MUTED, lw=1.0, zorder=1)
bx.set_xlim(-40, 20)
bx.set_ylim(y + 2.2, 1.1)
bx.set_yticks([])
bx.spines["left"].set_visible(False)
bx.grid(axis="x", color=GRID, lw=0.8)
bx.set_xlabel("Points of majority-vote accuracy at the same agreement level", labelpad=8)

fig.text(0.062, 0.012, "bitsandbytes 8/4-bit, stored runs. Intervals: paired bootstrap over questions; cells with 100-200 questions have wide intervals. "
         "Numbers: outputs/01_same_curve.txt", fontsize=8, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.062, right=0.985, top=0.83, bottom=0.125)
fig.savefig(os.path.join(HERE, "outputs", "same_curve.png"), facecolor=SURFACE)
