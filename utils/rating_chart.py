import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

ratings = [
    # (label, pct_mid, pct_display, min_reviews, color, section)
    ("Overwhelmingly positive", 97.5, "95–100%", "500+", "#22c55e", "High volume (500+ reviews)"),
    ("Very positive",           87.0, "80–94%",  "500+", "#4ade80", "High volume (500+ reviews)"),
    ("Positive",                90.0, "80–100%", "10–499", "#86efac", "Low volume (10–499 reviews)"),
    ("Mostly positive",         74.5, "70–79%",  "10–499", "#fbbf24", "Low volume (10–499 reviews)"),
    ("Mixed",                   54.5, "40–69%",  "10–499", "#a1a1aa", "Low volume (10–499 reviews)"),
    ("Mostly negative (low)",   32.0, "25–39%",  "10–499", "#f87171", "Low volume (10–499 reviews)"),
    ("Negative",                12.0, "0–24%",   "10–499", "#ef4444", "Low volume (10–499 reviews)"),
    ("Mostly negative",         32.0, "25–39%",  "500+", "#f87171", "High volume negative (500+)"),
    ("Very negative",           17.5, "11–24%",  "500+", "#ef4444", "High volume negative (500+)"),
    ("Overwhelmingly negative",  5.0, "0–10%",   "500+", "#dc2626", "High volume negative (500+)"),
]

sections = [
    ("High volume (500+ reviews)",       [0, 1]),
    ("Low volume (10–499 reviews)",      [2, 3, 4, 5, 6]),
    ("High volume negative (500+)",      [7, 8, 9]),
]

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#0f0f0f")
ax.set_facecolor("#0f0f0f")

BAR_HEIGHT = 0.38
GAP = 0.18
SECTION_GAP = 0.55
LABEL_X = 0.0
BAR_X = 2.55
BAR_W = 5.5
PCT_X = BAR_X + BAR_W + 0.15
REV_X = PCT_X + 1.05

y = 0
y_positions = []
section_spans = []

for sec_name, indices in sections:
    sec_start_y = y
    for i, idx in enumerate(indices):
        y_positions.append(y)
        y += BAR_HEIGHT + GAP
    section_spans.append((sec_name, sec_start_y, y - GAP))
    y += SECTION_GAP

total_height = y
ax.set_xlim(0, 10)
ax.set_ylim(-0.3, total_height + 0.3)
ax.invert_yaxis()
ax.axis("off")

# Column headers
header_y = -0.18
ax.text(LABEL_X, header_y, "Rating", fontsize=8.5, color="#6b7280",
        fontweight="bold", va="center", ha="left")
ax.text(BAR_X + BAR_W / 2, header_y, "Positive review %", fontsize=8.5,
        color="#6b7280", fontweight="bold", va="center", ha="center")
ax.text(PCT_X + 0.45, header_y, "Range", fontsize=8.5, color="#6b7280",
        fontweight="bold", va="center", ha="center")
ax.text(REV_X + 0.35, header_y, "Min reviews", fontsize=8.5, color="#6b7280",
        fontweight="bold", va="center", ha="center")

# Section dividers + labels
for sec_name, y_start, y_end in section_spans:
    mid_y = (y_start + y_end) / 2
    ax.plot([LABEL_X - 0.05, LABEL_X - 0.05], [y_start, y_end],
            color="#374151", linewidth=2, solid_capstyle="round")
    ax.text(LABEL_X - 0.12, mid_y, sec_name, fontsize=7.5, color="#6b7280",
            va="center", ha="right", rotation=90)

# Draw each row
for pos_idx, (label, pct_mid, pct_display, min_rev, color, _) in enumerate(ratings):
    y_pos = y_positions[pos_idx]
    center_y = y_pos + BAR_HEIGHT / 2

    # Badge background
    badge_bg = FancyBboxPatch(
        (LABEL_X, y_pos), 2.45, BAR_HEIGHT,
        boxstyle="round,pad=0.03",
        facecolor=color + "22", edgecolor=color + "55", linewidth=0.6
    )
    ax.add_patch(badge_bg)

    # Dot + label
    ax.plot(LABEL_X + 0.13, center_y, "o", color=color, markersize=5)
    clean_label = label.replace(" (low)", "")
    ax.text(LABEL_X + 0.28, center_y, clean_label, fontsize=8.5,
            color=color, va="center", ha="left", fontweight="400")

    # Bar background
    bar_bg = FancyBboxPatch(
        (BAR_X, y_pos + BAR_HEIGHT * 0.25), BAR_W, BAR_HEIGHT * 0.5,
        boxstyle="round,pad=0.01",
        facecolor="#1f2937", edgecolor="none"
    )
    ax.add_patch(bar_bg)

    # Bar fill
    fill_w = BAR_W * pct_mid / 100
    bar_fill = FancyBboxPatch(
        (BAR_X, y_pos + BAR_HEIGHT * 0.25), fill_w, BAR_HEIGHT * 0.5,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor="none", alpha=0.85
    )
    ax.add_patch(bar_fill)

    # Percentage range
    ax.text(PCT_X + 0.45, center_y, pct_display, fontsize=8.5,
            color=color, va="center", ha="center", fontweight="400")

    # Min reviews
    ax.text(REV_X + 0.35, center_y, min_rev, fontsize=8.5,
            color="#9ca3af", va="center", ha="center")

# Title + footnote
ax.text(5.0, -0.55, "Steam Review Rating System",
        fontsize=13, color="#f9fafb", fontweight="bold",
        va="center", ha="center")

ax.text(5.0, total_height + 0.12,
        "Games with fewer than 10 reviews receive no label — only the raw score is shown.",
        fontsize=7.5, color="#6b7280", va="center", ha="center", style="italic")

plt.tight_layout(pad=0.5)
plt.savefig("../plots/steam_rating_system.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: steam_rating_system.png")