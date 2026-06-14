import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
RECOMMENDATIONS_CSV = "../dataset/recommendations.csv"
GAMES_CSV           = "../dataset/games.csv"
OUTPUT_CSV          = "../dataset/game_popularity.csv"

LAUNCH_WINDOW_DAYS  = 30    # days after release date (or first review for legacy games)
LATEST_WINDOW_DAYS  = 30    # days before the last review in the dataset

# Reviews are always computed regardless of count.
# These thresholds only drive the confidence flag columns.
LOW_REVIEWS_THRESHOLD = 10  # below this → flagged as low confidence

# Steam started collecting reviews around this date.
# Games released before it are flagged as "pre-review-system".
STEAM_REVIEWS_EPOCH = pd.Timestamp("2013-10-01")

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading data...")
recs  = pd.read_csv(RECOMMENDATIONS_CSV, parse_dates=["date"])
games = pd.read_csv(GAMES_CSV, parse_dates=["date_release"])

# ── MERGE release date into reviews ───────────────────────────────────────────
recs = recs.merge(games[["app_id", "date_release", "title"]], on="app_id", how="left")

missing_release = recs["date_release"].isna().sum()
if missing_release:
    print(f"Dropping {missing_release:,} reviews with no release date")
recs = recs.dropna(subset=["date_release"])

# ── GLOBAL LATEST DATE ────────────────────────────────────────────────────────
dataset_end = recs["date"].max()
print(f"Dataset spans up to: {dataset_end.date()}")

# ── COMPUTE PER-GAME WINDOWS ──────────────────────────────────────────────────
print("Computing popularity windows...")

def window_stats(group, start, end):
    """Return positive ratio and review count for reviews in [start, end].
    Always returns a ratio — even for 1 review. Count drives confidence flag."""
    mask   = (group["date"] >= start) & (group["date"] <= end)
    subset = group.loc[mask]
    n      = len(subset)
    if n == 0:
        return pd.Series({"positive_ratio": np.nan, "review_count": 0})
    ratio = subset["is_recommended"].mean()
    return pd.Series({"positive_ratio": round(ratio, 4), "review_count": n})

records = []
grouped = recs.groupby("app_id")

for app_id, group in grouped:
    release_date = group["date_release"].iloc[0]
    title        = group["title"].iloc[0]
    first_review = group["date"].min()
    last_review  = group["date"].max()

    # ── pre-review-system flag ────────────────────────────────────────────────
    pre_review_system = release_date < STEAM_REVIEWS_EPOCH

    # ── Beta window ───────────────────────────────────────────────────────────
    # Reviews that predate the official release date (e.g. early access / beta).
    has_beta_reviews = first_review < release_date
    if has_beta_reviews:
        beta_end = release_date - pd.Timedelta(days=1)
        beta     = window_stats(group, first_review, beta_end)
    else:
        beta = pd.Series({"positive_ratio": np.nan, "review_count": 0})

    # ── Launch window ─────────────────────────────────────────────────────────
    # For games released before Steam reviews existed, fall back to the first
    # review date so legacy titles still get a meaningful launch score.
    if first_review > release_date + pd.Timedelta(days=LAUNCH_WINDOW_DAYS):
        launch_start = first_review
    else:
        launch_start = release_date
    launch_end = launch_start + pd.Timedelta(days=LAUNCH_WINDOW_DAYS)
    launch     = window_stats(group, launch_start, launch_end)

    # ── Latest window ─────────────────────────────────────────────────────────
    latest_start = dataset_end - pd.Timedelta(days=LATEST_WINDOW_DAYS)
    latest       = window_stats(group, latest_start, dataset_end)

    # ── Overall ───────────────────────────────────────────────────────────────
    total_reviews = len(group)
    overall_ratio = round(group["is_recommended"].mean(), 4)

    records.append({
        "app_id":                   app_id,
        "title":                    title,
        "release_date":             release_date.date(),
        "pre_review_system":        pre_review_system,   # True = released before Oct 2013
        "first_review_date":        first_review.date(),
        "last_review_date":         last_review.date(),
        "total_reviews":            total_reviews,
        "overall_positive_ratio":   overall_ratio,
        # Beta window (reviews before official release date)
        "has_beta_reviews":         has_beta_reviews,
        "beta_reviews":             int(beta["review_count"]),
        "beta_positive_ratio":      beta["positive_ratio"],
        "beta_low_confidence":      beta["review_count"] < LOW_REVIEWS_THRESHOLD,
        # Launch window
        "launch_reviews":           int(launch["review_count"]),
        "launch_positive_ratio":    launch["positive_ratio"],
        "launch_low_confidence":    launch["review_count"] < LOW_REVIEWS_THRESHOLD,
        # Latest window
        "latest_reviews":           int(latest["review_count"]),
        "latest_positive_ratio":    latest["positive_ratio"],
        "latest_low_confidence":    latest["review_count"] < LOW_REVIEWS_THRESHOLD,
    })

df = pd.DataFrame(records)

# ── DERIVED COLUMNS ───────────────────────────────────────────────────────────
df["popularity_trend"] = (df["latest_positive_ratio"] - df["launch_positive_ratio"]).round(4)

def trend_label(row):
    if pd.isna(row["launch_positive_ratio"]) or pd.isna(row["latest_positive_ratio"]):
        return "no data"
    delta = row["popularity_trend"]
    if delta >  0.10: return "strong improvement"
    if delta >  0.02: return "slight improvement"
    if delta < -0.10: return "strong decline"
    if delta < -0.02: return "slight decline"
    return "stable"

df["trend_label"] = df.apply(trend_label, axis=1)

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}  ({len(df):,} games)")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
comparable = df.dropna(subset=["launch_positive_ratio", "latest_positive_ratio"])

print(f"\n── Coverage ─────────────────────────────────────────")
print(f"Total games:                     {len(df):,}")
print(f"Pre-review-system (old) games:   {df['pre_review_system'].sum():,}")
print(f"Games with beta reviews:         {df['has_beta_reviews'].sum():,}")
print(f"  of which low confidence:       {df.loc[df['has_beta_reviews'], 'beta_low_confidence'].sum():,}")
print(f"Games with launch data:          {df['launch_positive_ratio'].notna().sum():,}")
print(f"  of which low confidence:       {df['launch_low_confidence'].sum():,}")
print(f"Games with latest data:          {df['latest_positive_ratio'].notna().sum():,}")
print(f"  of which low confidence:       {df['latest_low_confidence'].sum():,}")
print(f"Games comparable (both windows): {len(comparable):,}")

print(f"\n── Trend distribution (comparable games) ───────────")
print(df["trend_label"].value_counts().to_string())

print(f"\n── Top 10 most improved games ──────────────────────")
top_improved = (comparable
.sort_values("popularity_trend", ascending=False)
.head(10)[["title", "pre_review_system", "launch_positive_ratio",
           "latest_positive_ratio", "popularity_trend", "total_reviews",
           "launch_low_confidence", "latest_low_confidence"]])
print(top_improved.to_string(index=False))

print(f"\n── Top 10 most declined games ──────────────────────")
top_declined = (comparable
.sort_values("popularity_trend", ascending=True)
.head(10)[["title", "pre_review_system", "launch_positive_ratio",
           "latest_positive_ratio", "popularity_trend", "total_reviews",
           "launch_low_confidence", "latest_low_confidence"]])
print(top_declined.to_string(index=False))