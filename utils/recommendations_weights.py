import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
RECOMMENDATIONS_CSV = "../dataset/recommendations.csv"
USERS_CSV           = "../dataset/users.csv"
OUTPUT_CSV          = "../dataset/recommendations_weighted.csv"

# How much each component contributes to the final weight (must sum to 1.0)
WEIGHT_PLAYTIME     = 0.4
WEIGHT_GAMES_OWNED  = 0.4
WEIGHT_REVIEWS      = 0.2

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading data...")
recs  = pd.read_csv(RECOMMENDATIONS_CSV)   # app_id, helpful, funny, date, is_recommended, hours, user_id, review_id
users = pd.read_csv(USERS_CSV)             # user_id, products, reviews

# ── MERGE ─────────────────────────────────────────────────────────────────────
# Bring in each reviewer's total games owned
df = recs.merge(users[["user_id", "products", "reviews"]], on="user_id", how="left")

# ── PLAYTIME COMPONENT  →  log(hours + 1), normalized to [0, 1] ───────────────
# Cap at the 99th percentile before applying log — users with extreme hours
# (500h, 2000h, 10000h+) are likely biased and shouldn't outscore someone who
# played a reasonable amount. Anything above the cap is clipped before the log.
hours_cap         = df["hours"].quantile(0.99)
hours_capped      = df["hours"].clip(upper=hours_cap)
log_hours         = np.log1p(hours_capped)
max_log_hours     = np.log1p(hours_cap)
df["hours_score"] = log_hours / max_log_hours if max_log_hours > 0 else 0.0

print(f"Hours cap (99th pct): {hours_cap:.1f}h")

# ── GAMES OWNED COMPONENT  →  linear, normalized to [0, 1] ───────────────────
# Cap at the 99th percentile to prevent outlier bot/trader accounts
# (some users own 30,000+ games) from crushing everyone else's score.
# Anything above the cap is clipped to 1.0.
products_cap         = df["products"].quantile(0.99)
df["products_score"] = (df["products"] / products_cap).clip(upper=1.0)

print(f"Products cap (99th pct): {products_cap:.0f} games")

# ── REVIEWS EXPERIENCE COMPONENT  →  log(reviews + 1), normalized to [0, 1] ──
# Users with many past reviews are likely experienced curators/reviewers.
# Cap at the 99th percentile before applying log — prolific outlier accounts
# (1000+ reviews) shouldn't dominate. Anything above the cap is clipped
# before the log.
reviews_cap          = df["reviews"].quantile(0.99)
reviews_capped       = df["reviews"].clip(upper=reviews_cap)
log_reviews          = np.log1p(reviews_capped)
max_log_reviews      = np.log1p(reviews_cap)
df["reviews_score"]  = log_reviews / max_log_reviews if max_log_reviews > 0 else 0.0

print(f"Reviews cap (99th pct): {reviews_cap:.0f} reviews")

# ── FINAL WEIGHT ──────────────────────────────────────────────────────────────
df["review_weight"] = (
        WEIGHT_PLAYTIME    * df["hours_score"] +
        WEIGHT_GAMES_OWNED * df["products_score"] +
        WEIGHT_REVIEWS     * df["reviews_score"]
)

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")

# ── QUICK SANITY CHECK ────────────────────────────────────────────────────────
print("\n── Weight distribution ──────────────────────────────")
print(df["review_weight"].describe().round(4))

print("\n── Sample rows ──────────────────────────────────────")
print(df[["user_id", "hours", "products", "reviews", "hours_score",
          "products_score", "reviews_score", "review_weight"]].head(10).to_string(index=False))

print("\n── Extreme examples ─────────────────────────────────")
print("Lowest weight review:")
print(df.loc[df["review_weight"].idxmin(),
["user_id", "hours", "products", "reviews", "review_weight"]])
print("\nHighest weight review:")
print(df.loc[df["review_weight"].idxmax(),
["user_id", "hours", "products", "reviews", "review_weight"]])