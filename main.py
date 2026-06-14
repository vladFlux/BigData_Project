import pandas as pd
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

# ── CONFIG ────────────────────────────────────────────────────────────────────
WEIGHTED_RECS_CSV = "./dataset/recommendations_weighted.csv"
USER_FACTORS_CSV  = "./dataset/als_user_factors.csv"
ITEM_FACTORS_CSV  = "./dataset/als_item_factors.csv"

ALPHA       = 4.0   # confidence scaling: confidence = 1 + ALPHA * review_weight
N_FACTORS   = 64
REGULARIZATION = 0.05
ITERATIONS  = 20

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading weighted recommendations...")
df = pd.read_csv(WEIGHTED_RECS_CSV)  # expects: user_id, app_id, is_recommended, review_weight, ...

required_cols = {"user_id", "app_id", "is_recommended", "review_weight"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print(f"Loaded {len(df):,} rows")

# ── BUILD CONFIDENCE MATRIX VALUES ────────────────────────────────────────────
# Implicit ALS expects a "confidence" matrix C, where the model fits toward
# preference p_ui in {0, 1} with confidence c_ui = 1 + alpha * r_ui.
#
# This dataset has no neutral interactions: every row is either a thumbs-up
# (is_recommended=True, preference=1) or a thumbs-down (is_recommended=False,
# preference=0). To encode "confidently disliked" in a single matrix that
# implicit's solver can consume, we store NEGATIVE confidence values for
# disliked items:
#
#   recommended:     value = +(1 + ALPHA * review_weight)   (target preference 1)
#   not recommended: value = -(1 + ALPHA * review_weight)   (target preference 0,
#                                                              with the same magnitude
#                                                              of confidence)
#
# implicit's AlternatingLeastSquares is documented to treat the matrix as
# confidence values C_ui: positive entries mean "liked, with confidence
# = value", and negative entries mean "disliked, with confidence = abs(value)".
# This lets a single sparse matrix encode both "definitely liked" and
# "definitely disliked" signals, weighted by review_weight. Note: alpha=1.0
# is set on the model itself since the ALPHA scaling is already baked into
# confidence_value below — otherwise it would be applied twice.
df["confidence_value"] = np.where(
    df["is_recommended"].astype(bool),
    1.0 + ALPHA * df["review_weight"],
    -(1.0 + ALPHA * df["review_weight"]),
    )

print(f"\nConfidence value stats:")
print(df["confidence_value"].describe().round(4))

# ── BUILD SPARSE USER-ITEM MATRIX ─────────────────────────────────────────────
print("\nBuilding sparse matrix...")

user_ids = df["user_id"].astype("category")
item_ids = df["app_id"].astype("category")

df["user_idx"] = user_ids.cat.codes
df["item_idx"] = item_ids.cat.codes

n_users = df["user_idx"].nunique()
n_items = df["item_idx"].nunique()
print(f"Users: {n_users:,}  Items: {n_items:,}  Interactions: {len(df):,}")

# Duplicate (user, item) pairs: if a user reviewed the same game more than
# once, sum their confidence values (later overrides combine additively).
matrix = sp.csr_matrix(
    (df["confidence_value"], (df["user_idx"], df["item_idx"])),
    shape=(n_users, n_items),
)
matrix.sum_duplicates()

# ── TRAIN ALS ─────────────────────────────────────────────────────────────────
print("\nTraining ALS model...")
model = AlternatingLeastSquares(
    factors=N_FACTORS,
    regularization=REGULARIZATION,
    iterations=ITERATIONS,
    alpha=1.0,  # confidence scaling already applied via ALPHA when building confidence_value
)

# implicit expects a user-item matrix (rows=users, cols=items)
model.fit(matrix)

# ── SAVE FACTORS ──────────────────────────────────────────────────────────────
print("\nSaving factor matrices...")

user_factors_df = pd.DataFrame(model.user_factors)
user_factors_df.insert(0, "user_id", user_ids.cat.categories[np.arange(n_users)])
user_factors_df.to_csv(USER_FACTORS_CSV, index=False)

item_factors_df = pd.DataFrame(model.item_factors)
item_factors_df.insert(0, "app_id", item_ids.cat.categories[np.arange(n_items)])
item_factors_df.to_csv(ITEM_FACTORS_CSV, index=False)

print(f"Saved: {USER_FACTORS_CSV}  ({len(user_factors_df):,} users)")
print(f"Saved: {ITEM_FACTORS_CSV}  ({len(item_factors_df):,} items)")

# ── SAMPLE RECOMMENDATIONS ─────────────────────────────────────────────────────
print("\n── Sample: top 5 recommendations for first user ───────")
sample_user_idx = 0
sample_user_id  = user_ids.cat.categories[sample_user_idx]

recommended_ids, scores = model.recommend(
    sample_user_idx, matrix[sample_user_idx], N=5, filter_already_liked_items=True
)

item_categories = item_ids.cat.categories
for app_idx, score in zip(recommended_ids, scores):
    print(f"  user {sample_user_id} -> app_id {item_categories[app_idx]}  (score={score:.4f})")