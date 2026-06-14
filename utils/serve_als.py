import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
USER_FACTORS_CSV = "../dataset/als_user_factors.csv"
ITEM_FACTORS_CSV = "../dataset/als_item_factors.csv"
WEIGHTED_RECS_CSV = "../dataset/recommendations_weighted.csv"
GAMES_CSV         = "../dataset/games.csv"

TOP_N = 10

# ── LOAD FACTORS ───────────────────────────────────────────────────────────────
print("Loading factor matrices...")
user_factors_df = pd.read_csv(USER_FACTORS_CSV)
item_factors_df = pd.read_csv(ITEM_FACTORS_CSV)

# First column is the id, the rest are the latent factor dimensions
user_ids     = user_factors_df["user_id"].to_numpy()
item_ids     = item_factors_df["app_id"].to_numpy()
user_vectors = user_factors_df.drop(columns=["user_id"]).to_numpy()
item_vectors = item_factors_df.drop(columns=["app_id"]).to_numpy()

print(f"Users: {len(user_ids):,}  Items: {len(item_ids):,}  Factors: {user_vectors.shape[1]}")

# ── LOAD GAME TITLES (optional, for readable output) ───────────────────────────
try:
    games = pd.read_csv(GAMES_CSV)
    app_id_to_title = dict(zip(games["app_id"], games["title"]))
except FileNotFoundError:
    print(f"Note: {GAMES_CSV} not found, will show app_id only")
    app_id_to_title = {}

def title_for(app_id):
    return app_id_to_title.get(app_id, f"app_id {app_id}")

# ── LOAD INTERACTIONS (to exclude already-reviewed games from recommendations) ─
try:
    recs = pd.read_csv(WEIGHTED_RECS_CSV, usecols=["user_id", "app_id"])
    user_seen_items = recs.groupby("user_id")["app_id"].apply(set).to_dict()
except FileNotFoundError:
    print(f"Note: {WEIGHTED_RECS_CSV} not found, recommendations won't exclude seen items")
    user_seen_items = {}

# ── INDEX LOOKUPS ────────────────────────────────────────────────────────────
user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
item_id_to_idx = {aid: idx for idx, aid in enumerate(item_ids)}


# ── FUNCTION 1: TOP-N RECOMMENDATIONS FOR A USER ───────────────────────────────
def recommend_for_user(user_id, n=TOP_N, exclude_seen=True):
    """Return the top-n recommended app_ids (and scores) for a given user_id."""
    if user_id not in user_id_to_idx:
        raise ValueError(f"user_id {user_id} not found in trained factors")

    u_idx = user_id_to_idx[user_id]
    u_vec = user_vectors[u_idx]

    # Score every item: dot product of user vector with each item vector
    scores = item_vectors @ u_vec

    if exclude_seen and user_id in user_seen_items:
        seen = user_seen_items[user_id]
        for app_id in seen:
            if app_id in item_id_to_idx:
                scores[item_id_to_idx[app_id]] = -np.inf

    top_idx = np.argsort(scores)[::-1][:n]
    return [(item_ids[i], scores[i]) for i in top_idx]


# ── FUNCTION 2: TOP-N SIMILAR GAMES TO A GIVEN GAME ────────────────────────────
def similar_games(app_id, n=TOP_N):
    """Return the top-n most similar app_ids (and similarity scores) to a given app_id,
    using cosine similarity between item factor vectors."""
    if app_id not in item_id_to_idx:
        raise ValueError(f"app_id {app_id} not found in trained factors")

    i_idx = item_id_to_idx[app_id]
    i_vec = item_vectors[i_idx]

    # Cosine similarity: normalize vectors before dot product
    norms = np.linalg.norm(item_vectors, axis=1)
    target_norm = np.linalg.norm(i_vec)
    denom = norms * target_norm
    denom[denom == 0] = 1e-10  # avoid division by zero for zero-vectors

    similarities = (item_vectors @ i_vec) / denom
    similarities[i_idx] = -np.inf  # exclude the game itself

    top_idx = np.argsort(similarities)[::-1][:n]
    return [(item_ids[i], similarities[i]) for i in top_idx]


# ── DEMO ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_user = user_ids[0]
    print(f"\n── Top {TOP_N} recommendations for user {sample_user} ───────")
    for app_id, score in recommend_for_user(sample_user):
        print(f"  {title_for(app_id):<40s}  (app_id={app_id}, score={score:.4f})")

    sample_item = item_ids[0]
    print(f"\n── Top {TOP_N} games similar to '{title_for(sample_item)}' (app_id={sample_item}) ───────")
    for app_id, sim in similar_games(sample_item):
        print(f"  {title_for(app_id):<40s}  (app_id={app_id}, similarity={sim:.4f})")