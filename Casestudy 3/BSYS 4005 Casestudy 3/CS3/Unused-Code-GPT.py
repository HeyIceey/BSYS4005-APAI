# chat gpt code for CS3 segmentation case study - NOT USING - JUST AN EXAMPLE/FOR REFRENCE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

print("Loading data...")
df = pd.read_csv("SantaClaraListings.csv")

# -----------------------------
# 1. CLEAN + FEATURE ENGINEERING
# -----------------------------

data = df[[
    "price",
    "review_scores_rating",
    "review_scores_value",
    "reviews_per_month",
    "estimated_occupancy_l365d",
    "accommodates",
    "host_is_superhost"
]].copy()

# Clean price
data["price"] = (
    data["price"]
    .astype(str)
    .str.replace(r"[$,]", "", regex=True)
    .astype(float)
)

# Convert superhost flag
data["host_is_superhost"] = data["host_is_superhost"].map({"t": 1, "f": 0})

# Drop missing
data = data.dropna()

# Remove extreme outliers
data = data[data["price"] < 1000]

# Feature engineering
data["log_price"] = np.log1p(data["price"])
data["demand_score"] = (
    data["reviews_per_month"] *
    data["estimated_occupancy_l365d"]
)
data["efficiency"] = (
    data["review_scores_rating"] /
    data["price"]
)

print(f"Clean dataset size: {len(data)} listings")

# -----------------------------
# 2. SELECT FEATURES FOR CLUSTERING
# -----------------------------

features = data[[
    "log_price",
    "review_scores_rating",
    "review_scores_value",
    "demand_score",
    "accommodates",
    "host_is_superhost"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# -----------------------------
# 3. CHOOSE BEST K
# -----------------------------

print("\nEvaluating cluster counts...")

scores = []
k_range = range(2, 8)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=30, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)
    print(f"K={k} → Silhouette Score: {score:.3f}")

best_k = k_range[np.argmax(scores)]
print(f"\nBest cluster count chosen: {best_k}")

# -----------------------------
# 4. FINAL CLUSTERING
# -----------------------------

kmeans = KMeans(n_clusters=best_k, random_state=30, n_init=10)
data["cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------
# 5. PCA VISUALIZATION
# -----------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=data["cluster"],
    cmap="viridis",
    alpha=0.7
)

plt.title("Market Segmentation — PCA View")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter)
plt.savefig("pca_clusters3.png")
plt.show()

print("\n[PCA cluster visualization saved → pca_clusters.png]")


cluster_means = data.groupby("cluster")[[
    "price",
    "review_scores_rating",
    "demand_score",
    "accommodates"
]].mean()

cluster_means.plot(kind="bar", figsize=(12,6))
plt.title("Cluster Business Profiles")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.legend(loc="best")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))

plt.scatter(
    data["price"],
    data["demand_score"],
    c=data["cluster"],
    alpha=0.6
)

plt.xlabel("Price")
plt.ylabel("Demand Score")
plt.title("Investment Landscape View")
plt.show()

# -----------------------------
# 6. BUSINESS SUMMARY
# -----------------------------

summary = data.groupby("cluster")[[
    "price",
    "review_scores_rating",
    "demand_score",
    "accommodates"
]].mean().round(2)

counts = data["cluster"].value_counts()

print("\n" + "="*60)
print("SEGMENT PROFILE SUMMARY")
print("="*60)
print(summary)
print("\nCluster sizes:")
print(counts)
print("="*60)

# -----------------------------
# 7. INTERPRETATION HELP
# -----------------------------

print("\n--- Segment Insights ---")

for c in summary.index:
    row = summary.loc[c]
    print(
        f"\nCluster {c}: "
        f"Avg price ${row['price']}, "
        f"rating {row['review_scores_rating']}, "
        f"demand score {row['demand_score']}, "
        f"accommodates {row['accommodates']}"
    )

print("\nDone.")
