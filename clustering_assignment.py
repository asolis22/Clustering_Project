import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# =========================
# 1. LOAD DATASET
# =========================
# Change this to your actual CSV file name
FILE_NAME = "database.csv"

df = pd.read_csv(FILE_NAME)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())


# =========================
# 2. RENAME COLUMNS
# =========================
# Edit this section to match your actual dataset columns exactly
rename_map = {
    "Victim Age": "Age",
    "Victim Sex": "Gender",
    "Victim Race": "Race",
    "City": "City",
    "State": "State",
    "Weapon": "Weapon",
    "Relationship": "Relationship",
    "Crime Solved": "Solved"
}

df = df.rename(columns=rename_map)

print("\nColumns after renaming:")
print(df.columns.tolist())


# =========================
# 3. SELECT FEATURES
# =========================
# These are the variables we want to use for clustering
features = ["Gender", "Race", "Age", "City", "State", "Weapon", "Relationship"]

# This is ONLY for later interpretation, not clustering
target_col = "Solved"

existing_features = [col for col in features if col in df.columns]
print("\nUsing features:", existing_features)

cols_to_keep = existing_features.copy()
if target_col in df.columns:
    cols_to_keep.append(target_col)

data = df[cols_to_keep].copy()


# =========================
# 4. CLEAN DATA
# =========================
# Replace common unknown values with NaN
unknown_values = ["Unknown", "unknown", "UNK", "", "nan", "NaN", "None"]
data = data.replace(unknown_values, np.nan)

# Clean text columns
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].astype(str).str.strip()

# Keep only valid ages if Age exists
if "Age" in data.columns:
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
    data = data[data["Age"].notna()]
    data = data[(data["Age"] >= 0) & (data["Age"] <= 110)]

print("\nShape after cleaning:", data.shape)
print("\nMissing values after cleaning:")
print(data.isnull().sum())


# =========================
# 5. SPLIT X AND y
# =========================
X = data[existing_features].copy()
y = data[target_col].copy() if target_col in data.columns else None


# =========================
# 6. PREPROCESSING
# =========================
numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
categorical_features = [col for col in X.columns if col not in numeric_features]

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_processed = preprocessor.fit_transform(X)

print("\nProcessed feature matrix shape:", X_processed.shape)

# Convert to dense array if needed
X_dense = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed


# =========================
# 7. K-MEANS: FIND BEST K
# =========================
k_values = range(2, 11)
inertias = []
sil_scores_kmeans = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_processed)

    inertias.append(kmeans.inertia_)

    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X_processed, labels)
    else:
        sil = np.nan
    sil_scores_kmeans.append(sil)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker='o')
plt.title("K-means Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), sil_scores_kmeans, marker='o')
plt.title("K-means Silhouette Scores")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

best_k = list(k_values)[np.nanargmax(sil_scores_kmeans)]
print("\nBest k for K-means:", best_k)


# =========================
# 8. FINAL K-MEANS MODEL
# =========================
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_processed)

data["KMeans_Cluster"] = kmeans_labels

print("\nK-means cluster counts:")
print(data["KMeans_Cluster"].value_counts().sort_index())


# =========================
# 9. GMM: FIND BEST NUMBER OF COMPONENTS
# =========================
bic_scores = []
aic_scores = []
sil_scores_gmm = []

for k in k_values:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_dense)

    bic_scores.append(gmm.bic(X_dense))
    aic_scores.append(gmm.aic(X_dense))

    labels = gmm.predict(X_dense)

    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X_dense, labels)
    else:
        sil = np.nan
    sil_scores_gmm.append(sil)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), bic_scores, marker='o', label='BIC')
plt.plot(list(k_values), aic_scores, marker='o', label='AIC')
plt.title("GMM Model Selection")
plt.xlabel("Number of Components")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

best_gmm_k = list(k_values)[np.argmin(bic_scores)]
print("\nBest number of GMM components:", best_gmm_k)


# =========================
# 10. FINAL GMM MODEL
# =========================
gmm_final = GaussianMixture(n_components=best_gmm_k, random_state=42)
gmm_labels = gmm_final.fit_predict(X_dense)

data["GMM_Cluster"] = gmm_labels

print("\nGMM cluster counts:")
print(data["GMM_Cluster"].value_counts().sort_index())


# =========================
# 11. DBSCAN: PARAMETER SEARCH
# =========================
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
min_samples_values = [3, 5, 8, 10]

dbscan_results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_processed)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        valid_mask = labels != -1
        unique_valid = np.unique(labels[valid_mask])

        if len(unique_valid) > 1 and valid_mask.sum() > len(unique_valid):
            sil = silhouette_score(X_processed[valid_mask], labels[valid_mask])
        else:
            sil = np.nan

        dbscan_results.append({
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": sil
        })

dbscan_df = pd.DataFrame(dbscan_results).sort_values(
    by=["silhouette", "n_clusters"], ascending=[False, False]
)

print("\nTop DBSCAN parameter results:")
print(dbscan_df.head(10))


# =========================
# 12. FINAL DBSCAN MODEL
# =========================
valid_dbscan = dbscan_df.dropna(subset=["silhouette"])

if len(valid_dbscan) > 0:
    best_db = valid_dbscan.iloc[0]
    best_eps = best_db["eps"]
    best_min_samples = int(best_db["min_samples"])

    print("\nBest DBSCAN parameters:")
    print("eps =", best_eps)
    print("min_samples =", best_min_samples)

    dbscan_final = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan_labels = dbscan_final.fit_predict(X_processed)

    data["DBSCAN_Cluster"] = dbscan_labels

    print("\nDBSCAN cluster counts:")
    print(data["DBSCAN_Cluster"].value_counts().sort_index())
else:
    print("\nNo valid DBSCAN clustering found with the tested parameters.")
    data["DBSCAN_Cluster"] = -1


# =========================
# 13. PCA VISUALIZATION
# =========================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_dense)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=data["KMeans_Cluster"], s=15)
axes[0].set_title("K-means Clusters")

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=data["GMM_Cluster"], s=15)
axes[1].set_title("GMM Clusters")

axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=data["DBSCAN_Cluster"], s=15)
axes[2].set_title("DBSCAN Clusters")

for ax in axes:
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

plt.tight_layout()
plt.show()


# =========================
# 14. COMPARE CLUSTERS WITH SOLVED
# =========================
if y is not None:
    print("\n--- KMeans Cluster vs Solved ---")
    print(pd.crosstab(data["KMeans_Cluster"], data["Solved"], normalize="index"))

    print("\n--- GMM Cluster vs Solved ---")
    print(pd.crosstab(data["GMM_Cluster"], data["Solved"], normalize="index"))

    print("\n--- DBSCAN Cluster vs Solved ---")
    print(pd.crosstab(data["DBSCAN_Cluster"], data["Solved"], normalize="index"))


# =========================
# 15. CLUSTER SUMMARIES
# =========================
if "Age" in data.columns:
    print("\nAverage Age by K-means Cluster:")
    print(data.groupby("KMeans_Cluster")["Age"].mean())

for col in ["Gender", "Race", "City", "State", "Weapon", "Relationship"]:
    if col in data.columns:
        print(f"\nMost common {col} in each K-means cluster:")
        print(
            data.groupby("KMeans_Cluster")[col]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        )


# =========================
# 16. FINAL COMPARISON TABLE
# =========================
best_dbscan_sil = valid_dbscan.iloc[0]["silhouette"] if len(valid_dbscan) > 0 else np.nan

comparison = pd.DataFrame({
    "Method": ["K-means", "GMM", "DBSCAN"],
    "Chosen Parameters": [
        f"k = {best_k}",
        f"components = {best_gmm_k}",
        f"eps = {best_eps}, min_samples = {best_min_samples}" if len(valid_dbscan) > 0 else "No valid clustering"
    ],
    "Quality Score": [
        np.nanmax(sil_scores_kmeans),
        np.nanmax(sil_scores_gmm),
        best_dbscan_sil
    ]
})

print("\nFinal Comparison Table:")
print(comparison)


# =========================
# 17. SAVE RESULTS
# =========================
data.to_csv("clustered_homicide_results.csv", index=False)
comparison.to_csv("clustering_comparison_table.csv", index=False)

print("\nDone. Results saved as:")
print("- clustered_homicide_results.csv")
print("- clustering_comparison_table.csv")