import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA


# ==========================================================
# SETUP
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = os.path.join(BASE_DIR, "database.csv")

OUTPUT_FOLDER = os.path.join(BASE_DIR, "clustering_outputs")
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

SAMPLE_SIZE = 10000
RANDOM_STATE = 42

print("Output folder:", OUTPUT_FOLDER)
print("Plots folder:", PLOTS_FOLDER)


# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(FILE_NAME, low_memory=False)

rename_map = {
    "Victim Age": "Age",
    "Victim Sex": "Gender",
    "Victim Race": "Race",
    "Crime Solved": "Solved"
}

df = df.rename(columns=rename_map)

features = ["Gender", "Race", "Age", "City", "State", "Weapon", "Relationship"]
target_col = "Solved"

existing_features = [col for col in features if col in df.columns]

cols_to_keep = existing_features.copy()
if target_col in df.columns:
    cols_to_keep.append(target_col)

data = df[cols_to_keep].copy()

unknown_values = ["Unknown", "unknown", "UNK", "", "nan", "NaN", "None"]
data = data.replace(unknown_values, np.nan)

for col in data.select_dtypes(include=["object", "string"]).columns:
    data[col] = data[col].astype(str).str.strip()
    data[col] = data[col].replace(["nan", "None", ""], np.nan)

if "Age" in data.columns:
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
    data = data[data["Age"].notna()]
    data = data[(data["Age"] >= 0) & (data["Age"] <= 110)]

print("Full cleaned shape:", data.shape)

# SAMPLE DATA SO THE PROGRAM DOES NOT FREEZE
if len(data) > SAMPLE_SIZE:
    data_model = data.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()
else:
    data_model = data.copy()

print("Modeling sample shape:", data_model.shape)


# ==========================================================
# PREPROCESSING
# ==========================================================
X = data_model[existing_features].copy()

numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
categorical_features = [col for col in X.columns if col not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_processed = preprocessor.fit_transform(X)

print("Processed shape:", X_processed.shape)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def save_plot(filename):
    path = os.path.join(PLOTS_FOLDER, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved plot:", path)
    plt.close()


def evaluate_model(X_eval, labels):
    labels = np.array(labels)

    valid_mask = labels != -1
    X_valid = X_eval[valid_mask]
    labels_valid = labels[valid_mask]

    n_clusters = len(set(labels_valid))
    n_noise = np.sum(labels == -1)

    if n_clusters < 2:
        return n_clusters, n_noise, np.nan, np.nan, np.nan

    sil = silhouette_score(X_valid, labels_valid)
    db = davies_bouldin_score(X_valid, labels_valid)
    ch = calinski_harabasz_score(X_valid, labels_valid)

    return n_clusters, n_noise, sil, db, ch


# ==========================================================
# K-MEANS
# ==========================================================
k_values = range(2, 11)
kmeans_results = []

for k in k_values:
    print(f"Running K-Means k={k}...")
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_processed)

    n_clusters, n_noise, sil, db, ch = evaluate_model(X_processed, labels)

    kmeans_results.append({
        "k": k,
        "inertia": model.inertia_,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch
    })

kmeans_df = pd.DataFrame(kmeans_results)
kmeans_df.to_csv(os.path.join(OUTPUT_FOLDER, "kmeans_results.csv"), index=False)

best_k = int(kmeans_df.loc[kmeans_df["silhouette"].idxmax(), "k"])

kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
data_model["KMeans_Cluster"] = kmeans_final.fit_predict(X_processed)

plt.figure(figsize=(8, 5))
plt.plot(kmeans_df["k"], kmeans_df["inertia"], marker="o")
plt.title("K-Means Elbow Method")
plt.xlabel("Number of Clusters, k")
plt.ylabel("Inertia")
plt.grid(True)
save_plot("kmeans_elbow_method.png")

plt.figure(figsize=(8, 5))
plt.plot(kmeans_df["k"], kmeans_df["silhouette"], marker="o")
plt.title("K-Means Silhouette Scores")
plt.xlabel("Number of Clusters, k")
plt.ylabel("Silhouette Score")
plt.grid(True)
save_plot("kmeans_silhouette_scores.png")


# ==========================================================
# GMM
# ==========================================================
gmm_results = []

for k in k_values:
    print(f"Running GMM components={k}...")
    model = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_processed)

    n_clusters, n_noise, sil, db, ch = evaluate_model(X_processed, labels)

    gmm_results.append({
        "components": k,
        "bic": model.bic(X_processed),
        "aic": model.aic(X_processed),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch
    })

gmm_df = pd.DataFrame(gmm_results)
gmm_df.to_csv(os.path.join(OUTPUT_FOLDER, "gmm_results.csv"), index=False)

best_gmm_k = int(gmm_df.loc[gmm_df["bic"].idxmin(), "components"])

gmm_final = GaussianMixture(n_components=best_gmm_k, random_state=RANDOM_STATE)
data_model["GMM_Cluster"] = gmm_final.fit_predict(X_processed)

plt.figure(figsize=(8, 5))
plt.plot(gmm_df["components"], gmm_df["bic"], marker="o", label="BIC")
plt.plot(gmm_df["components"], gmm_df["aic"], marker="o", label="AIC")
plt.title("GMM Model Selection")
plt.xlabel("Number of Components")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
save_plot("gmm_model_selection.png")


# ==========================================================
# DBSCAN
# ==========================================================
eps_values = [1, 2, 3, 4, 5, 6, 7, 8]
min_samples_values = [3, 5, 8, 10]

dbscan_results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        print(f"Running DBSCAN eps={eps}, min_samples={min_samples}...")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_processed)

        n_clusters, n_noise, sil, db, ch = evaluate_model(X_processed, labels)

        dbscan_results.append({
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch
        })

dbscan_df = pd.DataFrame(dbscan_results)
dbscan_df.to_csv(os.path.join(OUTPUT_FOLDER, "dbscan_results.csv"), index=False)

valid_dbscan = dbscan_df.dropna(subset=["silhouette"])

if len(valid_dbscan) > 0:
    best_dbscan = valid_dbscan.sort_values("silhouette", ascending=False).iloc[0]
    best_eps = best_dbscan["eps"]
    best_min_samples = int(best_dbscan["min_samples"])

    dbscan_final = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    data_model["DBSCAN_Cluster"] = dbscan_final.fit_predict(X_processed)
else:
    best_eps = None
    best_min_samples = None
    data_model["DBSCAN_Cluster"] = -1

plt.figure(figsize=(8, 5))
plt.scatter(dbscan_df["eps"], dbscan_df["silhouette"])
plt.title("DBSCAN Silhouette Scores by eps")
plt.xlabel("eps")
plt.ylabel("Silhouette Score")
plt.grid(True)
save_plot("dbscan_silhouette_by_eps.png")


# ==========================================================
# PCA PLOTS
# ==========================================================
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_processed)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=data_model["KMeans_Cluster"], s=10)
axes[0].set_title("K-Means Clusters")

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=data_model["GMM_Cluster"], s=10)
axes[1].set_title("GMM Clusters")

axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=data_model["DBSCAN_Cluster"], s=10)
axes[2].set_title("DBSCAN Clusters")

for ax in axes:
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(True)

plt.tight_layout()
save_plot("pca_all_models_comparison.png")


# ==========================================================
# FINAL COMPARISON
# ==========================================================
kmeans_scores = evaluate_model(X_processed, data_model["KMeans_Cluster"])
gmm_scores = evaluate_model(X_processed, data_model["GMM_Cluster"])
dbscan_scores = evaluate_model(X_processed, data_model["DBSCAN_Cluster"])

comparison = pd.DataFrame([
    {
        "Model": "K-Means",
        "Chosen Parameters": f"k = {best_k}",
        "Number of Clusters": kmeans_scores[0],
        "Noise Points": kmeans_scores[1],
        "Silhouette Score": kmeans_scores[2],
        "Davies-Bouldin Score": kmeans_scores[3],
        "Calinski-Harabasz Score": kmeans_scores[4]
    },
    {
        "Model": "GMM",
        "Chosen Parameters": f"components = {best_gmm_k}",
        "Number of Clusters": gmm_scores[0],
        "Noise Points": gmm_scores[1],
        "Silhouette Score": gmm_scores[2],
        "Davies-Bouldin Score": gmm_scores[3],
        "Calinski-Harabasz Score": gmm_scores[4]
    },
    {
        "Model": "DBSCAN",
        "Chosen Parameters": f"eps = {best_eps}, min_samples = {best_min_samples}",
        "Number of Clusters": dbscan_scores[0],
        "Noise Points": dbscan_scores[1],
        "Silhouette Score": dbscan_scores[2],
        "Davies-Bouldin Score": dbscan_scores[3],
        "Calinski-Harabasz Score": dbscan_scores[4]
    }
])

comparison.to_csv(os.path.join(OUTPUT_FOLDER, "final_model_comparison.csv"), index=False)

best_model = comparison.dropna(subset=["Silhouette Score"]).sort_values(
    "Silhouette Score",
    ascending=False
).iloc[0]["Model"]

plt.figure(figsize=(8, 5))
plt.bar(comparison["Model"], comparison["Silhouette Score"])
plt.title("Model Comparison Using Silhouette Score")
plt.xlabel("Model")
plt.ylabel("Silhouette Score")
plt.grid(axis="y")
save_plot("model_comparison_silhouette.png")


# ==========================================================
# CLUSTER SUMMARIES
# ==========================================================
summary_rows = []

for model_col in ["KMeans_Cluster", "GMM_Cluster", "DBSCAN_Cluster"]:
    for cluster_id in sorted(data_model[model_col].unique()):
        cluster_data = data_model[data_model[model_col] == cluster_id]

        row = {
            "Model": model_col.replace("_Cluster", ""),
            "Cluster": cluster_id,
            "Count": len(cluster_data)
        }

        if "Age" in cluster_data.columns:
            row["Average Age"] = cluster_data["Age"].mean()

        for col in ["Gender", "Race", "City", "State", "Weapon", "Relationship"]:
            if col in cluster_data.columns:
                mode_value = cluster_data[col].mode()
                row[f"Most Common {col}"] = mode_value.iloc[0] if not mode_value.empty else np.nan

        summary_rows.append(row)

cluster_summary = pd.DataFrame(summary_rows)
cluster_summary.to_csv(os.path.join(OUTPUT_FOLDER, "cluster_summaries.csv"), index=False)

data_model.to_csv(os.path.join(OUTPUT_FOLDER, "clustered_sample_results.csv"), index=False)


# ==========================================================
# REPORT
# ==========================================================
report = f"""
CLUSTERING ANALYSIS REPORT
==========================

Dataset
-------
The original dataset had {df.shape[0]} rows and {df.shape[1]} columns.

After cleaning, the dataset had {data.shape[0]} rows.

Because the dataset was very large, a random sample of {len(data_model)} rows was used for clustering.
This was done so the clustering models and silhouette scores could run in a reasonable amount of time.

Features Used
-------------
{existing_features}

Models Compared
---------------
1. K-Means
2. Gaussian Mixture Model, also called GMM
3. DBSCAN

What is K-Means?
----------------
K-Means is a clustering algorithm that separates data into k groups.

The letter k means the number of clusters.

For example, if k = 3, the model tries to divide the dataset into 3 groups.

Best K-Means k:
{best_k}

What is GMM?
------------
GMM stands for Gaussian Mixture Model.

It is similar to K-Means, but it uses probability instead of only distance.
This means GMM can handle overlapping clusters better.

Best GMM components:
{best_gmm_k}

What is DBSCAN?
---------------
DBSCAN groups points based on density.
It can also identify outliers or noise points.

Noise points are labeled as -1.

Best DBSCAN eps:
{best_eps}

Best DBSCAN min_samples:
{best_min_samples}

Final Model Comparison
----------------------
{comparison.to_string(index=False)}

Best Overall Model
------------------
The best overall model was {best_model}.

This was chosen mostly based on the silhouette score.

Higher silhouette score means better separated clusters.

Conclusion
----------
The clustering analysis compared K-Means, GMM, and DBSCAN.

K-Means is the easiest to explain and uses a chosen number of clusters.
GMM is more flexible because it uses probability.
DBSCAN can detect outliers but may struggle with high-dimensional categorical data.

For this dataset sample, {best_model} performed best based on the silhouette score.

Files Created
-------------
- kmeans_results.csv
- gmm_results.csv
- dbscan_results.csv
- final_model_comparison.csv
- cluster_summaries.csv
- clustered_sample_results.csv
- clustering_report.txt

Plots Created
-------------
- kmeans_elbow_method.png
- kmeans_silhouette_scores.png
- gmm_model_selection.png
- dbscan_silhouette_by_eps.png
- pca_all_models_comparison.png
- model_comparison_silhouette.png
"""

with open(os.path.join(OUTPUT_FOLDER, "clustering_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)


print("\nDONE! Files saved successfully.")
print("Best model:", best_model)
print("Output folder:", OUTPUT_FOLDER)
print("Plots folder:", PLOTS_FOLDER)