import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('heart_disease.csv')
df_original = df.copy()

print("Missing Values:\n", df.isnull().sum())

df['Heart Disease Status'] = df['Heart Disease Status'].astype(str).str.strip().map({'Yes': 1, 'No': 0})

numeric_cols = df.select_dtypes(include=['number']).columns

corr_matrix = df[numeric_cols].corr()

target_corr = corr_matrix[['Heart Disease Status']].sort_values(
    by='Heart Disease Status',
    ascending=False
)

plt.figure(figsize=(6,8))
sns.heatmap(target_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation with Heart Disease Status")
plt.show()

categorical_cols = ['Gender', 'Stress Level']

for col in categorical_cols:
    if col in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, hue='Heart Disease Status', data=df)
        plt.show()

target_cols = [col for col in df.columns if "Heart" in col]

if len(target_cols) > 0:
    X = df.drop(columns=target_cols)
else:
    X = df.copy()

X = pd.get_dummies(X, drop_first=True)

X = X.fillna(X.median())

X = X.apply(pd.to_numeric, errors='coerce')

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.show()

k = 3

kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

print(df['Cluster'].value_counts())

silhouette = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", silhouette)

print("Inertia:", kmeans.inertia_)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])

plt.figure(figsize=(10,7))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=clusters)
plt.title("K-Means Clustering (PCA)")
plt.show()

centroids = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(10,7))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=clusters)
plt.scatter(centroids_pca[:,0], centroids_pca[:,1], s=300, marker='X')
plt.title("Clusters with Centroids")
plt.show()

X_before = df_original.copy()

target_cols_before = [col for col in X_before.columns if "Heart" in col]

if len(target_cols_before) > 0:
    X_before = X_before.drop(columns=target_cols_before)

X_before = pd.get_dummies(X_before, drop_first=True)

X_before = X_before.fillna(X_before.median())

X_before = X_before.apply(pd.to_numeric, errors='coerce')

X_before = X_before.fillna(X_before.median())

X_before_scaled = scaler.fit_transform(X_before)

kmeans_before = KMeans(n_clusters=3, random_state=42)
clusters_before = kmeans_before.fit_predict(X_before_scaled)

silhouette_before = silhouette_score(X_before_scaled, clusters_before)

print("BEFORE:", silhouette_before)
print("AFTER:", silhouette)

df.to_csv("heart_clustered.csv", index=False)