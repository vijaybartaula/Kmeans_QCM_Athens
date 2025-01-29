# K-means Clustering of Quantum Circuits (IBMQ Athens)

This project applies K-means clustering to analyze the IBMQ Athens dataset, aiming to categorize quantum circuits based on their performance characteristics. The dataset includes key features such as quantum gate connectivity, error rates, circuit depth, and coherence times, all of which influence the performance and reliability of quantum circuits. By using K-means clustering, the project identifies patterns and groups circuits with similar behaviors, helping to better understand the system’s dynamics.

## Key Features:
- **Quantum Gate Connectivity**
- **Error Rates**
- **Circuit Depth**
- **Coherence Times**

The project uses the **Elbow Method** and **Silhouette Score** to determine the optimal number of clusters and integrates **PCA (Principal Component Analysis)** for visualizing high-dimensional data in a 2D plot.

Once clusters are formed, the project provides insights by analyzing the mean feature values within each cluster, such as identifying circuits with high error rates but low coherence times, or circuits that suggest optimal performance with low error rates.

## Model Evaluation:
- **Silhouette Score** to evaluate the quality of clustering and cluster separation.

Additionally, the project explores fine-tuning the K-means model by experimenting with different parameters like `n_init` and testing alternative clustering techniques (e.g., DBSCAN or Hierarchical Clustering). Future work includes expanding the feature set and validating results with new datasets.

---

## Steps to Run the Project

### Step 1: Import Necessary Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
```

### Step 2: Load the Dataset
```python
df = pd.read_csv('/content/IBMQAthens.csv')
```

### Step 3: Inspect the Dataset
```python
df.head()
print(df.info())  # Check for missing values
print(df.describe())  # Check summary statistics
```

### Step 4: Handle Missing Data
```python
df = df.dropna()  # Or use df.fillna() to fill missing values with the mean or median
```

### Step 5: Feature Selection
```python
features = df[['cx_0_1', 'cx_0_2', 'cx_0_3', 'cx_0_4', 'cx_1_0', 'cx_1_2', 'cx_1_3', 'cx_1_4', 'cx_2_0', 'cx_2_1', 'cx_2_3', 'cx_2_4', 'cx_3_0', 'cx_3_1', 'cx_3_2', 'cx_3_4', 'cx_4_0', 'cx_4_1', 'cx_4_2', 'cx_4_3']]
```

### Step 6: Normalize the Features
```python
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```

### Step 7: Determine the Optimal Number of Clusters (Elbow Method and Silhouette Score)
```python
inertia = []
sil_scores = []
for k in range(1, 11):  # Test for 1 to 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)
    if k > 1:  # Silhouette score requires at least 2 clusters
        sil_score = silhouette_score(features_scaled, kmeans.labels_)
        sil_scores.append(sil_score)
```

### Step 8: Plot Elbow Method and Silhouette Scores
```python
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', color='b')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), sil_scores, marker='o', color='g')
plt.title('Silhouette Score for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
```

### Step 9: Apply K-means Clustering
```python
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
df['cluster'] = clusters
```

### Step 10: Visualize the Clusters Using PCA
```python
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=df['cluster'], cmap='viridis', alpha=0.7)
plt.title("K-means Clustering of Quantum Circuits (IBMQ Athens)")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()
```

### Step 11: Plot Cluster Distribution
```python
cluster_dist = df['cluster'].value_counts()
```

### Step 12: Display Cluster Centroids
```python
centroids = kmeans.cluster_centers_
```

### Step 13: Evaluate the Quality of Clustering
```python
sil_score = silhouette_score(features_scaled, clusters)
```

### Step 14: Investigate Cluster Characteristics
```python
cluster_summaries = {}
for cluster_num in range(3):  # Loop through each cluster (0, 1, 2)
    cluster_summaries[cluster_num] = df[df['cluster'] == cluster_num].describe()
```

### Step 15: Fine-Tuning the Model (Optional)
```python
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)  # Increased n_init for better results
clusters = kmeans.fit_predict(features_scaled)
df['cluster'] = clusters
```

### Step 16: Model Validation and Interpretation
```python
from sklearn.metrics import adjusted_rand_score

kmeans_run_1 = KMeans(n_clusters=3, random_state=42)
clusters_run_1 = kmeans_run_1.fit_predict(features_scaled)

kmeans_run_2 = KMeans(n_clusters=3, random_state=43)
clusters_run_2 = kmeans_run_2.fit_predict(features_scaled)

ari = adjusted_rand_score(clusters_run_1, clusters_run_2)
print(f"Adjusted Rand Index (ARI) between two K-means runs: {ari:.4f}")
```

---

## Conclusion

K-means clustering is a powerful unsupervised learning technique, and it is widely used in various fields for grouping similar data points. While it works best for well-separated clusters, it can be adapted and fine-tuned with methods like **K-means++ initialization** and **Silhouette Analysis** to enhance the clustering process.

In this project, K-means clustering was applied to analyze quantum circuits from the **IBMQ Athens** dataset, providing valuable insights into quantum circuit performance. By identifying patterns in the data, this project opens the door to improving quantum computing systems and optimizing their performance.

---

## Future Work

- **Expand the feature set** by incorporating additional quantum circuit parameters.
- **Explore more advanced clustering algorithms** for better flexibility and robustness (e.g., DBSCAN, Hierarchical Clustering).
- **Validate clustering results** on new datasets to ensure generalizability.

---

### Source Code
[GitHub Repository Link](https://github.com/vijaybartaula/Kmeans_QCM_Athens/blob/main/IBMQ_Athens_Kmeans_Quantum_Circuits_Clustering.ipynb)

### Dataset
[IBMQ Athens Dataset](https://data.mendeley.com/datasets/pmycgb2bt7/1)
