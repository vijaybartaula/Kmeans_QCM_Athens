# K-means Clustering of Quantum Circuits (IBMQ Athens)

## **Project Overview**

This project employs **K-means clustering** to analyze and categorize quantum circuits from the **IBMQ Athens** dataset based on their **performance characteristics**. Quantum computing, while revolutionary, faces significant **performance and reliability challenges** due to the inherent fragility of quantum states. These states are susceptible to errors, noise, and instability, making the performance of quantum circuits highly unpredictable.

The **IBMQ Athens dataset** offers a comprehensive resource for understanding these challenges, providing detailed insights into the behavior of quantum circuits on IBM's **Athens quantum processor**.

The primary objective of this project is to apply **unsupervised learning** techniques, specifically K-means clustering, to identify patterns in quantum circuits that correlate with their performance.

### Key Performance Features:
| Feature                | Description |
|------------------------|-------------|
| **Quantum Gate Connectivity** | The arrangement of quantum gates and their impact on performance (higher connectivity = more complexity but increased noise). |
| **Error Rates**         | The degree of errors in gate operations, qubit initialization, and measurements. |
| **Circuit Depth**       | The number of sequential gate layers in a circuit. Deep circuits are prone to noise, shallow ones may lack complexity. |
| **Coherence Times**     | The time a qubit maintains its quantum state before decoherence occurs, affecting the circuit’s complexity and reliability. |

---

### **Clustering Process**

The project applies **K-means clustering** to segment the dataset into distinct clusters based on performance similarities. This helps to identify:

- **Outliers** or poorly performing circuits.
- **Well-performing circuit groups** for future designs.

#### Steps in the Clustering Process:
1. **Preprocessing**: Scaling and normalizing the features for better accuracy.
2. **Clustering**: Applying K-means to identify clusters based on features like error rates, coherence times, and circuit depth.
3. **Evaluation**: Using metrics like the **Silhouette Score** and **Elbow Method** to fine-tune the number of clusters.

---

### **Model Evaluation**

| Metric                | Purpose                                              |
|-----------------------|------------------------------------------------------|
| **Silhouette Score**   | Measures how similar an object is to its own cluster compared to other clusters. |
| **Elbow Method**       | Determines the optimal number of clusters by looking at the "elbow" point in a graph of inertia values. |

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

K-means clustering is a powerful unsupervised learning technique widely used for grouping similar data points. While it performs best on well-separated clusters, it can be enhanced with methods like **K-means++ initialization** and **Silhouette Analysis**.

In this project, K-means clustering was applied to analyze quantum circuits from the **IBMQ Athens** dataset, offering valuable insights into quantum circuit performance. By identifying patterns in the data, this project paves the way for improving quantum computing systems and optimizing their performance.

---

## Future Work

- **Expand the feature set** by incorporating additional quantum circuit parameters.
- **Explore advanced clustering algorithms** for greater flexibility and robustness (e.g., DBSCAN, Hierarchical Clustering).
- **Validate clustering results** on new datasets to ensure generalizability.

---

### Source Code
[GitHub Repository](https://github.com/vijaybartaula/Kmeans_QCM_Athens/blob/main/IBMQ_Athens_Kmeans_Quantum_Circuits_Clustering.ipynb)

### Dataset
[IBMQ Athens Dataset](https://data.mendeley.com/datasets/pmycgb2bt7/1)
