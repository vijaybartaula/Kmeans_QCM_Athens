# K-means Clustering of Quantum Circuits (IBMQ Athens)

## **Project Overview**

This project leverages **K-means clustering** to analyze and categorize quantum circuits from the **IBMQ Athens** dataset based on their **performance characteristics**. Quantum computing, though a groundbreaking technology, faces several **performance and reliability challenges** that stem from the inherent nature of quantum mechanics. Unlike classical computing, quantum computing relies on delicate quantum states that are prone to errors, noise, and other forms of instability. As such, the performance of quantum circuits—sequences of quantum operations designed to solve computational problems—is often unpredictable and varies significantly from one quantum processor to another. The IBMQ Athens dataset provides valuable insights into these factors, allowing for a comprehensive analysis of the behavior of quantum circuits on IBM's quantum processor.

The primary goal of this project is to apply **unsupervised learning** techniques, specifically K-means clustering, to identify patterns in the quantum circuits that correlate with their performance. The **IBMQ Athens dataset** serves as a rich source of data for understanding and addressing these challenges. It contains detailed information on various **performance characteristics** of quantum circuits executed on IBM's **Athens quantum processor**, such as:

1. **Quantum Gate Connectivity:** This refers to the arrangement of quantum gates that operate on the quantum bits (qubits) in a circuit. The performance of quantum circuits is heavily influenced by how well-connected the qubits are. High gate connectivity may result in more complex quantum operations but can lead to greater susceptibility to noise or errors. Conversely, lower connectivity might lead to less powerful circuits, but they could be more stable in terms of performance.

2. **Error Rates:** In quantum circuits, errors occur due to imperfections in quantum gate operations, qubit initialization, and measurement processes. These errors are unavoidable in current quantum computing systems but can significantly impact the reliability of a quantum circuit's results. The **error rates** of gates and measurements are one of the most critical indicators of the quality of a quantum circuit. A high error rate indicates a less reliable circuit, which may lead to inaccurate results, while a low error rate reflects better performance and stability.

3. **Circuit Depth:** Circuit depth refers to the number of layers or gates that are applied sequentially to a quantum circuit. Deeper circuits are often necessary for more complex computations but may encounter **increased noise accumulation** and **greater likelihood of errors** over time, especially if qubits have limited coherence times. Shallow circuits may mitigate these issues, but they may not be suitable for solving more complex problems. Analyzing circuit depth in conjunction with other factors helps to balance the need for more complex circuits with the practical limits of quantum hardware.

4. **Coherence Times:** Coherence time is the amount of time a qubit can maintain its quantum state before it interacts with the external environment and becomes corrupted. Longer coherence times are crucial for the successful execution of quantum algorithms, as they allow for more operations to be performed before the quantum state decays. Short coherence times can severely limit the performance of quantum circuits, especially for deeper or more complex algorithms. **Coherence times** are thus a critical factor in assessing whether a quantum circuit can be reliably executed on a given quantum processor.

The IBMQ Athens dataset encapsulates all these key performance features, providing a comprehensive picture of how quantum circuits behave in real-world conditions. **Categorizing these circuits based on their performance characteristics** allows researchers and engineers to identify **patterns, clusters, and outliers** in the data. For instance, by grouping circuits with similar error rates and coherence times, the project can highlight which types of quantum circuits are likely to succeed or fail on a given processor. This categorization can also help identify hardware limitations and **inform future hardware improvements**.

By applying clustering techniques such as **K-means**, the project enables a better understanding of how different factors—such as gate connectivity, error rates, circuit depth, and coherence times—interact to shape circuit performance. This not only aids in optimizing individual circuits but also provides insights into **system-wide improvements** needed in quantum computing hardware. With a clearer understanding of the behavior of different types of circuits, researchers can focus on improving specific areas, such as reducing error rates or extending coherence times, thereby improving the overall reliability and capability of quantum processors like **IBMQ Athens**.

By grouping the quantum circuits based on these features, the clustering algorithm can reveal subtle but important patterns, such as:

- **Clusters of circuits** exhibiting low error rates but shorter coherence times, which may suggest challenges in maintaining quantum states over long periods.
- **High error rate clusters** that might highlight problematic circuits, potentially due to faulty gates or poor qubit connections.
- **Efficient circuits** with favorable combinations of low error rates, optimal gate connectivity, and sufficient coherence times.

K-means clustering helps uncover these underlying patterns, offering insights that could potentially guide future improvements in quantum circuit design and processor optimization.

### **Clustering Process**

Through the application of **K-means clustering**, the project segments the dataset into distinct clusters based on the similarity of their performance characteristics. Each cluster corresponds to a group of circuits that share similar behaviors, making it easier to identify:

- **Outliers** or poorly performing circuits that fall outside the norm.
- **Well-performing circuit groups** that might serve as templates for future quantum circuit designs.

Once the clusters are formed, detailed analysis of the cluster’s characteristics—such as mean feature values—helps provide a clearer understanding of the types of quantum circuits in each group. This step is crucial for pinpointing areas where quantum processors can be improved, whether it's addressing specific quantum gate errors, optimizing coherence times, or reducing circuit depth.

---

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
[GitHub Repository](https://github.com/vijaybartaula/Kmeans_QCM_Athens/blob/main/IBMQ_Athens_Kmeans_Quantum_Circuits_Clustering.ipynb)

### Dataset
[IBMQ Athens Dataset](https://data.mendeley.com/datasets/pmycgb2bt7/1)
