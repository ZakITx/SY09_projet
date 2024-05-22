#!/usr/bin/env python
# coding: utf-8

# In[124]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# ## Read the data

# In[76]:


data = pd.read_csv(
    "/Users/azmanizakary/Desktop/GI04/SY09/projet/SY09_projet/spotify-2023.csv",
    encoding="latin-1",
)
data.head()


# In[3]:


data.info()


# ## 858 lignes pour key et 903 lignes pour in_shazam_charts

# In[21]:


df_key = data[data["in_shazam_charts"].notnull()]
data = df_key[df_key["key"].notnull()]
data.loc[:, "mode"] = [0 if x == "Minor" else 1 for x in data["mode"]]

key_mapping = {"A#": "A", "C#": "C", "D#": "D", "F#": "F", "G#": "G"}
data.loc[:, "key_merged"] = data["key"].replace(key_mapping)


# In[5]:


sns.countplot(x="key", data=data, palette="viridis")
# Nombre de musiques totales par clés


# In[22]:


numerical_df = data.select_dtypes(include=np.number)

# Standardize the data (Z = (X - mean)/sd)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_df)

# Perform PCA
pca = PCA()
pca.fit(scaled_data)

# Get the eigenvalues
eigenvalues = pca.explained_variance_

# Plot the eigenvalues
plt.figure(figsize=(10, 6))
sns.barplot(x=np.arange(1, len(eigenvalues) + 1), y=eigenvalues, color="skyblue")
plt.xlabel("PCs")
plt.ylabel("Eigenvalue")
plt.title("Eigenvalue Barplot")
for i in range(len(eigenvalues) - 1):
    plt.plot(
        [i + 0.5, i + 1.5],
        [eigenvalues[i], eigenvalues[i + 1]],
        color="red",
        linestyle="--",
    )
plt.show()


# In[7]:


class_intervals = [
    (0, 80),
    (81, 100),
    (101, 120),
    (121, 140),
    (141, 160),
    (161, 180),
    (181, float("inf")),
]
class_labels = ["0-80", "81-100", "101-120", "121-140", "141-160", "161-180", "180+"]

# Assign each bpm value to a class
data["bpm_class"] = pd.cut(
    data["bpm"],
    bins=[interval[0] for interval in class_intervals] + [float("inf")],
    labels=class_labels,
)

# Create the barplot
plt.figure(figsize=(10, 6))
sns.countplot(x="bpm_class", data=data, palette="viridis")
plt.xlabel("BPM Class")
plt.ylabel("Count")
plt.title("Distribution of BPM Classes")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# 121-140 max count


# In[8]:


bpm_121_140_df = data[(data["bpm"] >= 121) & (data["bpm"] <= 140)]

# Create the barplot
plt.figure(figsize=(10, 6))
sns.countplot(x="bpm", data=bpm_121_140_df, palette="viridis")
plt.xlabel("BPM Class")
plt.ylabel("Count")
plt.title("Distribution of BPM Classes")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# 130 bpm is the most counted, next is 140


# ## PCA sur la bpm_class ne donne rien

# In[9]:


pca_data = pca.transform(scaled_data)
pca_df = pd.DataFrame(data=pca_data[:, :3], columns=["PC1", "PC2", "PC3"])
pca_df["bpm_class"] = data["bpm_class"]
pca_df["mode"] = data["mode"]
pca_df["key"] = data["key_merged"]

# Plot the scatterplot of the first two principal components colored by 'bpm_class'
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC2", hue="key", data=pca_df, palette="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Scatterplot PC1 vs PC2 (Colored by key)")
plt.legend(title="BPM Class", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC3", hue="key", data=pca_df, palette="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 3")
plt.title("Scatterplot PC1 vs PC3 (Colored by key)")
plt.legend(title="BPM Class", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC2", y="PC3", hue="key", data=pca_df, palette="viridis")
plt.xlabel("Principal Component 2")
plt.ylabel("Principal Component 3")
plt.title("Scatterplot PC2 vs PC3 (Colored by key)")
plt.legend(title="BPM Class", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()


# In[10]:


# Plot the scatterplot of the first two principal components colored by 'bpm_class'
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC2", hue="mode", data=pca_df, palette="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Scatterplot PC1 vs PC2 (Colored by mode)")
plt.legend(title="BPM Class", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC3", hue="mode", data=pca_df, palette="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 3")
plt.title("Scatterplot PC1 vs PC3 (Colored by mode)")
plt.legend(title="BPM Class", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC2", y="PC3", hue="mode", data=pca_df, palette="viridis")
plt.xlabel("Principal Component 2")
plt.ylabel("Principal Component 3")
plt.title("Scatterplot PC2 vs PC3 (Colored by mode)")
plt.legend(title="BPM Class", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()


# In[11]:


mds = MDS(n_components=3)
mds_data = mds.fit_transform(scaled_data)

# Convert MDS result to DataFrame
mds_df = pd.DataFrame(data=mds_data, columns=["MDS1", "MDS2", "MDS3"])
mds_df["bpm_class"] = data["bpm_class"]
mds_df["mode"] = data["mode"]
mds_df["key"] = data["key_merged"]


# In[12]:


# Plot the scatterplot of the MDS result
plt.figure(figsize=(10, 6))
sns.scatterplot(x="MDS1", y="MDS2", hue="key", data=mds_df, palette="viridis")
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.title("Multidimensional Scaling (MDS)")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="MDS1", y="MDS3", hue="key", data=mds_df, palette="viridis")
plt.xlabel("MDS1")
plt.ylabel("MDS3")
plt.title("Multidimensional Scaling (MDS)")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="MDS2", y="MDS3", hue="key", data=mds_df, palette="viridis")
plt.xlabel("MDS2")
plt.ylabel("MDS3")
plt.title("Multidimensional Scaling (MDS)")
plt.show()


# In[13]:


# Plot the scatterplot of the MDS result
plt.figure(figsize=(10, 6))
sns.scatterplot(x="MDS1", y="MDS2", hue="mode", data=mds_df, palette="viridis")
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.title("Multidimensional Scaling (MDS)")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="MDS1", y="MDS3", hue="mode", data=mds_df, palette="viridis")
plt.xlabel("MDS1")
plt.ylabel("MDS3")
plt.title("Multidimensional Scaling (MDS)")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="MDS2", y="MDS3", hue="mode", data=mds_df, palette="viridis")
plt.xlabel("MDS2")
plt.ylabel("MDS3")
plt.title("Multidimensional Scaling (MDS)")
plt.show()


# In[14]:


plt.figure(figsize=(12, 8))
dendrogram(linkage(scaled_data, method="ward"))
plt.title("Hierarchical Clustering Dendrogram (Ward)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()


# ## 5 clusters

# In[15]:


kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_data)

# Add cluster labels to the DataFrame
pca_df["cluster"] = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC2", hue="cluster", data=pca_df, palette="viridis")
plt.title("K-means Clustering (PC1 vs PC2)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC2", y="PC3", hue="cluster", data=pca_df, palette="viridis")
plt.title("K-means Clustering (PC2 vs PC3)")
plt.xlabel("Principal Component 2")
plt.ylabel("Principal Component 3")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC3", hue="cluster", data=pca_df, palette="viridis")
plt.title("K-means Clustering (PC1 vs PC3)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 3")
plt.show()


# In[16]:


# Add cluster labels to the DataFrame
mds_df["cluster"] = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x="MDS1", y="MDS2", hue="cluster", data=mds_df, palette="viridis")
plt.title("K-means Clustering (MDS1 vs MDS2)")
plt.xlabel("MDS 1")
plt.ylabel("MDS 2")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="MDS1", y="MDS3", hue="cluster", data=mds_df, palette="viridis")
plt.title("K-means Clustering (MDS1 vs MDS3)")
plt.xlabel("MDS 1")
plt.ylabel("MDS 3")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="MDS2", y="MDS3", hue="cluster", data=mds_df, palette="viridis")
plt.title("K-means Clustering (MDS2 vs MDS3)")
plt.xlabel("MDS 2")
plt.ylabel("MDS 3")
plt.show()
