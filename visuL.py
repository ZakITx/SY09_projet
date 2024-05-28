# %%
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage


# %%
data = pd.read_csv('spotify-2023.csv', encoding='latin-1')
data.head()


# %%
data.info()


# %%
# Remove line with missing values
data = data.drop(data.iloc[[574]].index)
data = data.dropna()
data.info()
# Reinitalize index
data.reset_index(inplace=True)


# %%
data['streams'] = data['streams'].astype(dtype='int64')

# Remove coma from integer values
data['in_deezer_playlists'] = data['in_deezer_playlists'].apply(lambda x: x.replace(',', ''))
data['in_deezer_playlists'] = data['in_deezer_playlists'].astype(dtype='int64')

data['in_shazam_charts'] = data['in_shazam_charts'].apply(lambda x: x.replace(',', ''))
data['in_shazam_charts'] = data['in_shazam_charts'].astype(dtype='int64')


# %%
numerical_df = data.select_dtypes(include=np.number)

# Standardize the data (Z = (X - mean)/sd)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_df)


# %%
# Only musical features
musical_features = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
musical_df = data[musical_features]


# %%
# PCA
pca = PCA()
pca.fit(musical_df)

# Get the explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Plot the explained variance ratio
sns.barplot(x=np.arange(1, len(explained_variance)+1), y=explained_variance)


# %%
# Add the other columns to the musical features
df = data.loc[musical_df.index]

# scatter plot of the first two principal components
musical_pca = pca.transform(musical_df)
sns.scatterplot(x=musical_pca[:, 0], y=musical_pca[:, 1], hue=df['key'], style=df['mode'], size=df['bpm'], sizes=(20, 200)) 


# %%
sns.scatterplot(x=musical_pca[:, 0], y=musical_pca[:, 1], hue=df['key'], style=df['mode'], size=df['streams'], sizes=(20, 200))


# %%
sns.scatterplot(x=musical_pca[:, 0], y=musical_pca[:, 1], hue=df['key'], style=df['mode'], size=df['in_spotify_playlists'], sizes=(20, 200))


# %%
sns.scatterplot(x=musical_pca[:, 0], y=musical_pca[:, 1], hue=df['key'], style=df['mode'], size=df['in_spotify_charts'], sizes=(20, 200))


# %%
sns.scatterplot(x=musical_pca[:, 2], y=musical_pca[:, 0], hue=df['key'], style=df['mode'], size=df['bpm'], sizes=(20, 200))

