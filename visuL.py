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

# Perform PCA
pca = PCA()
pca.fit(scaled_data)

# %%
