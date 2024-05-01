# %%
import numpy as np # linear algebra operations
import pandas as pd # used for data preparation
#import plotly.express as px #used for data visualization
#import plotly.graph_objects as go
import seaborn as sns 
import matplotlib as plt


# %%
# Load the data
data = pd.read_csv('spotify-2023.csv', encoding='latin-1')


# %%
## Data Exploration
data.head(2)


# %%
data.columns


# %%
data.info()


# %%
# check for rows 
print(data.iloc[574])
print("\n")
print(data.iloc[574]['streams'])


# %%
# drop rows (574) 
data = data.drop(data.iloc[[574]].index)


# %%
data['streams'] = data['streams'].astype(dtype='int64')


# %%
# check for duplicated rows
data.duplicated().sum()


# %%
# Data Analysis and Visualising
# top 10 songs by streams 
top_10_streams = data.filter(items = ['track_name','streams', 'artist(s)_name'], axis = 1)
# sorting the values by stream in decsending order
top_10_streams = top_10_streams.sort_values(by = 'streams',ascending = False).head(10)
top_10_streams


# %%
sns.barplot(x = 'streams', y = 'track_name', data = top_10_streams)


# %%
# top 10 artist by number of streams 
top_10_artist = data.filter(items = ['artist(s)_name','streams'], axis = 1)
# sorting and sum the number of streams 
top_10_artist =top_10_artist.groupby(['artist(s)_name'])['streams'].sum().sort_values(ascending =False).to_frame().head(10)
top_10_artist.reset_index(inplace = True)
top_10_artist


# %%
sns.barplot(x = 'streams', y = 'artist(s)_name', data = top_10_artist)


# %%
# How Featuring songs affect numbers of streams 
feat_songs = data.filter(items =['artist(s)_name','artist_count','streams'],axis = 1)
feat_songs


# %%
sns.stripplot(x = 'artist_count', y = 'streams', data = feat_songs, jitter=0.35)


# %%
# Wilcoxon test to check if the number of artist affect the number of streams
from scipy.stats import wilcoxon
stat, p = wilcoxon(feat_songs['artist_count'], feat_songs['streams'])
print('Statistics=%.3f, p=%.3f' % (stat, p))


# %%
# Nombre de son en fonction du nombre d'artistes qui ont participé
feat_songs_by_artist_count = data.filter(items=['artist_count'], axis = 1)
feat_songs_by_artist_count['nb'] = 0
feat_songs_by_artist_count = feat_songs_by_artist_count.groupby('artist_count').count().reset_index()
feat_songs_by_artist_count


# %%
sns.barplot(x = 'artist_count', y = 'nb', data = feat_songs_by_artist_count)


# %%
# How Month affect streams 
mont_st = data.filter(items = ['released_month','streams'],axis = 1)
mont_st = mont_st.groupby(['released_month'])['streams'].sum().to_frame()
mont_st.reset_index(inplace = True)
mont_st = mont_st.sort_values(by = 'released_month')
mont_st


# %%
sns.barplot(x = 'released_month', y = 'streams', data = mont_st)


# %%
month_nb = data.filter(items = ['released_month'], axis = 1)
month_nb['nb'] = 0
month_nb = month_nb.groupby('released_month').count().reset_index()
month_nb


# %%
sns.barplot(x = 'released_month', y = 'nb', data = month_nb)


# %%
# Is Being in spotify playlists affect number of streams 
sp_playlist = data.filter(items =['in_spotify_playlists','streams'],axis = 1)
# sorting by number of existence in spotify playlist in decending order
sp_playlist = sp_playlist.sort_values(by = 'in_spotify_playlists',ascending = False)
sp_playlist


# %%
sns.scatterplot(x = 'in_spotify_playlists', y = 'streams', data = sp_playlist)

