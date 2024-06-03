# SY09_projet

Projet SY09 sur le dataset des musiques les plus streamées en 2023 


# Indications et exploration 
Peut normaliser pour l'acp, pas pour le k-means


Pour stream en fct de nb artist
- stripplot :ballot_box_with_check:

non paramétrique de l'anova wilkonson 2 à 2

- jitter sur le nombre d'éléments :ballot_box_with_check:

streams_mode.png major minor : wilkonson :ballot_box_with_check:

aftd sur la musicalité

créer une distance entre les chansons avec les éléments de la musicalité

k-means avec les interdistances : kamedoids (sklearn_extra.cluster.KMedoids)
vérifier si c'est logique

prédiction sur musicalité en fonction bpm, musicalité, tempo, rythme, tonalité

## Correlation between Variables ##

1. (streams, in_spotify_playlists): **0.76**
2. (in_spotify_playlists, in_apple_playlists): **0.70**
3. (in_spotify_playlists, in_deezer_playlists): **0.79**
4. (streams, in_deezer_playlists): **0.71**
5. (streams, in_apple_playlists): **0.67**


