# SY09_projet

Projet SY09 sur le dataset des musiques les plus streamées en 2023

# Indications et exploration

Peut normaliser pour l'acp, pas pour le k-means

Pour stream en fct de nb artist

- [X] stripplot

non paramétrique de l'anova wilkonson 2 à 2

- [X] jitter sur le nombre d'éléments
- [X] streams_mode.png major minor : wilkonson

* [ ] aftd sur la musicalité
* [ ] créer une distance entre les chansons avec les éléments de la musicalité
* [ ] k-means avec les interdistances : kamedoids (sklearn_extra.cluster.KMedoids)
  vérifier si c'est logique
* [ ] prédiction sur musicalité en fonction bpm, musicalité, tempo, rythme, tonalité

## PCA's PCs on quantitative variables

PC1 =  'streams' (+-**100%**),
PC2 = 'in_spotify_playlists' (**99.9984%**),
PC3 = 'in_deezer_playlists' (**0.98%**), 'in_shazam_charts' (**0.17%**), 'in_apple_playlists' (**0.085%**)

## KMeans' clusters

Frontière de décision = f('streams', 'in_spotify_playlists', 'released_year')

## Correlation between Variables ##

1. (streams, in_spotify_playlists): **0.83**
2. (in_spotify_playlists, in_apple_playlists): **0.78**
3. (streams, in_apple_playlists): **0.67**
4. (streams, released_year): **-0.68**
5. (released_year, in_spotify_playlists): **-0.66**

