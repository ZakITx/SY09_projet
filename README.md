# SY09_projet

Projet SY09 sur le dataset des musiques les plus streamées en 2023 

## Indication

Peut normaliser pour l'acp, pas pour le k-means

Pour stream en fct de nb artist

- stripplot OK
- non paramétrique de la nova
- wilkonson 2 à 2
- jitter sur le nombre d'éléments OK

streams_mode.png major minor : wilkonson OK

aftd sur la musicalité
créer une distance entre les chansons avec les éléments de la musicalité

k-means avec les interdistances : kamedoids (sklearn_extra.cluster.KMedoids)
vérifier si c'est logique

prédiction sur musicalité en fonction bpm, musicalité, tempo, rythme, tonalité
