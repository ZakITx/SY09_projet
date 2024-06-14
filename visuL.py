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


# %% Arbre de décision
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
import matplotlib as mpl

def add_decision_boundary(
    model,
    resolution=100,
    ax=None,
    levels=None,
    label=None,
    color=None,
    region=True,
    model_classes=None,
):
    """Trace une frontière et des régions de décision sur une figure existante.

    :param model: Un modèle scikit-learn ou une fonction `predict`
    :param resolution: La discrétisation en nombre de points par abcisses/ordonnées à utiliser
    :param ax: Les axes sur lesquels dessiner
    :param label: Le nom de la frontière dans la légende
    :param color: La couleur de la frontière
    :param region: Colorer les régions ou pas
    :param model_classes: Les étiquettes des classes dans le cas où `model` est une fonction

    """

    # Set axes
    if ax is None:
        ax = plt.gca()

    # Add decision boundary to legend
    color = "red" if color is None else color
    sns.lineplot(x=[0], y=[0], label=label, ax=ax, color=color, linestyle="dashed")

    # Create grid to evaluate model
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    def draw_boundaries(XX, YY, Z_num, color):
        # Boundaries
        mask = np.zeros_like(Z_num, dtype=bool)
        for k in range(len(model_classes) - 1):
            mask |= Z_num == k - 1
            Z_num_mask = np.ma.array(Z_num, mask=mask)
            ax.contour(
                XX,
                YY,
                Z_num_mask,
                levels=[k + 0.5],
                linestyles="dashed",
                corner_mask=True,
                colors=[color],
                antialiased=True,
            )

    def get_regions(predict_fun, xy, shape, model_classes):
        Z_pred = predict_fun(xy).reshape(shape)
        cat2num = {cat: num for num, cat in enumerate(model_classes)}
        num2cat = {num: cat for num, cat in enumerate(model_classes)}
        vcat2num = np.vectorize(lambda x: cat2num[x])
        Z_num = vcat2num(Z_pred)
        return Z_num, num2cat

    def draw_regions(ax, model_classes, num2cat, Z_num):
        # Hack to get colors
        # TODO use legend_out = True
        slabels = [str(l) for l in model_classes]
        hdls, hlabels = ax.get_legend_handles_labels()
        hlabels_hdls = {l: h for l, h in zip(hlabels, hdls)}

        color_dict = {}
        for label in model_classes:
            if str(label) in hlabels_hdls:
                hdl = hlabels_hdls[str(label)]
                color = hdl.get_markerfacecolor()
                color_dict[label] = color
            else:
                raise Exception("No corresponding label found for ", label)

        colors = [color_dict[num2cat[i]] for i in range(len(model_classes))]
        cmap = mpl.colors.ListedColormap(colors)

        ax.imshow(
            Z_num,
            interpolation="nearest",
            extent=ax.get_xlim() + ax.get_ylim(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            alpha=0.2,
        )

    if isinstance(model, BaseEstimator):
        if model_classes is None:
            model_classes = model.classes_

        if levels is not None:
            if len(model.classes_) != 2:
                raise Exception("Lignes de niveaux supportées avec seulement deux classes")

            # Scikit-learn model, 2 classes + levels
            Z = model.predict_proba(xy)[:, 0].reshape(XX.shape)
            Z_num, num2cat = get_regions(model.predict, xy, XX.shape, model_classes)

            # Only 2 classes, simple contour
            ax.contour(
                XX,
                YY,
                Z,
                levels=levels,
                colors=[color]
            )

            draw_regions(ax, model_classes, num2cat, Z_num)
        else:
            # Scikit-learn model + no levels
            Z_num, num2cat = get_regions(model.predict, xy, XX.shape, model_classes)

            draw_boundaries(XX, YY, Z_num, color)
            if region:
                draw_regions(ax, model_classes, num2cat, Z_num)
    else:
        if model_classes is None:
            raise Exception("Il faut spécifier le nom des classes")
        if levels is not None:
            raise Exception("Lignes de niveaux avec fonction non supporté")

        # Model is a predict function, no levels
        Z_num, num2cat = get_regions(model, xy, XX.shape, model_classes)
        draw_boundaries(XX, YY, Z_num, color)
        if region:
            draw_regions(ax, model_classes, num2cat, Z_num)


# %%
from sklearn.utils import resample
tree_df = numerical_df[['in_spotify_playlists', 'streams', 'released_year']]
tree_df['in_spotify_playlists'] = tree_df['in_spotify_playlists'].apply(lambda x: 0 if x < 2200 else 1)

df_resampled1 = resample(tree_df, n_samples=200, replace=True)
df_resampled2 = resample(tree_df, n_samples=200, replace=True)
df_resampled3 = resample(tree_df, n_samples=200, replace=True)


# %%
DT1 = DecisionTreeClassifier(max_leaf_nodes=50)
DT1.fit(df_resampled1[['streams', 'released_year']], df_resampled1['in_spotify_playlists'])

DT2 = DecisionTreeClassifier(max_leaf_nodes=50)
DT2.fit(df_resampled2[['streams', 'released_year']], df_resampled2['in_spotify_playlists'])

DT3 = DecisionTreeClassifier(max_leaf_nodes=50)
DT3.fit(df_resampled3[['streams', 'released_year']], df_resampled3['in_spotify_playlists'])


# %%
def aggregating(X):
    y1 = DT1.predict(X)
    y2 = DT2.predict(X)
    y3 = DT3.predict(X)

    return np.where(y1 + y2 + y3 > 1, 1, 0)

# %%
axes = sns.scatterplot(data=tree_df, x='streams', y='released_year', hue='in_spotify_playlists')
axes.set_ylim(1900, 2024)


# %%
axes = sns.scatterplot(data=tree_df, x='streams', y='released_year', hue='in_spotify_playlists')
axes.set_ylim(1900, 2024)
add_decision_boundary(aggregating, model_classes=[0, 1])


# %%
axes = sns.scatterplot(data=tree_df, x='streams', y='released_year', hue='in_spotify_playlists')
axes.set_ylim(1900, 2024)
add_decision_boundary(DT1)


# %%
### Avec musical features
musical_tree = pd.DataFrame(musical_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
musical_tree['in_spotify_playlists'] = df['in_spotify_playlists'].apply(lambda x: 0 if x < 2200 else 1)
df_resampled1 = resample(musical_tree, n_samples=200, replace=True)
df_resampled2 = resample(musical_tree, n_samples=200, replace=True)
df_resampled3 = resample(musical_tree, n_samples=200, replace=True)


# %%
DT1 = DecisionTreeClassifier(max_leaf_nodes=50)
DT1.fit(df_resampled1[['PC1', 'PC2']], df_resampled1['in_spotify_playlists'])

DT2 = DecisionTreeClassifier(max_leaf_nodes=50)
DT2.fit(df_resampled2[['PC1', 'PC2']], df_resampled2['in_spotify_playlists'])

DT3 = DecisionTreeClassifier(max_leaf_nodes=50)
DT3.fit(df_resampled3[['PC1', 'PC2']], df_resampled3['in_spotify_playlists'])


# %%
axes = sns.scatterplot(data=musical_tree, x='PC1', y='PC2', hue='in_spotify_playlists')
add_decision_boundary(aggregating, model_classes=[0, 1])


# %%
DT1 = DecisionTreeClassifier(max_leaf_nodes=50)
DT1.fit(df_resampled1[['PC3', 'PC2']], df_resampled1['in_spotify_playlists'])

DT2 = DecisionTreeClassifier(max_leaf_nodes=50)
DT2.fit(df_resampled2[['PC3', 'PC2']], df_resampled2['in_spotify_playlists'])

DT3 = DecisionTreeClassifier(max_leaf_nodes=50)
DT3.fit(df_resampled3[['PC3', 'PC2']], df_resampled3['in_spotify_playlists'])


# %%
axes = sns.scatterplot(data=musical_tree, x='PC3', y='PC2', hue='in_spotify_playlists')
add_decision_boundary(aggregating, model_classes=[0, 1])


# %%
pca = PCA()
pca.fit(numerical_df)
pca_df = pca.transform(numerical_df)
pca_tree = pd.DataFrame(pca_df, columns=[f'PC{i}' for i in range(1, pca_df.shape[1] + 1)])
pca_tree['in_spotify_playlists'] = df['in_spotify_playlists'].apply(lambda x: 0 if x < 2200 else 1)
df_resampled1 = resample(pca_tree, n_samples=200, replace=True)
df_resampled2 = resample(pca_tree, n_samples=200, replace=True)
df_resampled3 = resample(pca_tree, n_samples=200, replace=True)


# %%
DT1 = DecisionTreeClassifier()
DT1.fit(df_resampled1[['PC1', 'PC2']], df_resampled1['in_spotify_playlists'])

DT2 = DecisionTreeClassifier()
DT2.fit(df_resampled2[['PC1', 'PC2']], df_resampled2['in_spotify_playlists'])

DT3 = DecisionTreeClassifier()
DT3.fit(df_resampled3[['PC1', 'PC2']], df_resampled3['in_spotify_playlists'])


# %%
axes = sns.scatterplot(data=pca_tree, x='PC1', y='PC2', hue='in_spotify_playlists')
add_decision_boundary(aggregating, model_classes=[0, 1])


# %%
DT1 = DecisionTreeClassifier()
DT1.fit(df_resampled1[['PC3', 'PC2']], df_resampled1['in_spotify_playlists'])

DT2 = DecisionTreeClassifier()
DT2.fit(df_resampled2[['PC3', 'PC2']], df_resampled2['in_spotify_playlists'])

DT3 = DecisionTreeClassifier()
DT3.fit(df_resampled3[['PC3', 'PC2']], df_resampled3['in_spotify_playlists'])


# %%
axes = sns.scatterplot(data=pca_tree, x='PC3', y='PC2', hue='in_spotify_playlists')
add_decision_boundary(aggregating, model_classes=[0, 1])


# %%
### Avec validation croisée
DT1 = DecisionTreeClassifier()
DT1.fit(pca_tree[['PC1', 'PC2']], pca_tree['in_spotify_playlists'])

DT2 = DecisionTreeClassifier()
DT2.fit(pca_tree[['PC3', 'PC2']], pca_tree['in_spotify_playlists'])

path1 = DT1.cost_complexity_pruning_path(pca_tree[['PC1', 'PC2']], pca_tree['in_spotify_playlists'])
path2 = DT2.cost_complexity_pruning_path(pca_tree[['PC3', 'PC2']], pca_tree['in_spotify_playlists'])

alphas1 = path1['ccp_alphas']
alphas2 = path2['ccp_alphas']


# %%
import math as m
alphas_mean1 = [m.sqrt(alphas1[i] * alphas1[i+1]) for i in range(len(alphas1) - 1)]
alphas_mean2 = [m.sqrt(alphas2[i] * alphas2[i+1]) for i in range(len(alphas2) - 1)]


# %%
# Cross validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y


def decision_tree_cross_validation_accuracies(X, y, n_folds, lambdas):
    X, y = check_X_y(X, y)

    # Création d'un object `KFold` pour la validation croisée
    kf = KFold(n_splits=n_folds)
    kf.get_n_splits(X)

    for _, (train_index, val_index) in enumerate(kf.split(X)):
        # Création de `X_train`, `y_train`, `X_val` et `y_val`
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        for k, lmb in enumerate(lambdas):
            # Création d'un arbre avec un coefficient coût-complexité
            # égal à `lmb`
            clf = DecisionTreeClassifier(ccp_alpha=lmb)

            # Apprentissage sur l'ensemble d'apprentissage et calcul
            # du taux de bonne classification sur l'ensemble de
            # validation
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)

            yield k, lmb, acc


# %%
gen = decision_tree_cross_validation_accuracies(pca_tree[['PC1', 'PC2']].values, pca_tree['in_spotify_playlists'].values, 5, alphas_mean1)
errors1 = pd.DataFrame(gen, columns=['k', 'lambda', 'accuracy'])

gen = decision_tree_cross_validation_accuracies(pca_tree[['PC3', 'PC2']].values, pca_tree['in_spotify_playlists'].values, 5, alphas_mean2)
errors2 = pd.DataFrame(gen, columns=['k', 'lambda', 'accuracy'])


# %%
### PC1, PC2
errors1.groupby('k').mean()
max_acc = errors1["accuracy"].max()
best = errors1[errors1["accuracy"] == max_acc]
min_lambda = best["lambda"].min()
best = best[best["lambda"] == min_lambda]

DT1 = DecisionTreeClassifier(ccp_alpha=min_lambda)
DT1.fit(pca_tree[['PC1', 'PC2']], pca_tree['in_spotify_playlists'])

sns.scatterplot(data=pca_tree, x='PC1', y='PC2', hue='in_spotify_playlists')
add_decision_boundary(DT1)


# %%
## PC3, PC2
errors2.groupby('k').mean()
max_acc = errors2["accuracy"].max()
best = errors2[errors2["accuracy"] == max_acc]
min_lambda = best["lambda"].min()
best = best[best["lambda"] == min_lambda]

DT2 = DecisionTreeClassifier(ccp_alpha=min_lambda)
DT2.fit(pca_tree[['PC3', 'PC2']], pca_tree['in_spotify_playlists'])

sns.scatterplot(data=pca_tree, x='PC3', y='PC2', hue='in_spotify_playlists')
add_decision_boundary(DT2)





# %%
### meme chose avec musical features
DT1 = DecisionTreeClassifier()
DT1.fit(musical_tree[['PC1', 'PC2']], musical_tree['in_spotify_playlists'])

DT2 = DecisionTreeClassifier()
DT2.fit(musical_tree[['PC3', 'PC2']], musical_tree['in_spotify_playlists'])

path1 = DT1.cost_complexity_pruning_path(musical_tree[['PC1', 'PC2']], musical_tree['in_spotify_playlists'])
path2 = DT2.cost_complexity_pruning_path(musical_tree[['PC3', 'PC2']], musical_tree['in_spotify_playlists'])

alphas1 = path1['ccp_alphas']
alphas2 = path2['ccp_alphas']

alphas_mean1 = [m.sqrt(alphas1[i] * alphas1[i+1]) for i in range(len(alphas1) - 1)]
alphas_mean2 = [m.sqrt(alphas2[i] * alphas2[i+1]) for i in range(len(alphas2) - 1)]

gen = decision_tree_cross_validation_accuracies(musical_tree[['PC1', 'PC2']].values, musical_tree['in_spotify_playlists'].values, 5, alphas_mean1)
errors1 = pd.DataFrame(gen, columns=['k', 'lambda', 'accuracy'])

gen = decision_tree_cross_validation_accuracies(musical_tree[['PC3', 'PC2']].values, musical_tree['in_spotify_playlists'].values, 5, alphas_mean2)
errors2 = pd.DataFrame(gen, columns=['k', 'lambda', 'accuracy'])


# %%	
### PC1, PC2
errors1.groupby('k').mean()
max_acc = errors1["accuracy"].max()
best = errors1[errors1["accuracy"] == max_acc]
min_lambda = best["lambda"].min()
best = best[best["lambda"] == min_lambda]

DT1 = DecisionTreeClassifier(ccp_alpha=min_lambda)
DT1.fit(musical_tree[['PC1', 'PC2']], musical_tree['in_spotify_playlists'])

sns.scatterplot(data=musical_tree, x='PC1', y='PC2', hue='in_spotify_playlists')
add_decision_boundary(DT1)


# %%
### PC3, PC2
errors2.groupby('k').mean()
max_acc = errors2["accuracy"].max()
best = errors2[errors2["accuracy"] == max_acc]
min_lambda = best["lambda"].min()
best = best[best["lambda"] == min_lambda]

DT2 = DecisionTreeClassifier(ccp_alpha=min_lambda)
DT2.fit(musical_tree[['PC3', 'PC2']], musical_tree['in_spotify_playlists'])

sns.scatterplot(data=musical_tree, x='PC3', y='PC2', hue='in_spotify_playlists')
add_decision_boundary(DT2)


# %%
