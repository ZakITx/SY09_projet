
data.info()

# # Finding a predictive variable

# In[54]:


# Create a new DataFrame with relevant columns
relevant_columns = [
    "streams",
    "bpm",
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
    "in_spotify_playlists",
    "in_spotify_charts",
    "in_apple_playlists",
    "in_deezer_playlists",
    "in_shazam_charts",
]
df_relevant = data[relevant_columns].copy()
# Convert 'streams' and other relevant columns to numeric
df_relevant["streams"] = pd.to_numeric(df_relevant["streams"], errors="coerce")
df_relevant["in_deezer_playlists"] = pd.to_numeric(
    df_relevant["in_deezer_playlists"], errors="coerce"
)
df_relevant["in_shazam_charts"] = pd.to_numeric(
    df_relevant["in_shazam_charts"], errors="coerce"
)


# Drop rows with missing values in relevant columns
df_relevant = df_relevant.dropna()
# Define the independent variables (X) and the dependent variable (y)
X = df_relevant[
    [
        "bpm",
        "danceability_%",
        "valence_%",
        "energy_%",
        "acousticness_%",
        "instrumentalness_%",
        "liveness_%",
        "speechiness_%",
        "in_spotify_playlists",
        "in_spotify_charts",
        "in_apple_playlists",
        "in_deezer_playlists",
        "in_shazam_charts",
    ]
]
y = df_relevant["streams"]

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())


# In[86]:


def calculate_and_plot_correlation(df, col1, col2, plot=True):

    df.loc[:, col1] = pd.to_numeric(df[col1], errors="coerce")
    df.loc[:, col2] = pd.to_numeric(df[col2], errors="coerce")

    # Drop rows with missing values in the specified columns
    df_clean = df.dropna(subset=[col1, col2])

    # Calculate the correlation
    correlation = df_clean[col1].corr(df_clean[col2])

    # Print the correlation coefficient
    print(f"Correlation between {col1} and {col2}: {correlation}")

    # Optionally plot the data
    if plot:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_clean, x=col1, y=col2)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f"Correlation between {col1} and {col2}")
        plt.show()

    return correlation


# In[133]:


data = data.dropna()
data.loc[:, "streams"] = pd.to_numeric(data["streams"], errors="coerce")
data.loc[:, "in_deezer_playlists"] = pd.to_numeric(
    data["in_deezer_playlists"], errors="coerce"
)
data.loc[:, "in_shazam_charts"] = pd.to_numeric(
    data["in_shazam_charts"], errors="coerce"
)
data.loc[:, "mode"] = [0 if x == "Minor" else 1 for x in data["mode"]]
data["mode"] = data["mode"].dropna()
data["mode"] = data["mode"].astype(int)
data["streams"] = data["streams"].astype(int)
data["in_deezer_playlists"] = data["in_deezer_playlists"].astype(int)
data["in_shazam_charts"] = data["in_shazam_charts"].astype(int)


# In[134]:


print(data.dtypes)


# In[136]:


# Example usage:
correlation = calculate_and_plot_correlation(data, "streams", "mode", True)


# In[120]:


def compute_correlation_matrix(df):
    # Select only integer columns
    int_df = df.select_dtypes(include="int64")

    # Compute the correlation matrix
    correlation_matrix = int_df.corr()

    return correlation_matrix


correlation_matrix = compute_correlation_matrix(data)
print(correlation_matrix)


# ## Correlation between Variables ##
#
# 1. (streams, in_spotify_playlists): **0.76**
# 2. (in_spotify_playlists, in_apple_playlists): **0.70**
# 3. (in_spotify_playlists, in_deezer_playlists): **0.79**
# 4. (streams, in_deezer_playlists): **0.71**
# 5. (streams, in_apple_playlists): **0.67**
#
