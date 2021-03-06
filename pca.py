# Package ---------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 18)

# Dataset ---------------------------------------------------------------------------------
df = pd.read_csv("sobar-72.csv")
target = df['ca_cervix'].map({1: 'Yes', 0: 'No'})
features = df.drop(columns='ca_cervix')
print('Shape')
print('Target: ', target.shape, 'Features: ', features.shape)

# KMO (Kaiser-Meyer-Olkin test) -----------------------------------------------------------
kmo_per_variable, kmo_total = calculate_kmo(features)
print("")
print("KMO Test")
print("per variable:", kmo_per_variable, "total:", kmo_total)

count = 0
to_keep = []
for i in range(0, kmo_per_variable.size):
    if kmo_per_variable[i] >= 0.0:
        to_keep.append(count)
    count += 1

print("")
print("To keep:", to_keep)

X = features.iloc[:, to_keep]
print("")
print("Shape of X: ", X.shape)

# Scale -----------------------------------------------------------------------------------
X = scale(X)
features = scale(features)

# PCA for all features --------------------------------------------------------------------
pca_0 = PCA(n_components=np.shape(features)[1])
pca_0.fit(features)
pca_0.transform(features)

plt.figure(figsize=(10, 8))
plt.plot(pca_0.explained_variance_ratio_,
         marker='o', color='black')
plt.title('Elbow graph')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.xticks(np.arange(0, np.shape(features)[1], 1))
plt.show()

# PCA -------------------------------------------------------------------------------------
pca = PCA(n_components=3)
pca.fit(X)
scores = pca.transform(X)

scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3'])
df_scores = pd.concat([scores_df, target], axis=1)
print("")
print("Final DF")
print(df_scores)

# Explained variance
explained_var = pca.explained_variance_ratio_
print("")
print("Explained variance")
print(explained_var)

# Plot for variance ------------------------------------------------------------------------
explained_var = np.insert(explained_var, 0, 0)
cumulative_var = np.cumsum(np.round(explained_var, decimals=3))
pc_df = ['', 'PC1', 'PC2', 'PC3']

# Explained variance
plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)  # one row, two columns - first plot
plt.bar(x=pc_df, height=explained_var,
        fc='lightgray', ec='black')

# values for bars
for i in range(len(pc_df)):
    plt.text(i, explained_var[i], np.round(explained_var[i], decimals=3), ha='center', va='bottom')

plt.title("Explained variance")
plt.ylim(0, 1)  # from o to 100%

# Cumulative variance
plt.subplot(1, 2, 2)  # one row, two columns - second plot
plt.bar(x=pc_df, height=cumulative_var,
        fc='lightgray', ec='black')

# values for bars
for i in range(len(pc_df)):
    plt.text(i, cumulative_var[i], np.round(cumulative_var[i], decimals=3), ha='center', va='bottom')

plt.title("Cumulative variance")
plt.ylim(0, 1)  # from o to 100%
plt.show()

# Final plot ---------------------------------------------------------------------------
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_scores, x='PC1', y='PC2',
                hue='ca_cervix', palette=['red', 'green'], alpha=0.8)
plt.title('PCA Plot')
plt.show()
#
# Plot 3D ------------------------------------------------------------------------------
fig = px.scatter_3d(data_frame=df_scores, x='PC1', y="PC2", z='PC3',
                    color='ca_cervix', color_discrete_sequence=['red', 'green'])
fig.show()

# PCA - case inputs
components_imp = pd.DataFrame(abs(pca.components_), columns=[df.drop(columns=['ca_cervix']).columns.values.tolist()]).T
components_imp.columns = ['PC1', 'PC2', 'PC3']

print('')
print('Components importance')
print(components_imp)