# packages -------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

np.random.seed(42)
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 18)

# dataset --------------------------------------------------------
df = pd.read_csv('sobar-72.csv')
X = df.drop(columns='ca_cervix')
y = df.loc[:, 'ca_cervix'].values
print('X shape: ', X.shape, '\ny shape: ', y.shape)

# pre-processing and models --------------------------------------
models = {"LR": LogisticRegression(),
          "NB": GaussianNB()}

# Lists for results stores
names = []
results = []
sc = 'roc_auc'  # 'roc_auc' or 'accuracy'

# Loop through models
for name, model in models.items():
    pipe = Pipeline([
        ("preProc", StandardScaler()),
        ("model", model)
    ])

    cv_res = cross_val_score(pipe, X, y, cv=10, scoring=sc)
    results.append(cv_res)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_res.mean(), cv_res.std()))

# boxplot algorithm comparison
# plt.rcParams['figure.figsize'] = [8, 8]
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)  # all on the same box
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.ylim(0, 1.1)
# plt.ylabel(sc.capitalize())
# plt.show()


