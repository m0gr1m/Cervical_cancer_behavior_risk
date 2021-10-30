# Packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

np.random.seed(42)
sns.set()
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 18)

# dataset --------------------------------------------------------
df = pd.read_csv('sobar-72.csv')
X = df.drop(columns='ca_cervix')
y = df.loc[:, 'ca_cervix'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

print(" Shape of complete dataset")
print('X shape: ', X.shape, '\ny shape: ', y.shape)
print("")
print(" Shape of split dataset")
print('X_train shape: ', X_train.shape, '\ny_train shape: ', y_train.shape,
      '\nX_test shape: ', X_test.shape, '\ny_test shape: ', y_test.shape)
# Logistic Regression --------------------------------------------
pipe_lr = Pipeline([
    ("preProc", StandardScaler()),
    ("model", LogisticRegression())
])

model_1 = GridSearchCV(
    estimator=pipe_lr,
    param_grid={'model__class_weight': [{0: 1, 1: v} for v in np.linspace(1, 10, 20)],
                'model__C': np.logspace(-4, 4, 20)},
    cv=10,
    n_jobs=-1
)
model_1.fit(X_train, y_train)
y_pred_m1 = model_1.best_estimator_.predict(X_test)

print('')
print('Best model parameters')
print(model_1.best_params_)
print('')
print('Best model score (accuracy)')
print(model_1.best_score_)
print('-----------------------------------------------------')
print('Test data results')
print('')
print('Classification report')
print(classification_report(y_test, y_pred_m1))
print('')
print('AUC score')
print(roc_auc_score(y_test, y_pred_m1))
print('')
print('Accuracy score')
print(accuracy_score(y_test, y_pred_m1))
