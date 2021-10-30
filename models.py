# Packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split

np.random.seed(42)
sns.set()
plt.style.use('seaborn-white')
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
y_proba_m1 = model_1.best_estimator_.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba_m1)

'''
G-Mean is a metric for imbalanced classification that, if optimized,
will seek a balance between the sensitivity and the specificity.

G-Mean = np.sqrt(Sensitivity * Specificity)
'''
gmeans_lr = np.sqrt(tpr * (1-fpr))
ix_lr = np.argmax(gmeans_lr)

y_hat_lr = np.where(y_proba_m1 >= thresholds[ix_lr], 1, 0)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, marker='.', label='Logistic Regression')
plt.scatter(fpr[ix_lr], tpr[ix_lr], marker='o', color='black', label='Best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LR ROC Curve')
plt.show()

print('')
print('-----------------------------------------------------')
print('')
print('Logistic Regression results')
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
print('-----------------------------------------------------')
print('Test data results with changed threshold')
print('')
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix_lr], gmeans_lr[ix_lr]))
print('')
print('Classification report')
print(classification_report(y_test, y_hat_lr))
print('')
print('AUC score')
print(roc_auc_score(y_test, y_hat_lr))
print('')
print('Accuracy score')
print(accuracy_score(y_test, y_hat_lr))

# categorical Naive Bayes ----------------------------------------

model_2 = GridSearchCV(
    estimator=CategoricalNB(),
    param_grid={'alpha': np.linspace(1e-10, 4, 60)},
    cv=10,
    n_jobs=-1
)
model_2.fit(X_train, y_train)

y_pred_m2 = model_2.predict(X_test)
y_proba_m2 = model_2.predict_proba(X_test)[:, 1]
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_proba_m2)


gmeans_nb = np.sqrt(tpr2 * (1-fpr2))
ix_nb = np.argmax(gmeans_nb)

y_hat_nb = np.where(y_proba_m2 >= thresholds2[ix_nb], 1, 0)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr2, tpr2, marker='.', label='Naive Bayes')
plt.scatter(fpr2[ix_nb], tpr2[ix_nb], marker='o', color='black', label='Best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('NB ROC Curve')
plt.show()

print('')
print('-----------------------------------------------------')
print('')
print('categorical Naive Bayes results')
print('')
print('Best model parameters')
print(model_2.best_params_)
print('')
print('Best model score (accuracy)')
print(model_2.best_score_)
print('-----------------------------------------------------')
print('Test data results')
print('')
print('Classification report')
print(classification_report(y_test, y_pred_m2))
print('')
print('AUC score')
print(roc_auc_score(y_test, y_pred_m2))
print('')
print('Accuracy score')
print(accuracy_score(y_test, y_pred_m2))
print('-----------------------------------------------------')
print('Test data results with changed threshold')
print('')
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds2[ix_nb], gmeans_nb[ix_nb]))
print('')
print('Classification report')
print(classification_report(y_test, y_hat_nb))
print('')
print('AUC score')
print(roc_auc_score(y_test, y_hat_nb))
print('')
print('Accuracy score')
print(accuracy_score(y_test, y_hat_nb))