# Packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
from factor_analyzer.factor_analyzer import calculate_kmo

sns.set()
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 18)

# Dataset ----
df = pd.read_csv("sobar-72.csv")

print("Types of data:")
print(df.dtypes)

target = df['ca_cervix'].map({1: 'Yes', 0: 'No'})
features = df.drop(columns='ca_cervix')

# Target ----
print("")
print("Target:")
print(target.value_counts())
print("")
print("Percentage of people with cancer: %.2f%%" % (df['ca_cervix'].sum()/len(df['ca_cervix'])*100))

# Plot
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_style(style="ticks", rc=custom_params)
plt.subplots(figsize=(8, 8))
sns.countplot(x=target)
plt.xlabel("")
plt.title("Does the patient have cervical cancer?")
plt.show()

# Features ----

# 1 - descriptive statistics
print("")
print(features.describe().T)


# 2 - Coefficient of variation
def cv(x): return np.std(x) / np.mean(x) * 100


print("")
print("Coefficient of variation")
print(features.apply(cv))

# 3 - Kurtosis
print("")
print("Kurtosis")
print(features.apply(kurtosis, bias=False))

# 4 - Skewness
print("")
print("Skewness")
print(features.apply(skew, bias=False))

# Box plots
sns.set_style(style="ticks")
plt.subplots(figsize=(18, 10))
sns.boxplot(data=features,
            orient="h",
            color="y",
            whis=1.5)
plt.show()

# 5 - outliers
Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)
IQR = Q3 - Q1
print("")
print("Standard outliers")
print(((features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))).sum())
print("")
print("Extreme outliers")
print(((features < (Q1 - 3 * IQR)) | (features > (Q3 + 3 * IQR))).sum())

# 6 - correlation
plt.figure(figsize=(16, 16))
sns.heatmap(features.corr(),
            annot=True,
            cmap='coolwarm',
            vmax=1, vmin=-1,  # cause corr can be in range from 1 to -1
            mask=np.triu(features.corr())).set_title("Correlogram")
plt.show()

# 7 - KMO (Kaiser-Meyer-Olkin test)
kmo_per_variable, kmo_total = calculate_kmo(features)
print("")
print("KMO Test")
print("per variable:", kmo_per_variable, "total:", kmo_total)