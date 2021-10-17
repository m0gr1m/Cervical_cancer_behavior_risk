# Packages ----
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset ----
df = pd.read_csv("sobar-72.csv")

target = df['ca_cervix'].map({1: 'Yes', 0: 'No'})
features = df.drop(columns='ca_cervix')

# Target ----
print(target.value_counts())

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_style(style="ticks", rc=custom_params)
plt.subplots(figsize=(8, 8))
sns.countplot(target)
plt.xlabel("")
plt.title("Does the patient have cervical cancer?")
plt.show()

# Features ----
