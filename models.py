# Packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

sns.set()
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 18)

# Dataset ----
df = pd.read_csv("sobar-72.csv")

y = df['ca_cervix'].values
X = df.drop(columns='ca_cervix')

print(y)
print(X.head())
