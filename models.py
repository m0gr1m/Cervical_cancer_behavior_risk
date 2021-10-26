# Packages ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

sns.set()
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 18)

# dataset --------------------------------------------------------
df = pd.read_csv('sobar-72.csv')
X = df.drop(columns='ca_cervix')
y = df.loc[:, 'ca_cervix'].values
print('X shape: ', X.shape, '\ny shape: ', y.shape)
