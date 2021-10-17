import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt

df = pd.read_csv("sobar-72.csv")

# Checking the percentage of missing values
print(np.round(df.isnull().sum()/len(df)*100, 2))

# Visualization
md_plot = msno.matrix(df, figsize=[8, 8], fontsize=10)
md_dendrogram = msno.dendrogram(df, figsize=[8, 8], fontsize=10)
plt.show()
