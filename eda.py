import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic-Dataset.csv")

df['Survived'].value_counts().plot(kind='bar')
plt.show()

df.groupby('Sex')['Survived'].mean().plot(kind='bar')
plt.show()