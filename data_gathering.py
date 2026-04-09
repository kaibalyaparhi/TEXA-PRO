
import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.info())
print(df.describe())