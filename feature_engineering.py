import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

df['FamilySize'] = df['SibSp'] + df['Parch']

print(df.head())