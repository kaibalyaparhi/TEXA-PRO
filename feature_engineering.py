import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

# 🔥 NEW FEATURE
df['FamilySize'] = df['SibSp'] + df['Parch']

# 🔥 Age group (extra improvement)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['AgeGroup'] = pd.cut(df['Age'],
                       bins=[0, 12, 18, 40, 60, 100],
                       labels=['Child', 'Teen', 'Adult', 'MidAge', 'Senior'])

print(df.head())