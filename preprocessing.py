import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

print("Missing Before:\n", df.isnull().sum())

# 🔥 Better handling (IMPROVED)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin (too many missing)
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

# Convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

print("\nMissing After:\n", df.isnull().sum())
print("\nCleaned Data:\n", df.head())