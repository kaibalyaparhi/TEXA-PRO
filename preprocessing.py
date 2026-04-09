import pandas as pd

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("Before Cleaning:\n", df.isnull().sum())

# Handle missing values
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].mean())

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin (too many missing values)
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

# Convert categorical to numeric
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nAfter Cleaning:\n", df.isnull().sum())
print("\nCleaned Data:\n", df.head())