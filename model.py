import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Titanic-Dataset.csv")

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch']

# Convert categorical
df = pd.get_dummies(df, drop_first=True)

# Remove NaN
df = df.dropna()

# Split
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 IMPROVED MODEL
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Improved Accuracy:", accuracy_score(y_test, y_pred))