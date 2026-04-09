import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# -------------------------------
# 🔹 Data Preprocessing (Improved)
# -------------------------------

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin (too many missing values)
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

# -------------------------------
# 🔹 Feature Engineering
# -------------------------------

# Create Family Size
df['FamilySize'] = df['SibSp'] + df['Parch']

# -------------------------------
# 🔹 Convert categorical to numeric
# -------------------------------

df = pd.get_dummies(df, drop_first=True)

# Remove any remaining missing values
df = df.dropna()

# -------------------------------
# 🔹 Split Data
# -------------------------------

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 🔹 Model Training (Improved)
# -------------------------------

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 🔹 Prediction
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# 🔹 Evaluation
# -------------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))