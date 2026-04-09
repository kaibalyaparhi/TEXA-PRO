import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic-Dataset.csv")

# Survival count
df['Survived'].value_counts().plot(kind='bar')
plt.title("Survival Count")
plt.show()

# Gender vs survival
df.groupby('Sex')['Survived'].mean().plot(kind='bar')
plt.title("Survival by Gender")
plt.show()

# 🔥 NEW: Class vs survival (IMPROVEMENT)
df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
plt.title("Survival by Passenger Class")
plt.show()

# Age distribution
df['Age'].hist(bins=20)
plt.title("Age Distribution")
plt.show()