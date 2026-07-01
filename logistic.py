import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Load dataset
df = pd.read_csv("data.csv")

# 2. Missing values ko average (mean) se fill kiya
df = df.fillna(df.mean())

# 3. Features (X)
X = df[['Pulse', 'Maxpulse', 'Calories']]

# 4. Target (y) ko Classification ke liye ready karein
# Agar Duration >= 60 hai toh 1 ban jayega, nahi toh 0
y = (df['Duration'] >= 60).astype(int)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)   #data ek he tarike sA Split ho taki sAME RESult mile

# 6. Logistic Regression Model Banayein
model = LogisticRegression(max_iter=1000)

# 7. Model ko Train karein
model.fit(X_train, y_train)

# 8. Predictions karein
y_pred = model.predict(X_test)

# 9. Task ki Requirement: Accuracy aur Confusion Matrix Print karein
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


