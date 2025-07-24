import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load your dataset
df = pd.read_csv("train.csv")

# Fill missing values if any
df.fillna(method='ffill', inplace=True)

# Encode categorical features using one-hot encoding
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features and target
X = df.drop(['id', 'loan_status'], axis=1)
y = df['loan_status']  # 0 or 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "loan_model.pkl")
print("✅ Model saved as loan_model.pkl")
