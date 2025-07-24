import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load test data
test_df = pd.read_csv("test.csv")  # Make sure test.csv is in the same folder
loan_ids = test_df['Loan_ID']

# Preprocess
test_df['Gender'] = test_df['Gender'].fillna('Male')
test_df['Dependents'] = test_df['Dependents'].fillna('0')
test_df['Self_Employed'] = test_df['Self_Employed'].fillna('No')
test_df['LoanAmount'] = test_df['LoanAmount'].fillna(test_df['LoanAmount'].mean())
test_df['Loan_Amount_Term'] = test_df['Loan_Amount_Term'].fillna(test_df['Loan_Amount_Term'].mode()[0])
test_df['Credit_History'] = test_df['Credit_History'].fillna(1.0)

# Encode categoricals
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
    test_df[col] = LabelEncoder().fit_transform(test_df[col])

# Predict
X = test_df.drop(columns=['Loan_ID'])
model = joblib.load("loan_model.pkl")
predictions = model.predict(X)

# Output
output_df = pd.DataFrame({
    "Loan_ID": loan_ids,
    "Loan_Status": ['Y' if p == 1 else 'N' for p in predictions]
})
output_df.to_csv("predictions.csv", index=False)
print("✅ Predictions saved to predictions.csv")
