import pandas as pd

df = pd.read_csv('data/credit_risk_dataset.csv')

# First 5 columns of dataset
print(df.head())

# Dataset description
print(df.describe())

# Dataset info
df.info()

# NaN values
print(df.isna().sum())

# NaN filling
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].mean())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())

# Encode categorical columns
categorical_columns = df.select_dtypes(include='object').columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_columns)

print(df_encoded.head())

# Correlation matrix
print(df_encoded.corr())