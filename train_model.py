import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("perfect_freelancer_subset.csv")

# Define features and target
categorical_features = ['Country', 'Freelancer_Platform']
numerical_features = ['Experience', 'Hours_Per_Week']
target = 'Earnings'

# Handle missing values
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop="first")
categorical_encoded = encoder.fit_transform(df[categorical_features])
categorical_columns = encoder.get_feature_names_out(categorical_features)
categorical_df = pd.DataFrame(categorical_encoded, columns=categorical_columns)

# Combine processed features
X = pd.concat([df[numerical_features], categorical_df], axis=1)
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
r2 = r2_score(y_test, model.predict(X_test))
print(f"Model R² Score: {r2:.4f}")

# Save model (including encoder for future predictions)
joblib.dump((model, encoder), "model.pkl")
