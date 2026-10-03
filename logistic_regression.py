"""Run in Google Colab and upload dl_practical.csv when prompted."""

import pandas as pd
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
)

# 1. Upload and load the CSV
uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_csv(filename, header=None)

# Remove the first row if it contains column names.
if pd.to_numeric(df.iloc[0], errors="coerce").isna().any():
    df = df.iloc[1:].reset_index(drop=True)
df = df.apply(pd.to_numeric, errors="raise")

# 2. First 10 columns = features; 11th column = label
X = df.iloc[:, :10]
y = df.iloc[:, 10].astype(int)

# 3. Split into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 4. Fit the scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Predict on the test set
predictions = model.predict(X_test_scaled)

# 7. Display results
f1 = f1_score(y_test, predictions)
errors = int((predictions != y_test).sum())

print("\nLOGISTIC REGRESSION RESULTS")
print(f"Test examples:     {len(y_test):,}")
print(f"Test F1:           {f1:.2%}")
print(f"Precision:         {precision_score(y_test, predictions):.2%}")
print(f"Recall:            {recall_score(y_test, predictions):.2%}")
print(f"Accuracy:          {accuracy_score(y_test, predictions):.2%}")
print(f"Prediction errors: {errors:,}")
print(f"Meets 80% target:   {f1 >= 0.80}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
