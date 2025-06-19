import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import pickle

# Load and clean data
df = pd.read_csv("diabetes.csv")
df.columns = df.columns.str.lower()
df = df.dropna()

# Split features and target
X = df.drop("outcome", axis=1)
y = df["outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Track best model
best_model = None
best_f1 = 0
best_name = ""

# Evaluate each model
for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_name = name

# Save only the best model
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\n✅ Best model saved: {best_name} (F1-score = {best_f1:.4f})")
