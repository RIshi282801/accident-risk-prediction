import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import load_data, create_preprocessor


# Load dataset
X, y = load_data()

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create preprocessing pipeline
preprocessor = create_preprocessor(X)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
# Create Decision Tree model
decision_tree = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)
# Create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Combine preprocessing and model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)
print("Random Forest Accuracy:",
      accuracy_score(y_test, pipeline.predict(X_test)))
# Train Decision Tree
decision_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", decision_tree)
])

decision_pipeline.fit(X_train, y_train)

decision_pred = decision_pipeline.predict(X_test)

print("Decision Tree Accuracy:",
      accuracy_score(y_test, decision_pred))
# Train Logistic Regression
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", logistic_model)
])

logistic_pipeline.fit(X_train, y_train)

logistic_pred = logistic_pipeline.predict(X_test)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, logistic_pred))

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Save the trained pipeline
joblib.dump(pipeline, "models/accident_risk_model.pkl")

print("\nModel saved successfully!")