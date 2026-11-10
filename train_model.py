from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Get project folder path
BASE_DIR = Path(__file__).resolve().parent

# Dataset path
DATA_PATH = BASE_DIR / "diabetes_prediction_dataset.csv"

# Model folder
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# Data cleaning
df = df.drop_duplicates().copy()
df = df.dropna().copy()

print("Dataset shape after cleaning:", df.shape)


# Separate features and target
target = "diabetes"

X = df.drop(columns=[target])
y = df[target]


# Identify categorical and numerical columns
categorical_features = ["gender", "smoking_history"]

numeric_features = [
    column for column in X.columns
    if column not in categorical_features
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Machine learning models
candidates = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )

}


# Train and evaluate models
results = []
trained = {}

for name, classifier in candidates.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    scores = {
        "Model": name,

        "Accuracy": accuracy_score(
            y_test, predictions
        ),

        "Precision": precision_score(
            y_test, predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test, predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test, predictions,
            zero_division=0
        ),

        "ROC AUC": roc_auc_score(
            y_test, probabilities
        )
    }

    results.append(scores)

    trained[name] = pipeline

    print("Model Performance:")
    print(scores)


# Compare model performance
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="ROC AUC",
    ascending=False
)


# Save comparison results
comparison.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False
)


# Select best model
best_name = comparison.iloc[0]["Model"]

best_model = trained[best_name]


# Save best model
joblib.dump(
    best_model,
    MODEL_DIR / "best_diabetes_model.pkl"
)


# Display final results
print("\n================================")
print("BEST MACHINE LEARNING MODEL")
print("================================")

print("Best Model:", best_name)

print("\nModel Comparison:")
print(comparison.to_string(index=False))

print("\n================================")
print("Files saved successfully!")
print("================================")

print("Model saved in:", MODEL_DIR)
print("Comparison saved in:", MODEL_DIR)