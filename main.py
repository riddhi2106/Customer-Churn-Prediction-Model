import pandas as pd
from src.eda import run_eda
from src.preprocess import preprocess_data
from src.train import train_models
from src.evaluate import evaluate_model, show_feature_importance

# Load data
df = pd.read_csv("data/Churn_Modelling.csv")

print(df.head())
print(df.shape)
print(df.info())

# EDA
run_eda(df)

# Preprocessing
df = preprocess_data(df)

X = df.drop("Exited", axis=1)
y = df["Exited"]

# Training
artifacts = train_models(X, y)

# Evaluation
evaluate_model(
    artifacts["lr"],
    artifacts["X_test_scaled"],
    artifacts["y_test"],
    model_name="Logistic Regression"
)

evaluate_model(
    artifacts["rf"],
    artifacts["X_test"],
    artifacts["y_test"],
    model_name="Random Forest"
)

# Feature importance
show_feature_importance(
    artifacts["rf"],
    X.columns
)
