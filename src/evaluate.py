from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import pandas as pd


def evaluate_model(model, X_test, y_test, model_name="Model"):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {model_name} =====")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))


def show_feature_importance(model, feature_names):
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print("\nTop 10 Important Features:")
    print(importance.head(10))
