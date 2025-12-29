from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_models(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -------- Logistic Regression (with scaling) --------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)

    # -------- Random Forest (NO scaling) --------
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(rf, "models/churn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return {
        "lr": lr,
        "rf": rf,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test
    }
