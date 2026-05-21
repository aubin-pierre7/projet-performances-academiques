import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score,
                              accuracy_score, classification_report,
                              confusion_matrix)
from config import Config
from data_processing import load_and_clean_data, get_features_and_target

def train_models():
    print("=== ENTRAÎNEMENT DES MODÈLES ML ===\n")

    df = load_and_clean_data()
    X, y_reg, y_clf, feature_cols = get_features_and_target(df)

    # Division train/test 80-20
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Modèle 1 : Régression linéaire ---
    print("1. Régression Linéaire (prédiction de la note G3)")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_reg)
    y_pred_lr = lr.predict(X_test_scaled)
    rmse_lr = np.sqrt(mean_squared_error(y_test_reg, y_pred_lr))
    r2_lr = r2_score(y_test_reg, y_pred_lr)
    print(f"   RMSE : {rmse_lr:.2f} | R² : {r2_lr:.2f}")
    results['linear_regression'] = {'rmse': rmse_lr, 'r2': r2_lr}

    # --- Modèle 2 : Decision Tree ---
    print("\n2. Decision Tree (classification réussite/échec)")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train_clf)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test_clf, y_pred_dt)
    print(f"   Accuracy : {acc_dt*100:.1f}%")
    print(classification_report(y_test_clf, y_pred_dt,
          target_names=['Échec', 'Réussite']))
    results['decision_tree'] = {'accuracy': acc_dt}

    # --- Modèle 3 : Random Forest ---
    print("\n3. Random Forest (classification réussite/échec)")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train_clf)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test_clf, y_pred_rf)
    print(f"   Accuracy : {acc_rf*100:.1f}%")
    print(classification_report(y_test_clf, y_pred_rf,
          target_names=['Échec', 'Réussite']))
    results['random_forest'] = {'accuracy': acc_rf}

    # --- Sauvegarder le meilleur modèle ---
    best_model = rf if acc_rf >= acc_dt else dt
    best_name = 'Random Forest' if acc_rf >= acc_dt else 'Decision Tree'
    print(f"\nMeilleur modèle : {best_name} ({max(acc_rf, acc_dt)*100:.1f}% accuracy)")

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, Config.MODEL_PATH)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    print(f"Modèle sauvegardé dans : {Config.MODEL_PATH}")

    return results, feature_cols

def predict_student(student_data: dict):
    """
    Prédit le niveau de risque d'un élève.
    Retourne : note prédite, niveau de risque, score de risque
    """
    model = joblib.load(Config.MODEL_PATH)
    scaler = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')

    import pandas as pd
    df_input = pd.DataFrame([student_data])

    # Aligner les colonnes avec celles du modèle
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_cols]

    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]
    risk_score = proba[0]  # probabilité d'échec

    if risk_score >= 0.7:
        risk_level = 'rouge'
    elif risk_score >= 0.4:
        risk_level = 'orange'
    else:
        risk_level = 'vert'

    predicted_grade = round(10 + (1 - risk_score) * 10, 1)

    return {
        'predicted_pass': int(prediction),
        'risk_score': round(risk_score, 3),
        'risk_level': risk_level,
        'predicted_grade': predicted_grade
    }

if __name__ == '__main__':
    train_models()