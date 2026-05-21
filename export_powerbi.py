import pandas as pd
import joblib
import os
from data_processing import load_and_clean_data, get_features_and_target
from config import Config

def export_for_powerbi():
    print("Export des données pour Power BI...")

    df_raw = pd.read_csv(Config.DATA_PATH, sep=';')
    df_clean = load_and_clean_data()
    X, y_reg, y_clf, feature_cols = get_features_and_target(df_clean)

    model = joblib.load(Config.MODEL_PATH)
    predictions = model.predict(X)
    probas = model.predict_proba(X)[:, 0]

    df_export = df_raw[['age', 'sex', 'studytime', 'failures',
                         'absences', 'G1', 'G2', 'G3']].copy()
    df_export['predicted_pass'] = predictions
    df_export['risk_score'] = probas.round(3)
    df_export['risk_level'] = df_export['risk_score'].apply(
        lambda x: 'Rouge' if x >= 0.7 else ('Orange' if x >= 0.4 else 'Vert'))
    df_export['pass_actual'] = (df_raw['G3'] >= 10).astype(int)

    os.makedirs('data', exist_ok=True)
    df_export.to_csv('data/resultats_powerbi.csv', index=False, sep=';')
    print(f"Fichier exporté : data/resultats_powerbi.csv ({len(df_export)} lignes)")
    print(f"Répartition des risques :\n{df_export['risk_level'].value_counts()}")

if __name__ == '__main__':
    export_for_powerbi()