import pandas as pd
import numpy as np
from config import Config

def load_and_clean_data():
    # Charger le CSV avec séparateur point-virgule
    df = pd.read_csv(Config.DATA_PATH, sep=';')

    print(f"Dataset chargé : {df.shape[0]} élèves, {df.shape[1]} colonnes")
    print(f"Valeurs manquantes : {df.isnull().sum().sum()}")

    # Encoder les colonnes binaires oui/non
    binary_cols = ['schoolsup', 'famsup', 'paid', 'activities',
                   'nursery', 'higher', 'internet', 'romantic']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Encoder les colonnes catégorielles
    df['sex'] = df['sex'].map({'M': 1, 'F': 0})
    df['address'] = df['address'].map({'U': 1, 'R': 0})
    df['famsize'] = df['famsize'].map({'GT3': 1, 'LE3': 0})
    df['Pstatus'] = df['Pstatus'].map({'T': 1, 'A': 0})

    # Encoder les colonnes avec plusieurs catégories
    df = pd.get_dummies(df, columns=['Mjob', 'Fjob', 'reason', 'guardian'],
                        drop_first=True)
    df['school'] = df['school'].map({'GP': 1, 'MS': 0})

    # Créer la colonne cible binaire : 1 = réussite (G3 >= 10), 0 = échec
    df['pass'] = (df['G3'] >= 10).astype(int)

    print(f"Taux de réussite : {df['pass'].mean()*100:.1f}%")
    print("Nettoyage terminé.")

    return df

def get_features_and_target(df):
    # Variables utilisées pour la prédiction
    feature_cols = ['age', 'sex', 'address', 'famsize', 'Pstatus',
                    'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                    'schoolsup', 'famsup', 'paid', 'activities', 'higher',
                    'internet', 'romantic', 'famrel', 'freetime', 'goout',
                    'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']

    # Garder seulement les colonnes qui existent dans le dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y_regression = df['G3']
    y_classification = df['pass']

    return X, y_regression, y_classification, feature_cols

if __name__ == '__main__':
    df = load_and_clean_data()
    print(df.head())
    print(df.dtypes)