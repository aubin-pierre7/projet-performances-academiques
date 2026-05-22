from flask import Flask, render_template, request, jsonify
from config import Config
from database import init_db, get_db
import json
import os
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)

with app.app_context():
    init_db()

# ============================================================
# ROUTE 1 : Page d'accueil
# ============================================================
@app.route('/')
def index():
    df = pd.read_csv(Config.DATA_PATH, sep=';')
    stats = {
        'total_etudiants': len(df),
        'taux_reussite': round((df['G3'] >= 10).mean() * 100, 1),
        'moyenne_generale': round(df['G3'].mean(), 1),
        'moyenne_absences': round(df['absences'].mean(), 1),
    }
    return render_template('index.html', stats=stats)

# ============================================================
# ROUTE 2 : Page Dashboard
# ============================================================
@app.route('/dashboard')
def dashboard():
    graphs = []
    graphs_info = [
        ('01_distribution_notes.png', 'Distribution des notes G1, G2, G3'),
        ('02_taux_reussite.png', 'Taux de réussite global'),
        ('03_absences_vs_notes.png', 'Impact des absences sur les notes'),
        ('04_temps_etude.png', 'Impact du temps d\'étude'),
        ('05_correlation.png', 'Matrice de corrélation'),
        ('06_alcool_notes.png', 'Impact de la consommation d\'alcool'),
        ('07_education_parents.png', 'Impact de l\'éducation des parents'),
    ]
    for filename, titre in graphs_info:
        path = os.path.join(Config.GRAPHS_PATH, filename)
        if os.path.exists(path):
            graphs.append({'filename': filename, 'titre': titre})
    return render_template('dashboard.html', graphs=graphs)

# ============================================================
# ROUTE 3 : Page Prédiction
# ============================================================
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result = None
    if request.method == 'POST':
        try:
            student_data = {
                'age': int(request.form.get('age', 17)),
                'sex': int(request.form.get('sex', 0)),
                'address': int(request.form.get('address', 1)),
                'famsize': int(request.form.get('famsize', 1)),
                'Pstatus': int(request.form.get('Pstatus', 1)),
                'Medu': int(request.form.get('Medu', 2)),
                'Fedu': int(request.form.get('Fedu', 2)),
                'traveltime': int(request.form.get('traveltime', 1)),
                'studytime': int(request.form.get('studytime', 2)),
                'failures': int(request.form.get('failures', 0)),
                'schoolsup': int(request.form.get('schoolsup', 0)),
                'famsup': int(request.form.get('famsup', 1)),
                'paid': int(request.form.get('paid', 0)),
                'activities': int(request.form.get('activities', 0)),
                'higher': int(request.form.get('higher', 1)),
                'internet': int(request.form.get('internet', 1)),
                'romantic': int(request.form.get('romantic', 0)),
                'famrel': int(request.form.get('famrel', 4)),
                'freetime': int(request.form.get('freetime', 3)),
                'goout': int(request.form.get('goout', 2)),
                'Dalc': int(request.form.get('Dalc', 1)),
                'Walc': int(request.form.get('Walc', 1)),
                'health': int(request.form.get('health', 4)),
                'absences': int(request.form.get('absences', 4)),
                'G1': int(request.form.get('G1', 10)),
                'G2': int(request.form.get('G2', 10)),
            }

            from ml_models import predict_student
            result = predict_student(student_data)

            # Sauvegarder dans la base de données
            db = get_db()
            db.execute('''
                INSERT INTO predictions
                (date_prediction, absences, studytime, failures, goout,
                 Dalc, Walc, health, famrel, predicted_grade, risk_level, risk_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                student_data['absences'], student_data['studytime'],
                student_data['failures'], student_data['goout'],
                student_data['Dalc'], student_data['Walc'],
                student_data['health'], student_data['famrel'],
                result['predicted_grade'], result['risk_level'],
                result['risk_score']
            ))
            db.commit()
            db.close()

        except Exception as e:
            result = {'error': str(e)}

    return render_template('prediction.html', result=result)

# ============================================================
# ROUTE 4 : Page Alertes
# ============================================================
@app.route('/alertes')
def alertes():
    df = pd.read_csv('data/resultats_powerbi.csv', sep=';')
    eleves_rouge = df[df['risk_level'] == 'Rouge'].sort_values(
        'risk_score', ascending=False).head(20).to_dict('records')
    eleves_orange = df[df['risk_level'] == 'Orange'].sort_values(
        'risk_score', ascending=False).to_dict('records')
    stats_risque = {
        'rouge': len(df[df['risk_level'] == 'Rouge']),
        'orange': len(df[df['risk_level'] == 'Orange']),
        'vert': len(df[df['risk_level'] == 'Vert']),
        'total': len(df)
    }
    return render_template('alertes.html',
                           eleves_rouge=eleves_rouge,
                           eleves_orange=eleves_orange,
                           stats_risque=stats_risque)

# ============================================================
# ROUTE 5 : API JSON pour les stats
# ============================================================
@app.route('/api/stats')
def api_stats():
    df = pd.read_csv(Config.DATA_PATH, sep=';')
    data = {
        'total': len(df),
        'taux_reussite': round((df['G3'] >= 10).mean() * 100, 1),
        'moyenne_g3': round(df['G3'].mean(), 1),
        'moyenne_absences': round(df['absences'].mean(), 1),
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)