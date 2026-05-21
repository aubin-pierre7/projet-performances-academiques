import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = 'mon_projet_performances_academiques_2024'
    DATABASE = os.path.join(BASE_DIR, 'instance', 'database.db')
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'student-mat.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
    GRAPHS_PATH = os.path.join(BASE_DIR, 'static', 'graphs')