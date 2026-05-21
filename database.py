import sqlite3
import os
from config import Config

def get_db():
    os.makedirs(os.path.dirname(Config.DATABASE), exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(Config.DATABASE), exist_ok=True)
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            school TEXT, sex TEXT, age INTEGER,
            address TEXT, famsize TEXT, Pstatus TEXT,
            Medu INTEGER, Fedu INTEGER, Mjob TEXT, Fjob TEXT,
            reason TEXT, guardian TEXT, traveltime INTEGER,
            studytime INTEGER, failures INTEGER, schoolsup TEXT,
            famsup TEXT, paid TEXT, activities TEXT, nursery TEXT,
            higher TEXT, internet TEXT, romantic TEXT,
            famrel INTEGER, freetime INTEGER, goout INTEGER,
            Dalc INTEGER, Walc INTEGER, health INTEGER,
            absences INTEGER, G1 INTEGER, G2 INTEGER, G3 INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_prediction TEXT,
            absences INTEGER,
            studytime INTEGER,
            failures INTEGER,
            goout INTEGER,
            Dalc INTEGER,
            Walc INTEGER,
            health INTEGER,
            famrel INTEGER,
            predicted_grade REAL,
            risk_level TEXT,
            risk_score REAL
        )
    ''')

    conn.commit()
    conn.close()
    print("Base de données créée avec succès.")

if __name__ == '__main__':
    init_db()