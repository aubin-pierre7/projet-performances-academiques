from flask import Flask
from config import Config
from database import init_db

app = Flask(__name__)
app.config.from_object(Config)

with app.app_context():
    init_db()

@app.route('/')
def index():
    return "Projet Performances Académiques — Serveur OK"

if __name__ == '__main__':
    app.run(debug=True)