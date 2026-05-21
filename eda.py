import matplotlib
matplotlib.use('Agg')  # Mode sans interface graphique
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from config import Config
from data_processing import load_and_clean_data

# Style global des graphiques
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

os.makedirs(Config.GRAPHS_PATH, exist_ok=True)

def save_fig(filename):
    path = os.path.join(Config.GRAPHS_PATH, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : {filename}")

def graph_distribution_notes(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Distribution des notes (G1, G2, G3)', fontsize=14, fontweight='bold')
    for i, (col, ax) in enumerate(zip(['G1', 'G2', 'G3'], axes)):
        ax.hist(df[col], bins=20, color=COLORS[i], edgecolor='white', alpha=0.85)
        ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Moyenne: {df[col].mean():.1f}')
        ax.set_title(f'Note {col}')
        ax.set_xlabel('Note')
        ax.set_ylabel('Nombre d\'élèves')
        ax.legend()
    save_fig('01_distribution_notes.png')

def graph_taux_reussite(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Taux de réussite', fontsize=14, fontweight='bold')
    counts = df['pass'].value_counts()
    labels = ['Échec (G3 < 10)', 'Réussite (G3 ≥ 10)']
    axes[0].pie(counts, labels=labels, autopct='%1.1f%%',
                colors=[COLORS[3], COLORS[0]], startangle=90)
    axes[0].set_title('Répartition réussite/échec')
    axes[1].bar(['Échec', 'Réussite'], counts.values,
                color=[COLORS[3], COLORS[0]], edgecolor='white')
    axes[1].set_title('Nombre d\'élèves')
    axes[1].set_ylabel('Nombre d\'élèves')
    save_fig('02_taux_reussite.png')

def graph_absences_vs_notes(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Impact des absences sur les notes', fontsize=14, fontweight='bold')
    axes[0].scatter(df['absences'], df['G3'],
                    alpha=0.5, color=COLORS[0], edgecolors='white', linewidth=0.5)
    axes[0].set_xlabel('Nombre d\'absences')
    axes[0].set_ylabel('Note finale (G3)')
    axes[0].set_title('Absences vs Note finale')
    df_group = df.groupby('absences')['G3'].mean().reset_index()
    axes[1].plot(df_group['absences'], df_group['G3'],
                 color=COLORS[2], linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel('Nombre d\'absences')
    axes[1].set_ylabel('Note moyenne')
    axes[1].set_title('Moyenne des notes par nombre d\'absences')
    save_fig('03_absences_vs_notes.png')

def graph_temps_etude(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Impact du temps d\'étude', fontsize=14, fontweight='bold')
    labels_temps = ['< 2h', '2-5h', '5-10h', '> 10h']
    moyennes = df.groupby('studytime')['G3'].mean()
    axes[0].bar(labels_temps[:len(moyennes)], moyennes.values,
                color=COLORS[0], edgecolor='white')
    axes[0].set_title('Note moyenne par temps d\'étude')
    axes[0].set_xlabel('Temps d\'étude hebdomadaire')
    axes[0].set_ylabel('Note moyenne (G3)')
    taux = df.groupby('studytime')['pass'].mean() * 100
    axes[1].bar(labels_temps[:len(taux)], taux.values,
                color=COLORS[1], edgecolor='white')
    axes[1].set_title('Taux de réussite par temps d\'étude')
    axes[1].set_xlabel('Temps d\'étude hebdomadaire')
    axes[1].set_ylabel('Taux de réussite (%)')
    save_fig('04_temps_etude.png')

def graph_correlation(df):
    cols_num = ['G1', 'G2', 'G3', 'absences', 'studytime',
                'failures', 'goout', 'Dalc', 'Walc', 'health', 'famrel']
    cols_num = [c for c in cols_num if c in df.columns]
    corr = df[cols_num].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    plt.title('Matrice de corrélation entre les variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('05_correlation.png')

def graph_alcool_notes(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Impact de la consommation d\'alcool', fontsize=14, fontweight='bold')
    dalc_means = df.groupby('Dalc')['G3'].mean()
    walc_means = df.groupby('Walc')['G3'].mean()
    axes[0].bar(dalc_means.index, dalc_means.values,
                color=COLORS[3], edgecolor='white')
    axes[0].set_title('Alcool en semaine vs Note')
    axes[0].set_xlabel('Consommation (1=faible, 5=élevée)')
    axes[0].set_ylabel('Note moyenne (G3)')
    axes[1].bar(walc_means.index, walc_means.values,
                color=COLORS[2], edgecolor='white')
    axes[1].set_title('Alcool le week-end vs Note')
    axes[1].set_xlabel('Consommation (1=faible, 5=élevée)')
    axes[1].set_ylabel('Note moyenne (G3)')
    save_fig('06_alcool_notes.png')

def graph_education_parents(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Impact du niveau d\'éducation des parents', fontsize=14, fontweight='bold')
    labels_edu = ['Aucun', 'Primaire', 'Collège', 'Lycée', 'Supérieur']
    medu_means = df.groupby('Medu')['G3'].mean()
    fedu_means = df.groupby('Fedu')['G3'].mean()
    axes[0].bar(labels_edu[:len(medu_means)], medu_means.values,
                color=COLORS[0], edgecolor='white')
    axes[0].set_title('Éducation mère vs Note')
    axes[0].set_xlabel('Niveau d\'éducation')
    axes[0].set_ylabel('Note moyenne (G3)')
    axes[1].bar(labels_edu[:len(fedu_means)], fedu_means.values,
                color=COLORS[1], edgecolor='white')
    axes[1].set_title('Éducation père vs Note')
    axes[1].set_xlabel('Niveau d\'éducation')
    axes[1].set_ylabel('Note moyenne (G3)')
    save_fig('07_education_parents.png')

def run_all():
    print("Génération des graphiques EDA...")
    df = load_and_clean_data()
    graph_distribution_notes(df)
    graph_taux_reussite(df)
    graph_absences_vs_notes(df)
    graph_temps_etude(df)
    graph_correlation(df)
    graph_alcool_notes(df)
    graph_education_parents(df)
    print(f"\nTous les graphiques sont dans : {Config.GRAPHS_PATH}")

if __name__ == '__main__':
    run_all()