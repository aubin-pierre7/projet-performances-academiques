def test_imports():
    try:
        import flask; print(f"Flask {flask.__version__} — OK")
        import pandas; print(f"Pandas {pandas.__version__} — OK")
        import sklearn; print(f"Scikit-learn {sklearn.__version__} — OK")
        import matplotlib; print(f"Matplotlib {matplotlib.__version__} — OK")
        import seaborn; print(f"Seaborn {seaborn.__version__} — OK")
        import numpy; print(f"NumPy {numpy.__version__} — OK")
        import joblib; print(f"Joblib {joblib.__version__} — OK")
        print("\nToutes les librairies sont installées correctement.")
    except ImportError as e:
        print(f"ERREUR : {e}")

test_imports()