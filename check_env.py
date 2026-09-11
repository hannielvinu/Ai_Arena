import sys
for pkg in ['xgboost', 'lightgbm', 'catboost', 'torch', 'sklearn', 'scipy', 'matplotlib', 'seaborn']:
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {getattr(mod, '__version__', 'available')}")
    except Exception as e:
        print(f"{pkg}: NOT available ({e})")
