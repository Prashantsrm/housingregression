# regression.py

from utils import load_data, evaluate_model
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

# Load data
X_train, X_test, y_train, y_test = load_data()

# Define baseline models
baseline_models = {
    "Lasso Regression (Baseline)": Lasso(random_state=42),
    "Ridge Regression (Baseline)": Ridge(random_state=42),
    "Random Forest (Baseline)": RandomForestRegressor(random_state=42)
}

# Hyperparameter tuning grid
param_grid = {
    "Lasso Regression (Tuned)": {
        'alpha': [0.01, 0.1, 1, 10],
        'max_iter': [1000, 5000, 10000],
        'tol': [1e-4, 1e-3]
    },
    "Ridge Regression (Tuned)": {
        'alpha': [0.01, 0.1, 1, 10],
        'max_iter': [1000, 5000, 10000],
        'tol': [1e-4, 1e-3]
    },
    "Random Forest (Tuned)": {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 4]
    }
}

# Evaluate baseline models
print("Model Performance Results\n" + "="*40)
results = []
for name, model in baseline_models.items():
    mse, r2, rmse = evaluate_model(model, X_train, X_test, y_train, y_test)
    results.append([name, mse, r2, rmse])

# Evaluate tuned models
print("\nModel Performance Results (Tuned)\n" + "="*40)
for name, params in param_grid.items():
    if "Lasso" in name:
        base_model = Lasso(random_state=42)
    elif "Ridge" in name:
        base_model = Ridge(random_state=42)
    elif "Random Forest" in name:
        base_model = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(base_model, params, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    tuned_model = grid.best_estimator_
    mse, r2, rmse = evaluate_model(tuned_model, X_train, X_test, y_train, y_test)
    results.append([name, mse, r2, rmse])

# Display comparison table
df = pd.DataFrame(results, columns=["Model", "MSE", "R²", "RMSE"])
print("\nModel Comparison:\n")
print(df.to_string(index=False))

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(df["Model"], df["MSE"], color="skyblue")
plt.xticks(rotation=45, ha='right')
plt.title("Model Comparison - MSE")
plt.ylabel("MSE")

plt.subplot(1, 2, 2)
plt.bar(df["Model"], df["R²"], color="lightgreen")
plt.xticks(rotation=45, ha='right')
plt.title("Model Comparison - R²")
plt.ylabel("R² Score")

plt.tight_layout()
plt.show()

