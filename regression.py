# regression.py
from utils import load_data, evaluate_models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Load data
X_train, X_test, y_train, y_test = load_data()

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": SVR()
}

# Evaluate models
evaluate_models(models, X_train, X_test, y_train, y_test)

