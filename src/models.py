# src/models.py
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def get_models(random_state: int = 42):
    """
    Returns a dictionary of regression models for IPL score prediction.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state)
    }
    return models
