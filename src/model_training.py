from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def train_model(X):
    y = X[:, -1]  # Assuming the last column is the target variable
    X = X[:, :-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse}")
    
    # Save the trained model
    joblib.dump(model, 'education_model.pkl')
