import joblib

def predict(input_data):
    model = joblib.load('education_model.pkl')
    prediction = model.predict(input_data)
    return prediction
