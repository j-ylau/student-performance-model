from data_preprocessing import preprocess_data
from model_training import train_model
from predict import predict

# Preprocessing
X = preprocess_data('./data/education_data.csv')

# Model Training
train_model(X)

# Prediction
input_data = X[0, :-1].reshape(1, -1)
result = predict(input_data)
print(f"Predicted performance: {result}")
