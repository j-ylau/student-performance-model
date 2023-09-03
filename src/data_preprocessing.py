import pandas as pd
from sklearn.preprocessing import StandardScaler

def convert_percent_to_float(percent_str):
    """Converts a string formatted percentage to float."""
    if isinstance(percent_str, str) and '%' in percent_str:
        return float(percent_str.replace('%', '')) / 100
    return percent_str

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Convert percentage strings to floats
    for col in df.columns:
        df[col] = df[col].apply(convert_percent_to_float)
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Feature selection (tailor this to your needs)
    selected_features = [
        'eattend10', 'mattend10', 'hsattend10',
        'eenrol11', 'menrol11', 'hsenrol11',
        'aastud10', 'wstud10', 'hstud10',
        'abse10', 'absmd10', 'abshs10',
        'farms10', 'sped10', 'ready11',
        'math310', 'read310', 'hsaeng10',
        'drop10', 'compl10'
    ]
    
    X = df[selected_features]
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled
