# house_price_prediction.py

from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('HousePricetrain.csv')

# Handle missing values by filling with mean
data.fillna(data.mean(), inplace=True)

# Handle categorical columns using Label Encoding
label_encoder = LabelEncoder()
data['size_units'] = label_encoder.fit_transform(data['size_units'])
data['lot_size_units'] = label_encoder.fit_transform(data['lot_size_units'])

# Select features and target variable
X = data[['beds', 'baths', 'size', 'lot_size']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler using pickle
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from the JSON data
    features = [data['beds'], data['baths'], data['size'], data['lot_size']]

    # Load the scaler used during training
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Scale the input features
    scaled_features = scaler.transform([features])

    # Load the trained model
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Make a prediction using the trained model
    predicted_price = model.predict(scaled_features)[0]

    # Format the predicted price to two decimal places
    formatted_predicted_price = '{:.2f}'.format(predicted_price)

    response = {'predicted_price': formatted_predicted_price}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
