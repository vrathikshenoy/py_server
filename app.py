from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

# importing model
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# creating flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://localhost:3001"}})  # Enable CORS for specific routes

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    N = float(data['nitrogen'])
    P = float(data['phosphorus'])
    K = float(data['potassium'])
    temp = float(data['temperature'])
    humidity = float(data['humidity'])
    ph = float(data['ph'])
    rainfall = float(data['rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {0: 'rice',
 1: 'maize',
 2: 'chickpea',
 3: 'kidneybeans',
 4: 'pigeonpeas',
 5: 'mothbeans',
 6: 'mungbean',
 7: 'blackgram',
 8: 'lentil',
 9: 'pomegranate',
 10: 'banana',
 11: 'mango',
 12: 'grapes',
 13: 'watermelon',
 14: 'muskmelon',
 15: 'apple',
 16: 'orange',
 17: 'papaya',
 18: 'coconut',
 19: 'cotton',
 20: 'jute',
 21: 'coffee'}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated right there"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
