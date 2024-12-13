from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load the model
model = pickle.load(open('model.sav','rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    input_data = request.get_json()
    input_data = input_data.get('data')

    if not input_data:
        return jsonify({"error": "Features are required"}), 400
    
    df = pd.DataFrame([input_data], columns=model.feature_names_in_)

    # Perform prediction
    try:
        prediction = model.predict(df)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app locally for testing
if __name__ == '__main__':
    app.run()
