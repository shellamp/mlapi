from flask import Flask, request
import joblib
import pandas as pd

app = Flask(__name__)
pipe = joblib.load('iris.mdl')

@app.route('/hello', methods=['POST'])
def hello():
    data = request.get_json()
    name = data.get('name')
    message = f'Hello {name}'
    return {'message': message}, 200

@app.route('/predict', methods=['POST'])
def predict():
    # Use the column names from your original training data
    column_names = ["Id", "Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]  
    
    data = request.get_json()

    # Create a list of values using the order of columns in the training data
    data_vector = [
        data.get("Id"),
        data.get("Sepal Length (cm)"),
        data.get("Sepal Width (cm)"),
        data.get("Petal Length (cm)"),
        data.get("Petal Width (cm)")
    ]
    
    X_new = pd.DataFrame([data_vector], columns=column_names)
    y_pred = pipe.predict(X_new)[0]
    return {'prediction': y_pred}, 200



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)