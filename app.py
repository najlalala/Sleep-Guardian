import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

def preprocessing(data, ohe):  
    data["Gender"] = data["Gender"].replace({
    "Male": 0,
    "Female": 1
    })
    
    data_ohe = pd.DataFrame(ohe.transform(data[["Occupation", "BMI Category"]]))
    data.drop(["Occupation", "BMI Category"], axis=1, inplace=True)
    data = pd.concat([data, data_ohe], axis=1)
    
    data["systolic"] = data["Blood Pressure"].apply(lambda x: int(x.split("/")[0]))
    data["diastolic"] = data["Blood Pressure"].apply(lambda x: int(x.split("/")[1]))

    data.drop(["Blood Pressure"], axis=1, inplace=True)
    
    return data

def index_to_label(index):
    label = {
        0 : "Normal",
        1 : "Insomnia",
        2 : "Sleep Apnea"
    }
    return label[index]

def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    return model

def load_ohe():
    ohe = pickle.load(open('ohe.pkl', 'rb'))
    return ohe

@app.route('/')
def index():
    return 'Hello World'

@app.route('/predict', methods=['GET'])
def predict():
    ohe = load_ohe()
    model = load_model()
    
    gender = request.args.get("gender")
    age = request.args.get("age")
    occupation = request.args.get("occupation")
    sleep_duration = request.args.get("sleep_duration")
    quality_of_sleep = request.args.get("quality_of_sleep")
    physical_activity = request.args.get("physical_activity")
    stress_level = request.args.get("stress_level")
    bmi_category = request.args.get("bmi_category")
    blood_pressure = request.args.get("blood_pressure")
    heart_rate = request.args.get("heart_rate")
    daily_steps = request.args.get("daily_steps")

    data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Occupation": [occupation],
        "Sleep Duration": [sleep_duration],
        "Quality of Sleep": [quality_of_sleep],
        "Physical Activity Level": [physical_activity],
        "Stress Level": [stress_level],
        "BMI Category": [bmi_category],
        "Blood Pressure": [blood_pressure],
        "Heart Rate": [heart_rate],
        "Daily Steps": [daily_steps]
    })
    
    processed_data = preprocessing(data, ohe)
    processed_data.columns = processed_data.columns.astype(str)
    prediction = model.predict(processed_data)
    hasil = index_to_label(prediction[0])
    
    return hasil 

@app.route('/predict1', methods=['POST'])
def predict():
    ohe = load_ohe()
    model = load_model()
    
    # Extracting data from the JSON request
    gender = request.form.get("gender")
    age = request.form.get("age")
    occupation = request.form.get("occupation")
    sleep_duration = request.form.get("sleep_duration")
    quality_of_sleep = request.form.get("quality_of_sleep")
    physical_activity = request.form.get("physical_activity")
    stress_level = request.form.get("stress_level")
    bmi_category = request.form.get("bmi_category")
    blood_pressure = request.form.get("blood_pressure")
    heart_rate = request.form.get("heart_rate")
    daily_steps = request.form.get("daily_steps")

    data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Occupation": [occupation],
        "Sleep Duration": [sleep_duration],
        "Quality of Sleep": [quality_of_sleep],
        "Physical Activity Level": [physical_activity],
        "Stress Level": [stress_level],
        "BMI Category": [bmi_category],
        "Blood Pressure": [blood_pressure],
        "Heart Rate": [heart_rate],
        "Daily Steps": [daily_steps]
    })

    processed_data = preprocessing(data, ohe)
    processed_data.columns = processed_data.columns.astype(str)
    prediction = model.predict(processed_data)
    result = index_to_label(prediction[0])
    
    return str(result)

if __name__ == '__main__':
    app.run(debug=True)