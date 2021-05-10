import pickle
import numpy as np
from scipy import stats
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
models, scalar = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['POST'])
def get_crop_recommendation():

    data = []
    predictions = []

    """json_data = request.from
    #print(json_file)
    #json_data = json.load(json_file)
    #print(json_data)
    for x in json_data:
        data.append( json_data[x])"""
        
    temp = request.form.get("Temperature")
    ph = request.form.get("Ph")
    humidity = request.form.get("Humidity")
    rain = request.form.get("Rainfall")
    nitrogenRatio = request.form.get("NitrogenRatio")
    phosphorousRatio = request.form.get("PhosphorousRatio")
    potasiumRatio = request.form.get("PotasiumRatio")

    data = [nitrogenRatio, phosphorousRatio, potasiumRatio, temp, humidity, ph, rain]

    data = scalar.transform(np.array(data).reshape(1,7))
    for _, model in models:
        pred = model.predict(data)
        predictions.append(pred[0])
    
    crop = stats.mode(predictions)[0][0]
    return jsonify({"Crop Name":crop})

if __name__ == '__main__':
    app.run()