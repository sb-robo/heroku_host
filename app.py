import pickle
import json
import numpy as np
from scipy import stats
from flask import Flask, request, jsonify

app = Flask(__name__)
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
        
    temp = request.from.get("Temperature")
    ph = request.from.get("ph")
    humidity = request.from.get("Humidity")
    temp = request.from.get("Temperature")
    rain = request.from.get("Rainfall")
    nitrogenRatio = request.from.get("NitrogenRatio")
    phosphorousRatio = request.from.get("PhosphorousRatio")
    potasiumRatio = request.from.get("PotasiumRatio")

    data = [nitrogenRatio, phosphorousRatio, potasiumRatio, temp, humidity, ph, rain]

    data = scalar.transform(np.array(data).reshape(1,7))
    for _, model in models:
        pred = model.predict(data)
        predictions.append(pred[0])
    
    crop = stats.mode(predictions)[0][0]
    return jsonify({"Crop Name":crop})

if __name__ == '__main__':
    app.run()