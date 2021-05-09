import pickle
import json
import numpy as np
from scipy import stats
from flask import Flask, request, jsonify

app = Flask(__name__)
models, scalar = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['POST'])
def get_crop_recommendation():
    json_data = request.json
    #print(json_file)
    #json_data = json.load(json_file)
    #print(json_data)
    data = []
    predictions = []
    for x in json_data:
        data.append( json_data[x])

    data = scalar.transform(np.array(data).reshape(1,7))
    for _, model in models:
        pred = model.predict(data)
        predictions.append(pred[0])
    
    crop = stats.mode(predictions)[0][0]
    return jsonify({"Crop Name":crop})

if __name__ == '__main__':
    app.run()