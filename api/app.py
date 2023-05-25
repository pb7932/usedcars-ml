from flask import Flask, request
import utilities
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.get('/')
def home():
    return '''Pristupne točke: <br>
    <br>
    HTTP GET: /info<br>
    Ovdje možete dobiti više informacija o uporabi API-a.<br>
    <br>
    HTTP POST: /predict<br>
    Ovdje možete poslati podatke o automobilu, a natrag dobivate predviđenu cijenu.
    '''

@app.get('/info')
def info():
    return '''Upute za uporabu: <br>
    <br>
    Pošaljite HTTP POST zahtjev na: http://127.0.0.1:5000/predict <br>
    <br>
    Primjer tijela poruke: <br> 
    {<br>
        "cars": <br>
        [<br>
            {<br>
                "year": 2018,<br>
                "condition": "good",<br>
                "cylinders": 6,<br>
                "odometer": 50000,<br>
                "fuel": "gas",<br>
                "title_status": "clean",<br>
                "transmission": "manual",<br>
                "drive": "4wd",<br>
                "type": "pickup",<br>
                "posting_date": 1<br>
            }, <br>
            {"year": 2018,<br>
                "condition": "good",<br>
                "cylinders": 8,<br>
                "odometer": 100000,<br>
                "fuel": "gas",<br>
                "title_status": "clean",<br>
                "transmission": "manual",<br>
                "drive": "4wd",<br>
                "type": "pickup",<br>
                "posting_date": 1237<br>
            }<br>
        ]<br>
    }<br>
    '''

@app.get('/predict')
def predict_get():
    cars = [
        {
        "year": 2018,
        "condition": "good",
        "cylinders": 6,
        "odometer": 50000,
        "fuel": "gas",
        "title_status": "clean",
        "transmission": "manual",
        "drive": "4wd",
        "type": "pickup",
        "posting_date": 1
        }, 
        {"year": 2018,
        "condition": "good",
        "cylinders": 8,
        "odometer": 100000,
        "fuel": "gas",
        "title_status": "clean",
        "transmission": "manual",
        "drive": "4wd",
        "type": "pickup",
        "posting_date": 1237
        }
    ]

    df = pd.DataFrame(cars, index=range(len(cars)))

    predictions = utilities.predict(df)
    return '''Predviđene cijene: ''' + str(predictions)


@app.post('/predict')
def predict():
    data = request.json
    cars = data['cars']
    df = pd.DataFrame(cars, index=range(len(cars)))

    predictions = utilities.predict(df)
    return '''Predviđene cijene: ''' + str(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



