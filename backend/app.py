from flask import Flask, request
from flask_cors import CORS
import json
from ml import ml
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def ml_endpoint():
    if request.method == 'POST':
        data = ml(json.loads(request.data).get("query"))
        return json.dumps({
            "result": data[0],
            "positive_percentage": data[1],
            "negative_percentage": data[2],
        })
    else:
        return "<h1>There is nothing to see here. If you wanna use this api send a POST request.</h1>"
