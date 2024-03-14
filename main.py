from flask import *
from flask_cors import CORS
from ml_prediction import load_images_get_predctions

app = Flask(__name__)
CORS(app) # Enable CORS for all routes


@app.route("/", methods=["GET"])
def home():
    try:
        data_set = {"status":"status ok"}
        json_dump = json.dumps(data_set)
        return json_dump
    
    except Exception as e:
        data_set = {"error":str(e)}
        json_dump = json.dumps(data_set)
        return json_dump

@app.route("/get_predictions/", methods=["GET"])
def get_and_save_predictions():  

    try:
        BID = str(request.args.get("business_id"))
        FID = str(request.args.get("farm_id"))

        results = load_images_get_predctions(BID,FID)
        data_set = {"results":results}
        json_dump = json.dumps(data_set)
        return json_dump

    except Exception as e:
        data_set = {"error":str(e)}
        json_dump = json.dumps(data_set)
        return json_dump
        

if __name__ == "__main__":
    app.run(debug=True)



