from flask import *
from flask_cors import CORS
from ml_prediction import get_predctions_forall_locations,get_predtictions_specific_location

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

        results = get_predctions_forall_locations(BID,FID)
        data_set = {"results":results}
        json_dump = json.dumps(data_set)
        return json_dump

    except Exception as e:
        data_set = {"error":"Check the business ID or farm ID"}
        json_dump = json.dumps(data_set)
        return json_dump


@app.route("/get_specific_predictions/", methods=["GET"])
def get_and_save_predictions_specific():  
    try:
        BID = str(request.args.get("business_id"))
        FID = str(request.args.get("farm_id"))
        area_code = str(request.args.get("area_code"))
        location_code = str(request.args.get("location_code"))

        results = get_predtictions_specific_location(BID,FID,area_code,location_code)
        data_set = {"results":results}
        json_dump = json.dumps(data_set)
        return json_dump

    except Exception as e:
        data_set = {"error":"Check the business ID, farm ID, area code or location code"}
        json_dump = json.dumps(data_set)
        return json_dump
               

if __name__ == "__main__":
    app.run(debug=True)



