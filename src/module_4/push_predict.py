import joblib
import pandas as pd
import json


def handler_predict(event, _):
    user = pd.DataFrame.from_dict(json.loads(event["users"]), orient="index")
    model = joblib.load(event["model_path"])
    predictions = model.predict(user)

    json_data = {}
    for user, prediction in zip(user.index, predictions):
        json_data[str(user)] = int(prediction)

    return {"statusCode": "200", "body": json.dumps({"prediction": json_data})}
