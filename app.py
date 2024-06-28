from flask import Flask, request, jsonify
from dotenv import load_env
import base64
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict as schema_predict
import os
import requests
from google.protobuf.json_format import MessageToDict

load_env()

app = Flask(__name__)

def download_image(url: str, local_filename: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    return local_filename

def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):      
    try:
        print("Current working directory:", os.getcwd())
        local_filename = "downloaded_image.jpg"
        file_path = download_image(filename, local_filename)
        print(f"Using file: {file_path}")

        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
        with open(file_path, "rb") as f:
            file_content = f.read()

        encoded_content = base64.b64encode(file_content).decode("utf-8")
        instance = schema_predict.instance.ImageClassificationPredictionInstance(
            content=encoded_content,
        ).to_value()
        instances = [instance]
        parameters = schema_predict.params.ImageClassificationPredictionParams(
            confidence_threshold=0.5,
            max_predictions=5,
        ).to_value()
        endpoint = client.endpoint_path(
            project=project, location=location, endpoint=endpoint_id
        )
        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )
        print("response")
        print(" deployed_model_id:", response.deployed_model_id)
        
        # Convert the entire response to a JSON-serializable dictionary
        response_dict = MessageToDict(response._pb)

        return response_dict
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/predict', methods=['POST'])
def handle_predict():
    try:
        data = request.get_json()
        filename = data.get('filename')
        print(filename)
        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        results = predict_image_classification_sample(
            project=os.getenv("PROJECTID"),
            endpoint_id=os.getenv("ENDPOINT"),
            location="us-central1",
            filename=filename
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=7000, debug=True)
