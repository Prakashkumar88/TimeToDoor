# Auto-generated file.
from TimeToDoor.constants import *
from TimeToDoor.logger import logging
from TimeToDoor.exceptions import CustomException
import os, sys
from TimeToDoor.config.configuration import *
from TimeToDoor.components.data_transformation import DataTransformation, DataTransformationConfig
from TimeToDoor.components.model_trainer import ModelTrainer, ModelTrainerConfig   
from TimeToDoor.components.data_ingestion import DataIngestion, DataIngestionConfig
from TimeToDoor.pipelines.prediction_pipeline import CustomData, PredictionPipeline
from TimeToDoor.pipelines.training_pipeline import Train
from Prediction.batch import batch_prediction  
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request  

feature_engeneering_file_path = FEATURE_ENGG_OBJ_PATH
transformer_file_path = PREPROCESSING_OBJ_FILE
model_file_path = MODEL_FILE_PATH

UPLOAD_FOLDER = "batch_prediction/Uploaded_csv_FILE"
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            data = CustomData(
                Delivery_person_Age=int(request.form.get('Delivery_person_Age')),
                Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
                Weather_conditions=request.form.get('Weather_conditions'),
                Road_traffic_density=request.form.get('Road_traffic_density'),
                Vehicle_condition=int(request.form.get('Vehicle_condition')),
                Type_of_order=request.form.get('Type_of_order'),
                Type_of_vehicle=request.form.get('Type_of_vehicle'),
                multiple_deliveries=int(request.form.get('multiple_deliveries')),
                distance=float(request.form.get('distance')),  
                Festival=request.form.get('Festival'),        
                City=request.form.get('City')
            )

            final_new_data = data.get_data_as_dataframe()
            prediction_pipeline = PredictionPipeline()
            pred = prediction_pipeline.predict(final_new_data)

            result = int(pred[0])
            return render_template('form.html', final_result=result)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return render_template('form.html', final_result="Error occurred")

@app.route('/batch_prediction', methods=['GET', 'POST'])
def perform_batch_prediction():
    if request.method == 'GET':
        return render_template('batch_prediction.html')
    else:
        try:
            file = request.files['csv_file']
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                # Remove old files
                for filename in os.listdir(UPLOAD_FOLDER):
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                logging.info(f"CSV file received and uploaded at {file_path}")

                batch = batch_prediction(file_path, model_file_path, transformer_file_path, feature_engeneering_file_path)
                batch.start_batch_prediction()

                output = "Batch prediction done"
                return render_template('batch_prediction.html', output=output, file_path=file_path)
            else:
                return render_template('batch_prediction.html', output="Error: Invalid file type")
        except Exception as e:
            logging.error(f"Error during batch prediction: {e}")
            return render_template('batch_prediction.html', output="Error in batch prediction")

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')
    else:
        try:
            pipeline = Train()
            pipeline.main()
            return render_template('train.html', message="Training completed successfully")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            return render_template('index.html', message="Error during training")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8888)  
