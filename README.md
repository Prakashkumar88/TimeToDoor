# TimeToDoor 🚚

**TimeToDoor** is a machine learning-powered delivery time prediction application. It predicts the expected delivery time based on various features like delivery person details, weather, traffic, order type, vehicle, and city information. The app also supports batch predictions via CSV files.

---

## Features

- Predict delivery time for a single order.
- Perform batch predictions using CSV uploads.
- Train and retrain the machine learning model with the latest dataset.
- Modern Flask-based web interface.

---

## Requirements

- Python 3.11+
- Conda (recommended)
- All dependencies are listed in `requirements.txt`

---

## Installation & Setup

1. **Clone the repository**

```bash
git clone [<your-repo-url>](https://github.com/Prakashkumar88/TimeToDoor.git)
cd TimeToDoor
```

2. **Activate your Conda environment**
```bash
conda create -n timetodoor python=3.11
conda activate timetodoor
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Flask application**

```bash
python app.py
```

5. **Access the app**

* Open your browser and navigate to: [http://localhost:5000](http://localhost:5000)

---

## Project Structure

```
TimeToDoor/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
├── static/                # CSS, JS, favicon
├── TimeToDoor/            # Main Python package
│   ├── components/        # Data ingestion, transformation, training modules
│   ├── pipelines/         # Prediction & training pipelines
│   ├── config/            # Configuration files
│   ├── utils.py           # Helper functions
│   ├── logger.py          # Logging setup
│   └── constants.py       # Constants like file paths
└── requirements.txt       # Python dependencies
```

---

## Usage

### Single Prediction

1. Go to `/predict`
2. Fill out the form with order details
3. Click **Predict** to get the estimated delivery time

### Batch Prediction

1. Go to `/batch_prediction`
2. Upload a CSV file with multiple orders
3. Click **Run Batch Prediction**
4. Download the output CSV with predicted delivery times

### Train Model

1. Go to `/train`
2. Click **Train Model** to retrain the model with the latest dataset

---

## Updating Libraries and Retraining Model

1. **Update libraries to the latest versions**

```bash
pip install --upgrade -r requirements.txt
```

2. **Retrain the model**

* Go to the `/train` page on the app and click **Train Model**
* Or run the training pipeline manually:

```bash
python -m TimeToDoor.pipelines.training_pipeline
```

3. **Predict using the latest model**

* The updated model will automatically be used for `/predict` and `/batch_prediction`.

---

## Notes

* Ensure to retrain the model if you update the dataset or dependencies.
* For production deployment, configure the Flask app with a proper web server (e.g., Gunicorn or Nginx).
* Output CSV files from batch prediction are saved in `batch_prediction/prediction_csv/output.csv`.

---

## License

MIT License © 2025 Prakash Kumar
