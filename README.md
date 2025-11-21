# ML Django Application

A Django web application for machine learning predictions on marketing campaign data. Upload CSV files containing ad campaign metrics and get sales predictions using a trained Random Forest model.

## Features

- CSV file upload for batch predictions
- Automatic data preprocessing and feature engineering
- Machine learning model training and prediction
- Beautiful, animated UI with responsive design
- Real-time prediction results displayed in an interactive table

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation & Setup

### 1. Clone the Repository

```
git clone https://github.com/Mritunjay1121/django_project.git
cd django_project
```


### 2. Install UV : Python package and project manager

Use curl to download the script and execute it with sh on macOS and Linux:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Use irm to download the script and execute it with iex:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3.Initialize 'uv' and install requirements 

```
uv init
```

Then install requirements

```
uv add -r requirements.txt
```

### 4. Activate Virtual Environment


**On Windows:**

```
.venv\Scripts\activate
```


### 5. Run the Development Server

```
python manage.py runserver
```


The application will be available at: `http://127.0.0.1:8000/`

## Usage

1. Navigate to the home page
2. Click "Choose File" to upload your CSV file
3. Ensure your CSV contains the required columns:
   - Ad_ID
   - Campaign_Name
   - Ad_Date
   - Clicks
   - Impressions
   - Cost
   - Leads
   - Conversions
   - Conversion Rate
   - Location
   - Device
   - Keyword
4. Click "Predict/Train" and choose "Yes" for training and "No" for prediction to get sales predictions
5. View results in the displayed table

## Training a New Model

To train a new model with your data. The model will be saved in path :
```
savedmodels/randomforestmodel.joblib
```

