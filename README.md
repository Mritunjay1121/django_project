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
git clone repo
cd project_name
```


### 2. Create Virtual Environment

```
python -m venv venv
```

### 3. Activate Virtual Environment

**On macOS/Linux:**

```
source venv/bin/activate
```


**On Windows:**

```
venv\Scripts\activate
```


### 4. Install Dependencies

```
pip install -r requirements.txt
```


### 5. Run Database Migrations (First Time Only)

```
python manage.py migrate
```


### 6. Run the Development Server

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

