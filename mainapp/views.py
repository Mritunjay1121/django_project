from django.shortcuts import render
import pandas as pd
from joblib import load
from src.preprocessing import DataProcessor
from loguru import logger

def predictor(request):
    logger.info("INSIDE PREDICTOR")
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        train_model = request.POST.get('train_model', 'no')
        is_training = (train_model == "yes")
        
        logger.info(f"TRAIN MODEL VALUE: {is_training}")

        if not csv_file:
            return render(request, 'main.html', {'error': 'No file uploaded'})

        try:
            df = pd.read_csv(csv_file)
            logger.info(f"INPUT FILE COLUMNS: {df.columns.tolist()}")

            expected_cols = [
                'Ad_ID', 'Campaign_Name', 'Clicks', 'Impressions', 'Cost', 'Leads',
                'Conversions', 'Conversion Rate', 'Ad_Date', 'Location', 'Device', 'Keyword'
            ]

            # Check if all expected columns exist
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                return render(request, 'main.html', {
                    'error': f'CSV is missing columns: {", ".join(missing_cols)}'
                })

            # Reorder columns to expected sequence
            df_ordered = df[expected_cols].copy()

            if is_training:
                # Training mode
                if 'Sale_Amount' not in df.columns:
                    return render(request, 'main.html', {
                        'error': 'Training requires Sale_Amount column in CSV'
                    })
                
                df_ordered['Sale_Amount'] = df['Sale_Amount']
                
                logger.info("TRAINING STARTED")
                preprocessor = DataProcessor(model_path='./savedmodels/randomforestmodelnew.joblib')
                r2 = preprocessor.train(df_ordered)
                
                if r2 is None:
                    return render(request, 'main.html', {
                        'error': 'Training failed. Check logs for details.'
                    })
                
                logger.info("Model Trained Successfully")
                return render(request, 'main.html', {
                    'training_complete': True,
                    'r2_score': r2
                })
            
            else:
                # Prediction mode
                logger.info("PREDICTION STARTED")
                preprocessor = DataProcessor(model_path='./savedmodels/randomforestmodel.joblib')
                df_scaled = preprocessor.preprocess(df_ordered, is_training=False)
                
                if df_scaled is None:
                    return render(request, 'main.html', {
                        'error': 'Preprocessing failed. Check logs for details.'
                    })
                
                logger.info(f"Columns before predicting: {df_scaled.columns.tolist()}")
                
                predictions = preprocessor.predict(df_scaled)
                
                if predictions is None:
                    return render(request, 'main.html', {
                        'error': 'Prediction failed. Check logs for details.'
                    })
                
                logger.info("PREDICTIONS DONE")
                
                # Add predictions to original dataframe
                df['Predicted_Sale_Amount'] = predictions
                results_html = df.to_html(index=False, classes='table')
                
                return render(request, 'main.html', {'results_table': results_html})

        except Exception as e:
            logger.error(f"Error in predictor: {str(e)}")
            return render(request, 'main.html', {'error': f'Error processing file: {str(e)}'})

    return render(request, 'main.html')