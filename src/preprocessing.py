# import pandas as pd
# import joblib

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# from loguru import logger

# class DataProcessor:
#     def __init__(self,model_path):
#         self.encoders = {}
#         self.model=None
#         self.model_path=model_path
#         # if model_path:
#         #     self.model = joblib.load(model_path)

#     # def separate_labels_and_test_data(self, df):
#     #     try:
#     #         if "Sale_Amount" in df.columns:
#     #             label_df = df["Sale_Amount"]
#     #             df = df.drop("Sale_Amount", axis=1)
#     #             return df
#     #         return df
#     #     except Exception as e:
#     #         print(f"Error in separate_labels_and_test_data: {e}")
#     #         return df, None
            
#     def separate_labels_and_test_data(self, df):
#         try:
#             if "Sale_Amount" in df.columns:
#                 if self.is_training==True:
#                     df = df.dropna(subset=['Sale_Amount'])
#                 label_df = df["Sale_Amount"]
#                 df = df.drop("Sale_Amount", axis=1)
#                 return df, label_df
#             return df, None
#         except Exception as e:
#             print(f"Error in separate_labels_and_test_data: {e}")
#             return df, None

#     def clean_column_and_categories(self, df):
#         try:
#             df.drop("Ad_ID",axis=1,inplace=True)
#             df['Campaign_Name'] = 'Data Analytics Course'
#             df['Location'] = 'hyderabad'
#             return df
#         except Exception as e:
#             print(f"Error in clean_column_categories: {e}")
#             return df

#     def fix_datatypes_andcategories(self, df):
#         try:
#             df['Cost'] = df['Cost'].replace('[\$,]', '', regex=True).astype(float)
#             return df
#         except Exception as e:
#             print(f"Error in fix_datatypes_andcategories: {e}")
#             return df

#     def fix_date(self, df):
#         try:
#             df['Ad_Date'] = pd.to_datetime(df['Ad_Date'], errors='coerce')
#             df['Ad_Day'] = df['Ad_Date'].dt.day
#             df['Ad_Weekday'] = df['Ad_Date'].dt.weekday
#             df['Ad_Week'] = df['Ad_Date'].dt.isocalendar().week
#             df.drop('Ad_Date', axis=1, inplace=True)
#             return df
#         except Exception as e:
#             print(f"Error in fix_date: {e}")
#             return df

#     def make_category_lower(self, df):
#         try:
#             df["Keyword"] = df['Keyword'].str.lower()
#             return df
#         except Exception as e:
#             print(f"Error in make_category_lower: {e}")
#             return df

#     def impute_missing(self, df):
#         try:
#             df = df.copy()
#             num_cols = df.select_dtypes(include=['number']).columns
#             for col in num_cols:
#                 mode_val = df[col].mode()[0]
#                 df[col].fillna(mode_val, inplace=True)
#             cat_cols = df.select_dtypes(include=['object', 'category']).columns
#             for col in cat_cols:
#                 freq_val = df[col].mode()[0]
#                 df[col].fillna(freq_val, inplace=True)
#             return df
#         except Exception as e:
#             print(f"Error in impute_missing: {e}")
#             return df

#     def encode_categorical(self, df):
#         try:
#             df = df.copy()
#             cat_cols = df.select_dtypes(include=['object', 'category']).columns
#             for col in cat_cols:
#                 unique_vals = df[col].dropna().unique()
#                 val_to_int = {v: i for i, v in enumerate(unique_vals)}
#                 df[col] = df[col].map(val_to_int)
#                 self.encoders[col] = val_to_int
#             return df
#         except Exception as e:
#             print(f"Error in encode_categorical: {e}")
#             return df

#     def scale_features(self, df):
#         try:
#             df_scaled = df.copy()
#             numeric_cols = df.select_dtypes(include=['number']).columns
#             for col in numeric_cols:
#                 mean = df[col].mean()
#                 std = df[col].std()
#                 if std != 0:
#                     df_scaled[col] = (df[col] - mean) / std
#                 else:
#                     df_scaled[col] = 0
#             return df_scaled
#         except Exception as e:
#             print(f"Error in scale_features: {e}")
#             return df
#     # def train_random_forest(X, y):
#     #     try:
#     #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     #         model = RandomForestRegressor(random_state=42)
#     #         model.fit(X_train, y_train)
#     #         y_pred = model.predict(X_test)
#     #         r2 = r2_score(y_test, y_pred)
#     #         return model, r2
#     #     except Exception as e:
#     #         print(f"Error in scale_features: {e}")
            
    
#     # def preprocess(self, df):
#     #     try:
#     #         df= self.separate_labels_and_test_data(df)
#     #         df = self.clean_column_and_categories(df)
#     #         df = self.fix_datatypes_andcategories(df)
#     #         df = self.fix_date(df)
#     #         df = self.make_category_lower(df)
#     #         df = self.impute_missing(df)
#     #         df = self.encode_categorical(df)
#     #         df_scaled = self.scale_features(df)
#     #         return df_scaled
#     #     except Exception as e:
#     #         print(f"Error in preprocess: {e}")
#     #         return None

#     def preprocess(self, df, is_training=False):
#         try:
#             if is_training:
#                 self.is_training=is_training
#                 df, label_df = self.separate_labels_and_test_data(df)
#             else:
#                 self.is_training=is_training
#                 df = self.separate_labels_and_test_data(df)[0]
#                 label_df = None
            
#             df = self.clean_column_and_categories(df)
#             df = self.fix_datatypes_andcategories(df)
#             df = self.fix_date(df)
#             df = self.make_category_lower(df)
#             df = self.impute_missing(df)
#             df = self.encode_categorical(df)
#             df_scaled = self.scale_features(df)
            
#             if is_training:
#                 return df_scaled, label_df
#             return df_scaled
#         except Exception as e:
#             print(f"Error in preprocess: {e}")
#             return None
#     def train(self, df):
#         try:
#             # Preprocess training data
#             logger.info("INSIDE TRAIN METHOD")
#             X, y = self.preprocess(df, is_training=True)
            
#             if X is None or y is None:
#                 raise ValueError("Preprocessing failed")
            
#             # Split data
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42
#             )
            
#             # Train model
#             self.model = RandomForestRegressor(
#                 n_estimators=100,
#                 random_state=42,
#                 n_jobs=-1
#             )
#             self.model.fit(X_train, y_train)
            
#             # Evaluate
#             y_pred = self.model.predict(X_test)
#             # mse = mean_squared_error(y_test, y_pred)
#             r2 = r2_score(y_test, y_pred)
            
#             print(f"Training completed!")
#             # print(f"MSE: {mse:.2f}")
#             print(f"R2 Score: {r2:.4f}")
            
#             # Save model
#             joblib.dump(self.model, self.model_path)
#             print(f"Model saved to {self.model_path}")
            
#             return r2
            
#         except Exception as e:
#             print(f"Error in train: {e}")
#             return None, None, None



#     def predict(self, df):
#         self.model = joblib.load(self.model_path)
#         if self.model is None:
#             raise ValueError("Model not loaded. Provide model_path during initialization.")
#         return self.model.predict(df)



import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from loguru import logger

class DataProcessor:
    def __init__(self, model_path):
        self.encoders = {}
        self.model = None
        self.model_path = model_path
        self.is_training = False

    def separate_labels_and_test_data(self, df):
        try:
            if "Sale_Amount" in df.columns:
                if self.is_training:
                    df = df.dropna(subset=['Sale_Amount'])
                    

                label_df = df["Sale_Amount"]
                label_df = label_df.replace('[\$,]', '', regex=True).astype(float)
                df = df.drop("Sale_Amount", axis=1)
                return df, label_df
            return df, None
        except Exception as e:
            logger.error(f"Error in separate_labels_and_test_data: {e}")
            return df, None

    def clean_column_and_categories(self, df):
        try:
            df = df.copy()
            df.drop("Ad_ID", axis=1, inplace=True)
            df['Campaign_Name'] = 'Data Analytics Course'
            df['Location'] = 'hyderabad'
            return df
        except Exception as e:
            logger.error(f"Error in clean_column_categories: {e}")
            return df

    def fix_datatypes_andcategories(self, df):
        try:
            df = df.copy()
            df['Cost'] = df['Cost'].replace('[\$,]', '', regex=True).astype(float)
            return df
        except Exception as e:
            logger.error(f"Error in fix_datatypes_andcategories: {e}")
            return df

    def fix_date(self, df):
        try:
            df = df.copy()
            df['Ad_Date'] = pd.to_datetime(df['Ad_Date'], errors='coerce')
            df['Ad_Day'] = df['Ad_Date'].dt.day
            df['Ad_Weekday'] = df['Ad_Date'].dt.weekday
            df['Ad_Week'] = df['Ad_Date'].dt.isocalendar().week
            df.drop('Ad_Date', axis=1, inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error in fix_date: {e}")
            return df

    def make_category_lower(self, df):
        try:
            df = df.copy()
            df["Keyword"] = df['Keyword'].str.lower()
            return df
        except Exception as e:
            logger.error(f"Error in make_category_lower: {e}")
            return df

    def impute_missing(self, df):
        try:
            df = df.copy()
            num_cols = df.select_dtypes(include=['number']).columns
            for col in num_cols:
                if len(df[col].mode()) > 0:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if len(df[col].mode()) > 0:
                    freq_val = df[col].mode()[0]
                    df[col].fillna(freq_val, inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error in impute_missing: {e}")
            return df

    def encode_categorical(self, df):
        try:
            df = df.copy()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                unique_vals = df[col].dropna().unique()
                val_to_int = {v: i for i, v in enumerate(unique_vals)}
                df[col] = df[col].map(val_to_int)
                self.encoders[col] = val_to_int
            return df
        except Exception as e:
            logger.error(f"Error in encode_categorical: {e}")
            return df

    def scale_features(self, df):
        try:
            df_scaled = df.copy()
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std != 0:
                    df_scaled[col] = (df[col] - mean) / std
                else:
                    df_scaled[col] = 0
            return df_scaled
        except Exception as e:
            logger.error(f"Error in scale_features: {e}")
            return df

    def preprocess(self, df, is_training=False):
        try:
            self.is_training = is_training
            df, label_df = self.separate_labels_and_test_data(df)
            logger.info("separate_labels_and_test_data done")
            
            df = self.clean_column_and_categories(df)
            logger.info("clean_column_and_categories done")
            df = self.fix_datatypes_andcategories(df)
            logger.info("fix_datatypes_andcategories done")
            df = self.fix_date(df)
            logger.info("fix_date done")
            df = self.make_category_lower(df)
            logger.info("make_category_lower done")
            df = self.impute_missing(df)
            logger.info("impute_missing done")
            df = self.encode_categorical(df)
            logger.info("encode_categorical done")
            df_scaled = self.scale_features(df)
            logger.info("scale_features done")
            
            if is_training:
                return df_scaled, label_df
            return df_scaled
        except Exception as e:
            logger.error(f"Error in preprocess: {e}")
            return None

    def train(self, df):
        try:
            logger.info("INSIDE TRAIN METHOD")
            X, y = self.preprocess(df, is_training=True)
            
            if X is None or y is None:
                raise ValueError("Preprocessing failed")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Training completed!")
            logger.info(f"R2 Score: {r2:.4f}")
            
            # Save model
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            return r2
            
        except Exception as e:
            logger.error(f"Error in train: {e}")
            return None

    def predict(self, df):
        try:
            if self.model is None:
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            return self.model.predict(df)
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            return None