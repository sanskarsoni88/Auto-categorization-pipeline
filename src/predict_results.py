from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import joblib
import numpy as np
from pathlib import Path
import pandas as pd

class Model:
    def __init__(self,model_path, joblib_path, array):
        self.array = array
        self.model_path = model_path
        self.joblib_path = joblib_path
        self.top_classes = 5

    def predict(self):
        model = load_model(self.model_path, custom_objects={'FBetaScore': tfa.metrics.FBetaScore})

        y_pred = model.predict(self.array)
        y_pred = np.argsort(y_pred)[:,-1*self.top_classes:]
        return(self.print_results(y_pred))

    def print_results(self, y_pred):
        encoder = joblib.load(self.joblib_path)
        predictions_1 = []
        predictions_2 = []
        predictions_3 = []
        predictions_4 = []
        predictions_5 = []

        for i in range(len(y_pred)):
            # Store predictions in respective lists
            predictions_1.append(encoder.inverse_transform([y_pred[i][-1]])[0])
            predictions_2.append(encoder.inverse_transform([y_pred[i][-2]])[0])
            predictions_3.append(encoder.inverse_transform([y_pred[i][-3]])[0])
            predictions_4.append(encoder.inverse_transform([y_pred[i][-4]])[0])
            predictions_5.append(encoder.inverse_transform([y_pred[i][-5]])[0])

            # Create DataFrame
            predictions_df = pd.DataFrame({
                '1st Prediction': predictions_1,
                '2nd Prediction': predictions_2,
                '3rd Prediction': predictions_3,
                '4th Prediction': predictions_4,
                '5th Prediction': predictions_5
            })
        print(predictions_df)
        return(predictions_df)

if __name__ == '__main__':
    model_path = str(Path(__file__).resolve().parent.parent / 'models/BnC_feb_5')
    joblib_path = str(Path(__file__).resolve().parent.parent / 'joblib_models/BnC_encoder.joblib')
    dummy_array = np.zeros((2, 300))

    model = Model(model_path, joblib_path, dummy_array)
    model.predict()
