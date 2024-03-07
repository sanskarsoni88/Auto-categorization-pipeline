import sys
import pandas as pd
from pathlib import Path
from text_processor import TextProcessor
from ft import FastText
from predict_results import Model
from datetime import datetime

class Pipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        # Load file
        df = pd.read_csv(self.filename)
        self.text_processor = TextProcessor(df)

        BnC_df = self.text_processor.preprocess()

        #TEST
        print(BnC_df['Product Name'].head())
        print()

        self.fasttext = FastText(BnC_df)
        word_embeddings = self.fasttext.get_word_embeddings()

        #TEST
        print(word_embeddings[:10])
        print()

        ## Hard coded to be BnC category products. In future, need to choose which model from which tier
        model_path = str(Path(__file__).resolve().parent.parent / 'models/BnC_feb_5')
        joblib_path = str(Path(__file__).resolve().parent.parent / 'joblib_models/BnC_encoder.joblib')

        BnC_Model = Model(model_path, joblib_path, word_embeddings)
        results_df = BnC_Model.predict()

        filename = datetime.now().strftime("%Y-%m-%d_%H-%M.csv")
        filename = f'Results_{filename}'
        file_save_path = str(Path(__file__).resolve().parent.parent / f'results/{filename}')

        ###Optional, just to make sure pipeline works
        results_df.to_csv(file_save_path, index = False)




    


if __name__ == "__main__":

    # Check if a command-line argument (CSV filename) was provided
    if len(sys.argv) == 2:
        csv_filename = sys.argv[1]
        print(f"Loading data from specified file: {csv_filename}")
        print()
    else:
        # No command-line argument provided, use default CSV file
        csv_filename = Path(__file__).resolve().parent.parent / 'product_database.csv'
        print(f"No file specified. Loading data from default file: {csv_filename}")
        print()

    BnCPipeline = Pipeline(csv_filename)
    BnCPipeline.predict()
    