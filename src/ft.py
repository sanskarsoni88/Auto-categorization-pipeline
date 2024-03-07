import fasttext
from pathlib import Path
import numpy as np


class FastText:
    def __init__(self, df):
        self.df = df
        

    def get_word_embeddings(self):
        model_path = str(Path(__file__).resolve().parent.parent / 'cc.en.300.bin')
        model = fasttext.load_model(model_path)

        vec_len = len(model.get_word_vector('shoe'))
        fasttext_list = []

        for index, row in self.df.iterrows():
            name_vector = np.zeros(vec_len)

            for word in row['Product Name']:
                name_vector += model.get_word_vector(word)

            fasttext_list.append(name_vector)

        # Convert the list of vectors to a numpy array
        fasttext_np_array = np.array(fasttext_list)

        return fasttext_np_array
