import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import sys

class TextProcessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self):

        BnC_df = self.df[self.df['Category']=='Building & Construction'][['Product Name']].reset_index().drop(columns='index')
        BnC_df['Product Name'] = BnC_df['Product Name'].astype(str)

        # Simplify product name
        BnC_df['Product Name'] = BnC_df['Product Name'].apply(self.simplify_text)

        BnC_df = self.postprocess(BnC_df)

        return BnC_df

    def simplify_text(self, text):
        # Lowercase the text
        text = text.lower()

        # Remove HTML tags, newline characters, etc.
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'([a-zA-Z])([^a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([^a-zA-Z])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
    
    def postprocess(self, df):

        # word tokenize
        try:
            nltk.data.find('tokenizers/punkt')
            df['Product Name'] = df['Product Name'].apply(word_tokenize)
        except LookupError:
            print("The NLTK 'punkt' dataset could not be found.")
            print()
            sys.exit(1)  # Exit the script due to the error


        try:
            nltk.data.find('corpora/stopwords')
            stop_words = set(stopwords.words('english'))
            df['Product Name'] = df['Product Name'].apply(lambda x: [word for word in x if word not in stop_words])
        except LookupError:
            print("The NLTK 'stopwords' dataset could not be found.")
            print()
            sys.exit(1)  # Exit the script due to the error


        # remove non alphabetic tokens
        df['Product Name'] = df['Product Name'].apply(lambda tokens: [token for token in tokens if token.isalpha()])

        return df