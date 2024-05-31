import argparse
import csv
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import gensim
from gensim.models import LdaModel, Word2Vec
from gensim.corpora.dictionary import Dictionary
from transformers import BertTokenizer, BertModel
import torch


class TextProcessor:
    def __init__(self, file_path, n_keywords):
        self.file_path = file_path
        self.n_keywords = n_keywords
        self.text = self._read_file()

    def _read_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def extract_keywords(self):
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words=['ну'], ngram_range=(2, 5))
        tfidf_matrix = vectorizer.fit_transform([self.text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        keywords = list(zip(feature_names, tfidf_scores))
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        return keywords[:self.n_keywords]

    @staticmethod
    def save_to_csv(keywords, output_file='keywords.csv'):
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Keyword', 'Score'])
            writer.writerows(keywords)

    @staticmethod
    def save_to_sqlite(keywords, db_name='keywords.db'):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS keywords (keyword TEXT, score REAL)''')
        cursor.executemany('INSERT INTO keywords (keyword, score) VALUES (?, ?)', keywords)
        conn.commit()
        conn.close()
