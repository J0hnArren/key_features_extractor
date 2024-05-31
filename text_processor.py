import csv
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import gensim
from gensim.models import LdaModel, Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phrases, Phraser
from transformers import BertTokenizer, BertModel
import torch
import nltk
from nltk import ngrams, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


class TextProcessor:
    def __init__(self, file_path, n_keywords, n_gram_range=(1, 5)):
        self.file_path = file_path
        self.n_keywords = n_keywords
        self.text = self._read_file()
        self.stop_words = set(stopwords.words('russian'))
        self.n_gram_range = (n_gram_range[0], n_gram_range[1]+1)

    def _read_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _tokenize_text(self):
        tokens = [word for word in word_tokenize(self.text.lower()) if word.isalnum() and word not in self.stop_words]
        n_grams = []
        for i_gram in range(self.n_gram_range):
            n_grams += [' '.join(gram) for gram in ngrams(tokens, i_gram)]
        return n_grams

    def extract_keywords_tfidf(self):
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words=self.stop_words, ngram_range=(2, self.n_gram_range[1]))
        tfidf_matrix = vectorizer.fit_transform([self.text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        keywords = list(zip(feature_names, tfidf_scores))
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        return keywords[:self.n_keywords]

    def extract_keywords_textrank(self):
        parser = PlaintextParser.from_string(self.text, Tokenizer("russian"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, self.n_keywords)
        keywords = [(str(sentence), 1.0) for sentence in summary]
        return keywords

    def extract_keywords_lda(self):
        tokens = [word for word in word_tokenize(self.text.lower()) if word.isalnum() and word not in self.stop_words]
        bigrams = Phrases([tokens], min_count=1, threshold=2)
        bigram_mod = Phraser(bigrams)
        tokens = bigram_mod[tokens]
        dictionary = Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        lda = LdaModel(corpus, num_topics=1, id2word=dictionary, passes=15)
        topics = lda.show_topic(0, self.n_keywords)
        keywords = [(dictionary[int(word_id)], score) if word_id.isdigit() else (word_id, score) for word_id, score in topics]
        return keywords

    def extract_keywords_word2vec(self):
        tokens = [word for word in word_tokenize(self.text.lower()) if word.isalnum() and word not in self.stop_words]
        bigrams = Phrases([tokens], min_count=1, threshold=2)
        bigram_mod = Phraser(bigrams)
        tokens = bigram_mod[tokens]
        model = Word2Vec([tokens], vector_size=100, window=5, min_count=1, workers=4)
        keywords = model.wv.index_to_key[:self.n_keywords]
        keywords = [(keyword, 1.0) for keyword in keywords if keyword not in self.stop_words]
        return keywords

    def extract_keywords_bert(self):
        tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')
        model = BertModel.from_pretrained('cointegrated/rubert-tiny2')
        tokens = self._tokenize_text()
        inputs = tokenizer(tokens, return_tensors='pt', max_length=2048, truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        scores = [(token, torch.norm(embedding).item()) for token, embedding in zip(tokens, embeddings)]
        keywords = sorted(scores, key=lambda x: x[1], reverse=True)
        return keywords[:self.n_keywords]

    @staticmethod
    def save_to_csv(keywords, output_file='keywords.csv'):
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Keyword', 'Score'])
            writer.writerows(keywords)

        print("CSV file created successfully!")

    @staticmethod
    def save_to_sqlite(keywords, table_name="keywords", db_name='keywords.db'):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
        cursor.execute(f'''CREATE TABLE {table_name} (keyword TEXT, score REAL)''')
        cursor.executemany(f'INSERT INTO {table_name} (keyword, score) VALUES (?, ?)', keywords)
        conn.commit()
        conn.close()

        print(f"SQL database table {table_name} created/overwritten successfully!")
