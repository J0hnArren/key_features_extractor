import argparse
from collections import defaultdict
import os
from pathlib import Path
from text_processor import TextProcessor

PATH_TO_SAVE_FILES = "./outputs/"


def main():
    parser = argparse.ArgumentParser(description='Extract keywords from a text file.')
    parser.add_argument('file_path', type=str, help='Path to the text file')
    parser.add_argument('n_keywords', type=int, help='Number of keywords to extract')
    parser.add_argument('--method', type=str, choices=['tfidf', 'textrank', 'lda', 'word2vec', 'bert'],
                        default='tfidf', help='Method to use for keyword extraction')
    parser.add_argument('--csv', action='store_true', help='Save results to a CSV file')
    parser.add_argument('--sqlite', action='store_true', help='Save results to an SQLite database')
    args = parser.parse_args()

    processor = TextProcessor(args.file_path, args.n_keywords)
    keywords = defaultdict(str)

    if args.method == 'tfidf':
        keywords = processor.extract_keywords_tfidf()
    elif args.method == 'textrank':
        keywords = processor.extract_keywords_textrank()
    elif args.method == 'lda':
        keywords = processor.extract_keywords_lda()
    elif args.method == 'word2vec':
        keywords = processor.extract_keywords_word2vec()
    elif args.method == 'bert':
        keywords = processor.extract_keywords_bert()

    Path(PATH_TO_SAVE_FILES).mkdir(parents=True, exist_ok=True)
    path_to_save = os.path.join(PATH_TO_SAVE_FILES, 'keywords_' + args.method + '.csv')

    if args.csv:
        processor.save_to_csv(keywords=keywords, output_file=path_to_save)

    path_to_save = os.path.join(PATH_TO_SAVE_FILES, 'keywords_' + args.method + '.db')
    if args.sqlite:
        processor.save_to_sqlite(keywords, table_name='keywords_' + args.method, db_name=path_to_save)


if __name__ == "__main__":
    main()
