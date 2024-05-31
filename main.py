import argparse
from text_processor import TextProcessor


def main():
    parser = argparse.ArgumentParser(description='Extract keywords from a text file.')
    parser.add_argument('file_path', type=str, help='Path to the text file')
    parser.add_argument('n_keywords', type=int, help='Number of keywords to extract')
    parser.add_argument('--csv', action='store_true', help='Save results to a CSV file')
    parser.add_argument('--sqlite', action='store_true', help='Save results to an SQLite database')
    args = parser.parse_args()

    processor = TextProcessor(args.file_path, args.n_keywords)
    keywords = processor.extract_keywords()

    for keyword, score in keywords:
        print(f'{keyword}: {score}')

    if args.csv:
        processor.save_to_csv(keywords=keywords, output_file='keywords.csv')

    if args.sqlite:
        processor.save_to_sqlite(keywords)


if __name__ == "__main__":
    main()
