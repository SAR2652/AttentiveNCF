import os
import argparse
import pandas as pd
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from goodreads_book_recommendation.utils import remove_book_outliers


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        help='Directory containing original data',
                        type=str, default='./data')
    parser.add_argument('--output_dir',
                        help='Directory to store output',
                        type=str, default='./output')
    return parser.parse_args()


def generate_and_save_label_encoder(df: pd.core.frame.DataFrame, column: str,
                                    output_dir: str, name: str):
    le = LabelEncoder()
    le.fit(df[column].values)
    dump(le,
         os.path.join(output_dir, name))
    print(len(le.classes_))


def prepare_interaction_matrix(args):
    data_dir = args.data_dir
    output_dir = args.output_dir

    books_path = os.path.join(data_dir, 'Books.csv')
    users_path = os.path.join(data_dir, 'Users.csv')

    books = pd.read_csv(books_path)
    users = pd.read_csv(users_path)

    remove_book_outliers(books)

    generate_and_save_label_encoder(books, 'Book-Title', output_dir,
                                    'book_title_encoder.joblib')
    generate_and_save_label_encoder(users, 'User-ID', output_dir,
                                    'user_id_encoder.joblib')


def main():
    args = get_arguments()
    prepare_interaction_matrix(args)


if __name__ == "__main__":
    main()
