import os
import argparse
import numpy as np
import pandas as pd
from joblib import load
from itertools import product
from more_itertools import ichunked
# from scipy.sparse import coo_matrix, save_npz
from common_utils import remove_book_outliers


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        help='Directory containing original data',
                        type=str, default='./data')
    parser.add_argument('--user_id_encoder_file',
                        help='Encoder file used to encode/decode User IDs',
                        type=str, default='./output/user_id_encoder.joblib')
    parser.add_argument('--book_title_encoder_file',
                        help='Encoder file used to encode/decode Book Titles',
                        type=str, default='./output/book_title_encoder.joblib')
    parser.add_argument('--output_dir',
                        help='Directory to store output',
                        type=str, default='./partition_data')
    parser.add_argument('--train_size',
                        help='Training set size as a fraction',
                        type=str, default=0.7)
    parser.add_argument('--random_state',
                        help='Random Seed for Reproducibility',
                        type=int, default=42)
    return parser.parse_args()


def prepare_partition(partition: str, use_col: str, target_col: str,
                      size: float, unique_values: list,
                      df: pd.core.frame.DataFrame, output_dir: str,
                      save_format: str):

    unique_split = np.random.choice(unique_values,
                                    size=int(size * len(unique_values)))
    df = df[df[use_col].isin(unique_split)]

    if save_format == 'csv':
        df.to_csv(os.path.join(output_dir, f'{partition}.csv'), index=False)
    elif save_format == 'npy':
        X = df[[col for col in df.columns if col != target_col]].values
        y = df[target_col].values
        np.save(os.path.join(output_dir, f'X_{partition}.npy'), X)
        np.save(os.path.join(output_dir, f'y_{partition}.npy'), y)

    return unique_split


def generate_remaining_pairs(valid_pairs, all_pairs, limit_pairs):
    remaining_pairs = list()
    counts = 0
    for chunk in ichunked(all_pairs, 50000000):
        rem_pairs = list(set(chunk).difference(valid_pairs))
        if len(rem_pairs) > limit_pairs:
            rem_pairs = rem_pairs[:limit_pairs]
        remaining_pairs = remaining_pairs + rem_pairs
        counts += len(list(chunk))
        print(f'Processed = {counts}, Remaining = {len(remaining_pairs)}')
        if len(remaining_pairs) == limit_pairs:
            break

    return remaining_pairs


def prepare_interaction_matrix(args):
    data_dir = args.data_dir
    user_id_encoder_file = args.user_id_encoder_file
    book_title_encoder_file = args.book_title_encoder_file
    train_size = args.train_size
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    random_state = args.random_state

    np.random.seed(random_state)

    books_path = os.path.join(data_dir, 'Books.csv')
    ratings_path = os.path.join(data_dir, 'Ratings.csv')

    user_id_encoder = load(user_id_encoder_file)
    book_title_encoder = load(book_title_encoder_file)

    user_ids = [i for i in range(len(user_id_encoder.classes_))]
    book_titles = [i for i in range(len(book_title_encoder.classes_))]

    books = pd.read_csv(books_path)
    ratings = pd.read_csv(ratings_path)

    remove_book_outliers(books)
    books = books[['ISBN', 'Book-Title']]

    relevant_df = ratings.merge(books, on='ISBN', how='inner')
    relevant_df = relevant_df[['User-ID', 'Book-Rating', 'Book-Title']]

    relevant_df['User-ID'] = user_id_encoder.transform(relevant_df['User-ID']
                                                       .values).T
    relevant_df['Book-Title'] = book_title_encoder.transform(
        relevant_df['Book-Title'].values).T

    limit_pairs = relevant_df.shape[0]
    print(f'Number of valid pairs = {limit_pairs}')

    relevant_df = relevant_df.sort_values('Book-Rating', ignore_index=True,
                                          ascending=False)
    relevant_df = relevant_df.drop_duplicates(subset=['User-ID', 'Book-Title'])
    limit_pairs = relevant_df.shape[0]
    print(f'Number of valid pairs without duplicates = {limit_pairs}')

    # convert into a set of tuples for set difference operation
    valid_ub_pairs = set([tuple(item) for item in
                         relevant_df[['User-ID',
                                      'Book-Title']].values.tolist()])

    print(f'Number of tuples in set = {len(valid_ub_pairs)}')
    # print(f'First 5 elements = {list(valid_ub_pairs)[:5]}')

    all_pairs = product(user_ids, book_titles)
    remaining_pairs = generate_remaining_pairs(valid_ub_pairs, all_pairs,
                                               limit_pairs)

    valid_ub_pairs = np.asarray(list(map(list, valid_ub_pairs)))
    remaining_pairs = np.asarray(list(map(list, remaining_pairs)))

    print(valid_ub_pairs.shape)
    print(remaining_pairs.shape)

    data = np.vstack((valid_ub_pairs, remaining_pairs))
    print(data.shape)
    new_df = pd.DataFrame(data, columns=['User-ID', 'Book-Title'])
    new_df['interaction'] = np.asarray([1 for _ in range(len(valid_ub_pairs))]
                                       + [0 for _ in range(len(remaining_pairs)
                                                           )]).T
    print(new_df.shape)
    new_df = new_df.drop_duplicates(subset=['User-ID', 'Book-Title'])
    print(new_df.shape)

    unique_values = prepare_partition('train', 'User-ID', 'interaction',
                                      train_size, user_ids, new_df, output_dir,
                                      'npy')

    unique_values = prepare_partition('validation', 'User-ID', 'interaction',
                                      0.5, unique_values, new_df, output_dir,
                                      'npy')

    _ = prepare_partition('test', 'User-ID', 'interaction',
                          1, unique_values, new_df, output_dir, 'npy')


def main():
    args = get_arguments()
    prepare_interaction_matrix(args)


if __name__ == "__main__":
    main()
