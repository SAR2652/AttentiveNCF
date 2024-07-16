import torch
from torch.utils.data import Dataset


def remove_book_outliers(books_df):
    outliers = [0, '0', 'DK Publishing Inc', 'Gallimard']
    for outlier in outliers:
        books_df.drop(books_df[
            books_df['Year-Of-Publication'] == outlier].index,
                   inplace=True)


class InteractionDataset(Dataset):
    def __init__(self, users, books, interactions):
        self.users = users
        self.books = books
        self.interactions = interactions

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        book = self.books[index]
        interaction = self.interactions[index]

        return {
            'user': torch.tensor(user, dtype=torch.long),
            'item': torch.tensor(book, dtype=torch.long),
            'interaction': torch.tensor(interaction, dtype=torch.long)
        }
