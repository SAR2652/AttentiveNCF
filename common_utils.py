import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    def __init__(self, users, books, interactions):
        self.users = users
        self.books = books
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, index):
        user = self.users[index]
        book = self.books[index]
        interaction = self.interactions[index]

        return {
            'user': torch.tensor(user, dtype=torch.long),
            'item': torch.tensor(book, dtype=torch.long),
            'interaction': torch.tensor(interaction, dtype=torch.long)
        }
