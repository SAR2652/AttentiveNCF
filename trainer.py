import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from joblib import load
import torch.nn.functional as F
from model import RecommenderNet
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from common_utils import InteractionDataset
from sklearn.metrics import accuracy_score, f1_score, \
    precision_score, recall_score


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        help='Directory containing partition data as CSV',
                        type=str, default='./partition_data')
    parser.add_argument('--random_state',
                        help='Random Seed for Reproducibility',
                        type=int, default=42)
    parser.add_argument('--output_dir',
                        help='Directory to store output',
                        type=str, default='./output')
    parser.add_argument('---embedding_dim',
                        help='Dimensions of Embedding Layer',
                        type=int, default=64)
    parser.add_argument('--batch_size',
                        help='Batch Size for Model Training',
                        type=int, default=64)
    parser.add_argument('--learning_rate',
                        help='Learning rate for model training',
                        type=float, default='1e-4')
    parser.add_argument('--user_id_encoder_file',
                        help='Encoder file used to encode/decode User IDs',
                        type=str, default='./output/user_id_encoder.joblib')
    parser.add_argument('--book_title_encoder_file',
                        help='Encoder file used to encode/decode Book Titles',
                        type=str, default='./output/book_title_encoder.joblib')
    parser.add_argument('--num_ancf_layers',
                        help='Number of ANCF layers',
                        type=int, default=1)
    parser.add_argument('--optimizer',
                        help='Optimizer to be used',
                        type=str, choices=['Adam', 'SGD'], default='Adam')
    parser.add_argument('--epochs',
                        help='Number of training epochs',
                        type=int, default=10)
    return parser.parse_args()


def get_predictions(model, device: torch.device, dataloader, return_gt=False):

    # set model to evaluation mode
    model.eval()

    # Inference in minibatches

    # store predictions of every batch as a list of lists
    batch_predictions = list()

    if return_gt:
        batch_gt = list()

    for _, batch in enumerate(dataloader):
        # disable gradients for faster inference
        with torch.no_grad():
            users = batch['user'].to(device)
            items = batch['item'].to(device)

            if return_gt:
                ratings = batch['interaction']

            logits = model(users, items)

            # Apply Softmax on logits to get probabilities
            outputs = F.softmax(logits, dim=1)

            # determine predicted class
            _, y_pred_batch = torch.max(outputs, 1)

        # flatten batch outputs
        pred_flat = y_pred_batch.detach().cpu().numpy().reshape(-1) \
            .tolist()
        batch_predictions.append(pred_flat)

        if return_gt:
            gt_flat = ratings.detach().cpu().numpy().reshape(-1) \
                .tolist()
            batch_gt.append(gt_flat)

    # flatten list of lists and convert to numpy array
    predictions = sum(batch_predictions, list())
    predictions_np = np.asarray(predictions)

    if return_gt:
        ground_truth = sum(batch_gt, list())
        ground_truth_np = np.asarray(ground_truth)
        return predictions_np, ground_truth_np
    else:
        # return predictions
        return predictions_np


def train_model(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    random_state = args.random_state
    embed_dim = args.embedding_dim
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    user_id_encoder_file = args.user_id_encoder_file
    book_title_encoder_file = args.book_title_encoder_file
    num_ancf_layers = args.num_ancf_layers
    optimizer_choice = args.optimizer
    epochs = args.epochs

    torch.manual_seed(random_state)

    X_train_path = os.path.join(data_dir, 'X_train.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')

    X_validation_path = os.path.join(data_dir, 'X_validation.npy')
    y_validation_path = os.path.join(data_dir, 'y_validation.npy')

    X_test_path = os.path.join(data_dir, 'X_test.npy')
    y_test_path = os.path.join(data_dir, 'y_test.npy')

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)

    X_validation = np.load(X_validation_path)
    y_validation = np.load(y_validation_path)

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    train_data = InteractionDataset(X_train[:, 0], X_train[:, 1], y_train)
    validation_data = InteractionDataset(X_validation[:, 0],
                                         X_validation[:, 1], y_validation)
    test_data = InteractionDataset(X_test[:, 0], X_test[:, 1], y_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=batch_size,
                                shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                 shuffle=False)

    user_id_encoder = load(user_id_encoder_file)
    book_title_encoder = load(book_title_encoder_file)

    n_users = len(user_id_encoder.classes_)
    n_items = len(book_title_encoder.classes_)

    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        accelerator = 'mps'
    device = torch.device(accelerator)

    criterion = nn.CrossEntropyLoss()

    all_user_ids_tensor = torch.tensor([i for i in range(n_users)],
                                       dtype=torch.long).to(device)
    all_item_ids_tensor = torch.tensor([i for i in range(n_items)],
                                       dtype=torch.long).to(device)

    model = RecommenderNet(n_users, n_items, embed_dim, num_ancf_layers)
    model = model.to(device)

    match optimizer_choice:

        case 'Adam':
            optimizer = Adam(model.parameters(), lr=learning_rate)

        case 'SGD':
            optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        case _:
            print('Invalid choice of optimizer!')
            sys.exit(1)

    best_val_acc = -np.inf

    for epoch in range(epochs):
        model.train()
        batch_losses = list()
        total = 0

        for i, batch in enumerate(train_dataloader):
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            y_true = batch['interaction']
            total += y_true.size(0)
            y_true = y_true.to(device)

            optimizer.zero_grad()

            logits = model(users, items, all_user_ids_tensor,
                           all_item_ids_tensor)

            loss = criterion(logits, y_true)
            loss.backward()

            optimizer.step()

            batch_losses.append(loss.item())
            print(f"{total}/{X_train.shape[0]} samples processed")

        epoch_loss = np.mean(batch_losses)

        y_pred_train = get_predictions(model, device, train_dataloader)
        y_pred_val = get_predictions(model, device, val_dataloader)
        y_pred_test = get_predictions(model, device, test_dataloader)

        train_acc = accuracy_score(y_train, y_pred_train) * 100
        val_acc = accuracy_score(y_validation, y_pred_val) * 100
        test_acc = accuracy_score(y_test, y_pred_test) * 100

        print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f} "
              f"Training Accuracy = {train_acc:.1f}%, "
              f"Validation Accuracy = {val_acc:.1f}%, "
              f"Test Accuracy = {test_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            test_precision = precision_score(y_test, y_pred_test) * 100
            test_recall = recall_score(y_test, y_pred_test) * 100
            test_f1_score = f1_score(y_test, y_pred_test) * 100

            state = {
                'val_accuracy': best_val_acc,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'test_f1_score': test_f1_score,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            torch.save(state, os.path.join(output_dir, 'best_model.pth'))

    y_pred_test = get_predictions(model, device, test_dataloader)

    test_acc = accuracy_score(y_test, y_pred_test) * 100
    print(f"Test Accuracy = {test_acc:.1f}%")

    test_precision = precision_score(y_test, y_pred_test) * 100
    print(f"Precision = {test_precision:.1f}%")

    test_recall = recall_score(y_test, y_pred_test) * 100
    print(f"Recall = {test_recall:.1f}%")

    test_f1_score = f1_score(y_test, y_pred_test) * 100
    print(f"Test F1 Score = {test_f1_score:.1f}%")


def main():
    args = get_arguments()
    train_model(args)


if __name__ == '__main__':
    main()
