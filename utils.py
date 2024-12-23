import torch
from tqdm import tqdm
from contrastive_loss import simclr_loss_vectorized
from utils import train_val  # Add this to import train_val



def train(model, data_loader, train_optimizer, epoch, epochs, batch_size=32, temperature=0.5, device='cuda'):
    """
    Trains the SimCLR model for one epoch using contrastive learning.

    Inputs:
    - model: SimCLR model object defined in model.py.
    - data_loader: DataLoader object for training data.
    - train_optimizer: Optimizer object for updating model parameters.
    - epoch: Current epoch number (int).
    - epochs: Total number of epochs (int).
    - batch_size: Number of samples in each batch.
    - temperature: Temperature parameter for contrastive loss.
    - device: Device for training ('cuda' or 'cpu').

    Returns:
    - Average training loss for the epoch.
    """
    model.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for data_pair in train_bar:
        x_i, x_j, target = data_pair  # x_i, x_j are augmented views of the same image
        x_i, x_j = x_i.to(device), x_j.to(device)

        # Forward pass through the model
        out_left = model(x_i)  # Projection for the first view
        out_right = model(x_j)  # Projection for the second view

        # Compute contrastive loss
        loss = simclr_loss_vectorized(out_left, out_right, tau=temperature, device=device)

        # Backpropagation and optimization
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(f'Train Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f}')

    return total_loss / total_num


def test(model, memory_data_loader, test_data_loader, epoch, epochs, c, temperature=0.5, k=200, device='cuda'):
    """
    Evaluates the SimCLR model using a k-nearest neighbor classifier on the representations.

    Inputs:
    - model: SimCLR model object defined in model.py.
    - memory_data_loader: DataLoader object for the memory dataset.
    - test_data_loader: DataLoader object for the test dataset.
    - epoch: Current epoch number (int).
    - epochs: Total number of epochs (int).
    - c: Number of classes in the dataset (int).
    - temperature: Temperature parameter for contrastive loss.
    - k: Number of nearest neighbors for kNN classifier.
    - device: Device for testing ('cuda' or 'cpu').

    Returns:
    - Top-1 accuracy (%).
    - Top-5 accuracy (%).
    """
    model.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []

    with torch.no_grad():
        # Generate feature bank for kNN
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            features = model(data.to(device))
            feature_bank.append(features)

        # Stack feature bank and corresponding labels
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()  # [D, N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)

        # Evaluate on test set
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device), target.to(device)
            features = model(data)  # Compute features for test set

            total_num += data.size(0)

            # Compute cosine similarity between test features and feature bank
            sim_matrix = torch.mm(features, feature_bank)

            # Retrieve top-k neighbors
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # Weighted voting for classification
            one_hot_label = torch.zeros(data.size(0) * k, c, device=device)
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            # Compute top-1 and top-5 accuracy
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            test_bar.set_description(f'Test Epoch: [{epoch}/{epochs}] Acc@1: {total_top1 / total_num * 100:.2f}% Acc@5: {total_top5 / total_num * 100:.2f}%')

    return total_top1 / total_num * 100, total_top5 / total_num * 100
