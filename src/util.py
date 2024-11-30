import torch
import numpy as np

def compute_metrics(output, target, top_k=[1, 3, 10]):
    """
    Compute MRR, Hit@1, Hit@3, Hit@10 for evaluation.
    :param output: model predictions (tensor)
    :param target: ground truth labels (tensor)
    :param top_k: List of values for Hit@k metrics
    :return: Dictionary with computed metrics.
    """
    metrics = {
        'MRR': 0.0,
        'Hit@1': 0.0,
        'Hit@3': 0.0,
        'Hit@10': 0.0
    }
    
    # Get top k predictions
    _, predicted = torch.topk(output, max(top_k), dim=1)
    
    for k in top_k:
        hits_at_k = (predicted[:, :k] == target.unsqueeze(1).expand_as(predicted[:, :k])).float()
        metrics[f'Hit@{k}'] = hits_at_k.mean().item()

    # Compute MRR
    ranks = torch.argsort(output, dim=1, descending=True)
    rank = torch.where(ranks == target.unsqueeze(1), 1, 0).float()
    mrr = torch.sum(1 / (torch.nonzero(rank)[:, 1] + 1)).item()
    metrics['MRR'] = mrr / len(output)

    return metrics

def save_model(model, optimizer, epoch, loss, model_save_path):
    """
    Save the model and optimizer states.
    :param model: The model to be saved.
    :param optimizer: The optimizer state to be saved.
    :param epoch: The current epoch number.
    :param loss: The current loss to be saved.
    :param model_save_path: Path where the model will be saved.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, model_save_path)
    print(f"Model saved to {model_save_path}")

def load_model(model, optimizer, model_save_path):
    """
    Load the model and optimizer states from a checkpoint.
    :param model: The model to load the weights into.
    :param optimizer: The optimizer to load the state into.
    :param model_save_path: Path where the model checkpoint is stored.
    :return: model, optimizer, epoch, loss
    """
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {model_save_path}, starting from epoch {epoch}")
    return model, optimizer, epoch, loss

def create_optimizer(model, lr=0.001, weight_decay=0.0):
    """
    Create an optimizer for the model.
    :param model: The model for which the optimizer is created.
    :param lr: Learning rate for the optimizer.
    :param weight_decay: Weight decay (L2 regularization).
    :return: optimizer
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def early_stopping(val_loss, best_val_loss, patience, counter):
    """
    Early stopping to halt training if validation loss doesn't improve for a certain number of epochs.
    :param val_loss: Current validation loss.
    :param best_val_loss: Best validation loss so far.
    :param patience: Number of epochs to wait before stopping.
    :param counter: Current counter of how many epochs without improvement.
    :return: True if early stopping should be triggered, otherwise False.
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print(f"Early stopping triggered after {counter} epochs without improvement.")
        return True, best_val_loss, counter
    return False, best_val_loss, counter

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    :param seed: The seed to use for random number generation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")
