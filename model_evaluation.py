import logging_functions as log
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path.cwd().resolve()
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_models"

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_loop(dataloader, model, loss_fn,):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    logger = log.get_model_logger(model.name)
    logger.info(f"Testing model: {model.name}")
    model.eval()
    logger.info("Model in eval")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    all_preds = []
    all_labels = []
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            logits = model(X)
            pred = logits.argmax(1)

            test_loss += loss_fn(logits, y).item()
            correct += (pred == y).type(torch.float).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    correct /= size
    logger.info(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    cm = confusion_matrix(all_labels, all_preds)
    return cm

def plot_confusion_matrix(model, cm, plt_name=None, save_plot=False, normalize=True, show_plot=True):
    """Plot (and optionally save) a confusion matrix for a given model."""

    # Create a new figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot heatmap on the specific axis
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=model.class_names,
        yticklabels=model.class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    fig.tight_layout()

    # Save before (or independent of) show
    if save_plot:
        subdirectory = SAVED_MODEL_DIR / model.name / "confusion matrix plots"
        subdirectory.mkdir(parents=True, exist_ok=True)

        # Default name if none provided
        if plt_name is None:
            plt_name = "confusion_matrix_normalized" if normalize else "confusion_matrix"

        if not plt_name.endswith(".png"):
            plt_name += ".png"

        path = subdirectory / plt_name
        fig.savefig(path, dpi=300)
        print(f"Confusion matrix saved to: {path}")

    # Show the plot (optional, so Colab / scripts can disable it)
    if show_plot:
        plt.show()

    # Free the figure from memory
    plt.close(fig)