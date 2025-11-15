import torch
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_accuracy(model, loader, device=DEVICE):
    model.eval()
    all_preds, all_targets = [], []
    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return accuracy_score(all_targets.numpy(), all_preds.numpy())

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
