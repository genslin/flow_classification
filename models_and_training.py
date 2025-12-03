import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchvision import models
from pathlib import Path
import pandas as pd
import logging_functions as log
import model_evaluation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path.cwd().resolve()
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_models"


def get_loss_fn(loss_fn_name="CrossEntropyLoss"):
    if loss_fn_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    if loss_fn_name == "NLLLoss":
        return nn.NLLLoss()
    raise ValueError(f"Unknown loss function: {loss_fn_name}")


def get_optimizer(
    model,
    layer="fc",
    optimizer_name="Adam",
    fc_learning_rate=1e-3,
    layer4_learning_rate=1e-4,
    momentum=0.9,
):
    optimizer_name = optimizer_name.lower()
    if layer == "fc":
        params = model.fc.parameters()

    elif layer == "layer4":
        params = [
            {"params": model.fc.parameters(), "lr": fc_learning_rate},
            {"params": model.layer4.parameters(), "lr": layer4_learning_rate},
        ]
    else:
        raise ValueError(f"Unknown layer selection: {layer}")

    if optimizer_name == "adam":
        if layer == "fc":
            return Adam(params, lr=fc_learning_rate)
        else:
            return Adam(params)

    elif optimizer_name == "sgd":
        if layer == "fc":
            return SGD(params, lr=fc_learning_rate, momentum=momentum)
        else:
            return SGD(params, momentum=momentum)


def get_pre_trained_resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1, model_name="default_resnet18"
):
    model = models.resnet18(weights=weights)
    model.name = model_name
    logger = log.get_model_logger(model_name=model.name)
    logger.info("Loaded resnet18 architecture")
    model = replace_fc(model)
    model.to(device=DEVICE)
    model.class_names = ["Bubbly", "Slug", "Churn", "Taylor"]
    return model


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    logger = log.get_model_logger(model_name=model.name)
    logger.info("All model parameters frozen")
    return model


def replace_fc(model):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 4)
    logger = log.get_model_logger(model_name=model.name)
    logger.info("fc replaced")
    return model


def unfreeze_fc(model):
    for p in model.fc.parameters():
        p.requires_grad = True
    logger = log.get_model_logger(model_name=model.name)
    logger.info("fc unfrozen")
    return model


def unfreeze_last_block(model):
    for p in model.layer4.parameters():
        p.requires_grad = True
    logger = log.get_model_logger(model_name=model.name)
    logger.info("Layer 4 unfrozen")
    return model


def train_loop(dataloader, model, loss_fn, optimizer, epoch, head_only=True):
    size = len(dataloader.dataset)
    logger = log.get_model_logger(model.name)
    logger.info(f"Training model: {model.name}")
    logger.info(f"Loss Function: {loss_fn.__class__.__name__}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    performance_data = []
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    logger.info("Model in train")
    logger.info(f"Starting epoch: {epoch}")
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), batch * dataloader.batch_size + len(X)
        performance_data.append(
            {
                "epoch": epoch,
                "batch": batch,
                "loss": loss,
                "current": current,
                "head_only": head_only,
            }
        )
        logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    logger.info("Finished epoch")
    return model, performance_data


def train_model_for_epochs(
    starting_epoch,
    number_of_epochs_to_train,
    train_dataloader,
    val_dataloader,
    model,
    loss_fn,
    optimizer,
    prior_performance_data=None,
    head_only=True,
    save_confusion_matrix=False,
):
    all_performance_data = (
        [] if prior_performance_data is None else list(prior_performance_data)
    )
    for epoch in range(starting_epoch, starting_epoch + number_of_epochs_to_train):
        model, performance_data = train_loop(
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            head_only=head_only,
        )
        all_performance_data.extend(performance_data)
        confusion_matrix = model_evaluation.test_loop(
            dataloader=val_dataloader,
            model=model,
            loss_fn=loss_fn,
        )
        if save_confusion_matrix:
            plt_name = f"epoch_{epoch}_head_only_{head_only}"
            model_evaluation.plot_confusion_matrix(
                model=model,
                cm=confusion_matrix,
                plt_name=plt_name,
                save_plot=True,
                show_plot=False,
            )
    return model, all_performance_data, confusion_matrix


def save_performance_data(performance_data, model):
    subdirectory = SAVED_MODEL_DIR / model.name / "performance data"
    subdirectory.mkdir(parents=True, exist_ok=True)
    path = subdirectory / "performance_data.csv"
    performance_data = pd.DataFrame(performance_data)
    performance_data.to_csv(path, index=False)
    logger = log.get_model_logger(model_name=model.name)
    logger.info("Performance Data Saved")


def load_performance_data(model, plot_only=False):
    subdirectory = SAVED_MODEL_DIR / model.name / "performance data"
    subdirectory.mkdir(parents=True, exist_ok=True)
    path = subdirectory / "performance_data.csv"
    logger = log.get_model_logger(model_name=model.name)
    if path.exists():
        try:
            performance_data = pd.read_csv(path).to_dict(orient="records")
            if not plot_only:
                logger.info("Loaded performance data")
        except pd.errors.EmptyDataError:
            performance_data = []
            if not plot_only:
                logger.info(
                    "Performance data file was empty, starting new performance data record"
                )
    else:
        if not plot_only:
            logger.info("New model, starting performance data record")
        performance_data = []
    return performance_data


def save_model(model):
    model_name = model.name
    model_weights = model_name + "_weights.pth"
    subdirectory = SAVED_MODEL_DIR / model.name / "weights"
    subdirectory.mkdir(parents=True, exist_ok=True)
    path = subdirectory / model_weights
    logger = log.get_model_logger(model_name=model.name)
    if path.exists():
        overwrite = (
            input(f"Model '{model_weights}' exists. Overwrite? (Y/N): ").strip().upper()
        )
        if overwrite == "N":
            logger.info("Model not saved")
            return
        elif overwrite != "Y":
            logger.info("Invalid entry, Model not saved")
            return
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to: {path}")


def load_model(model_name="default_resnet18", new_model=True):
    if model_name == "default_resnet18":
        model = get_pre_trained_resnet18(model_name=model_name)
        logger = log.get_model_logger(model_name=model.name)
        logger.info(f"Loaded: {model.name}")
    elif new_model:
        model = get_pre_trained_resnet18(model_name=model_name)
        logger = log.get_model_logger(model_name=model.name)
        logger.info(
            f"Created {model.name} from pre_trained_resnet18.\n"
            "If you have previously saved weights to this model name,\n"
            "they will be overwritten if you continue.\n"
            "Quit (Ctrl+C) and pass the --load-existing argument if you do not want this to happen."
        )
    else:
        model = get_pre_trained_resnet18(weights=None, model_name=model_name)
        model_weights = model_name + "_weights.pth"
        path = SAVED_MODEL_DIR / model.name / "weights" / model_weights
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        logger = log.get_model_logger(model_name=model.name)
        logger.info(f"{model.name} loaded from: {path}")
    return model
