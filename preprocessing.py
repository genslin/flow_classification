from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import logging_functions as log

PROJECT_ROOT = Path.cwd().resolve()
DATA_DIR = PROJECT_ROOT / "data"
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_models"


def get_train_val_test_indices(test_size=0.15, val_size=0.15, rng=42):
    """
    Generate stratified train, test, and validation indices for an ImageFolder dataset.

    This function performs a two-stage stratified split on an image classification dataset
    organized in subfolders (as expected by ``torchvision.datasets.ImageFolder``).
    It ensures that each subset (train, validation, test) maintains the same class
    distribution as the original dataset.

    The procedure works as follows:
        1. Load the dataset using ``datasets.ImageFolder`` and extract the class labels
           from the folder structure.
        2. Perform the first ``StratifiedShuffleSplit`` to divide the dataset into:
               - a training set
               - a combined test/validation set whose size equals ``test_size + val_size``.
        3. Perform a second ``StratifiedShuffleSplit`` on that combined subset to further
           divide it into distinct test and validation subsets, with their relative
           proportions defined by ``test_size`` and ``val_size``.
        4. Map the relative indices from the second split back to the original dataset
           indices so that all index arrays refer to the same base dataset.

    Parameters
    ----------
    test_size : float, optional (default=0.15)
        Proportion of the dataset to include in the test split.
        Must be between 0 and 1.
    val_size : float, optional (default=0.15)
        Proportion of the dataset to include in the validation split.
        Must be between 0 and 1.
    rng : int, optional (default=42)
        Random seed for reproducibility of the splits.

    Returns
    -------
    train_index : numpy.ndarray
        Array of indices corresponding to samples in the training subset.
    test_index : numpy.ndarray
        Array of indices corresponding to samples in the test subset.
    val_index : numpy.ndarray
        Array of indices corresponding to samples in the validation subset.

    Notes
    -----
    - The three subsets collectively span the entire dataset (train + test + val = 1.0).
    - The splits are stratified by class label to preserve the original class balance.
    - This function assumes that the global variable ``DATA_DIR`` points to the root
      directory of the dataset organized into subfolders (one per class).

    Examples
    --------
    >>> train_idx, test_idx, val_idx = get_train_test_validation_indices(test_size=0.2, val_size=0.1, rng=123)
    >>> len(train_idx) + len(test_idx) + len(val_idx)
    1000
    >>> len(test_idx) / 1000, len(val_idx) / 1000
    (0.2, 0.1)
    """

    base_ds = datasets.ImageFolder(root=DATA_DIR)
    folder_index = np.array([index for _, index in base_ds.samples])
    # Split data into train and test/val sets
    # https://scikit-learn.org/stable/modules/cross_validation.html#stratified-shuffle-split
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

    # In an image each pixels RGB values is a feature, for the sake of splitting in
    # this classification problem we only need to consider the classification of the image itself first
    dummy_features = np.zeros_like(folder_index)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(test_size + val_size), random_state=rng
    )
    train_index, test_and_val_index = next(sss1.split(dummy_features, folder_index))

    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=(test_size / (test_size + val_size)), random_state=rng
    )
    test_and_val_lables = folder_index[test_and_val_index]
    dummy_features = np.zeros_like(test_and_val_lables)
    test_index_rel, val_index_rel = next(
        sss2.split(dummy_features, test_and_val_lables)
    )

    # Second SSS splits relative to the index of test_and_val lables so these need
    # to be mapped back to original indices
    test_index = test_and_val_index[test_index_rel]
    val_index = test_and_val_index[val_index_rel]
    return train_index, val_index, test_index


def get_train_transform(
    image_size=(224, 224),
    random_rotation=10,
    brightness_jitter=0.2,
    contrast_jitter=0.2,
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
):
    train_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(random_rotation),
            transforms.ColorJitter(
                brightness=brightness_jitter, contrast=contrast_jitter
            ),
            transforms.ToTensor(),
            # From ImageNet, "universal defaults"
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )
    return train_tf


def get_evaluation_transform(
    image_size=(224, 224),
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
):
    eval_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )
    return eval_tf


def get_train_val_test_subsets(train_idx, val_idx, test_idx):
    base_ds_train = datasets.ImageFolder(root=DATA_DIR)
    base_ds_val = datasets.ImageFolder(root=DATA_DIR)
    base_ds_test = datasets.ImageFolder(root=DATA_DIR)

    train_ds = Subset(base_ds_train, train_idx)
    val_ds = Subset(base_ds_val, val_idx)
    test_ds = Subset(base_ds_test, test_idx)
    return train_ds, val_ds, test_ds


def get_test_subset(test_idx):
    base_ds_test = datasets.ImageFolder(root=DATA_DIR)
    test_ds = Subset(base_ds_test, test_idx)
    return test_ds


def set_train_val_test_subset_transforms(
    train_ds, val_ds, test_ds, train_tf=None, eval_tf=None
):
    if train_tf is None:
        train_tf = get_train_transform()
    if eval_tf is None:
        eval_tf = get_evaluation_transform()

    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = eval_tf
    test_ds.dataset.transform = eval_tf
    return train_ds, val_ds, test_ds

def set_test_subset_transform(
    test_ds, eval_tf=None
):
    if eval_tf is None:
        eval_tf = get_evaluation_transform()

    test_ds.dataset.transform = eval_tf
    return test_ds

def get_train_val_test_dataloaders(
    train_ds, val_ds, test_ds, batch_size=32, num_workers=2, pin_memory=True
):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader

def get_test_dataloader(
    test_ds, batch_size=32, num_workers=2, pin_memory=True
):
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader

def get_quick_test_loader():
    _, _, test_idx = get_train_val_test_indices()
    _, _, sub = get_train_val_test_subsets(test_idx, test_idx, test_idx)
    _, _, sub = set_train_val_test_subset_transforms(sub, sub, sub)
    _, _, loader = get_train_val_test_dataloaders(sub, sub, sub)
    return loader


def get_dataloaders_complete_preprocessing(
    test_size=0.15,
    val_size=0.15,
    rng=42,
    image_size=(224, 224),
    random_rotation=10,
    brightness_jitter=0.2,
    contrast_jitter=0.2,
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    batch_size=32,
    num_workers=2,
    pin_memory=True,
):
    train_idx, val_idx, test_idx = get_train_val_test_indices(
        test_size=test_size, val_size=val_size, rng=rng
    )
    train_transform = get_train_transform(
        image_size=image_size,
        random_rotation=random_rotation,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    eval_transform = get_evaluation_transform(
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    train_ds, val_ds, test_ds = get_train_val_test_subsets(
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
    )
    train_ds, val_ds, test_ds = set_train_val_test_subset_transforms(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_tf=train_transform,
        eval_tf=eval_transform,
    )
    train_loader, val_loader, test_loader = get_train_val_test_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader.indices = train_idx
    val_loader.indices = val_idx
    test_loader.indices = test_idx
    return train_loader, val_loader, test_loader


def get_dataloaders_from_indices(
    train_idx,
    val_idx,
    test_idx,
    image_size=(224, 224),
    random_rotation=10,
    brightness_jitter=0.2,
    contrast_jitter=0.2,
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    batch_size=32,
    num_workers=2,
    pin_memory=True,
):
    """
    Recreate train/val/test DataLoaders from explicit indices into the
    ImageFolder dataset at DATA_DIR.

    This is the 'inverse' of get_dataloaders_complete_preprocessing when
    you already know which samples belong in each split.
    """
    # 1) Build transforms from the same hyperparameters you used originally
    train_transform = get_train_transform(
        image_size=image_size,
        random_rotation=random_rotation,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    eval_transform = get_evaluation_transform(
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    # 2) Build Subset objects using the provided indices
    train_ds, val_ds, test_ds = get_train_val_test_subsets(
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
    )

    # 3) Attach transforms
    train_ds, val_ds, test_ds = set_train_val_test_subset_transforms(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_tf=train_transform,
        eval_tf=eval_transform,
    )

    # 4) Wrap in DataLoaders
    train_loader, val_loader, test_loader = get_train_val_test_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_test_dataloader_from_indices(
    test_idx,
    image_size=(224, 224),
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    batch_size=32,
    num_workers=2,
    pin_memory=True,
):
    """
    Recreate train/val/test DataLoaders from explicit indices into the
    ImageFolder dataset at DATA_DIR.

    This is the 'inverse' of get_dataloaders_complete_preprocessing when
    you already know which samples belong in each split.
    """
    # 1) Build transforms from the same hyperparameters you used originally
    eval_transform = get_evaluation_transform(
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    # 2) Build Subset object using the provided indices
    test_ds = get_test_subset(test_idx=test_idx)

    # 3) Attach transform
    test_ds = set_test_subset_transform(
        test_ds=test_ds,
        eval_tf=eval_transform,
    )

    # 4) Wrap in DataLoaders
    test_loader = get_test_dataloader(
        test_ds=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader


def save_train_val_test_indices(train_loader, val_loader, test_loader, model):
    subdirectory = SAVED_MODEL_DIR / model.name / "data splits"
    subdirectory.mkdir(parents=True, exist_ok=True)
    np.save(subdirectory / "train_idx.npy", train_loader.indices)
    np.save(subdirectory / "val_idx.npy", val_loader.indices)
    np.save(subdirectory / "test_idx.npy", test_loader.indices)
    logger = log.get_model_logger(model_name=model.name)
    logger.info("Training, validation, and test indices saved")


def load_train_val_test_indices(model):
    subdirectory = SAVED_MODEL_DIR / model.name / "data splits"
    subdirectory.mkdir(parents=True, exist_ok=True)
    train_idx = np.load(subdirectory / "train_idx.npy")
    val_idx = np.load(subdirectory / "val_idx.npy")
    test_idx = np.load(subdirectory / "test_idx.npy")
    logger = log.get_model_logger(model_name=model.name)
    logger.info("Training, validation, and test indices loaded")
    return train_idx, val_idx, test_idx


def load_train_val_test_dataloaders(
    model,
    image_size=(224, 224),
    random_rotation=10,
    brightness_jitter=0.2,
    contrast_jitter=0.2,
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    batch_size=32,
    num_workers=2,
    pin_memory=True,
):
    train_idx, val_idx, test_idx = load_train_val_test_indices(model)
    train_loader, val_loader, test_loader = get_dataloaders_from_indices(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        image_size=image_size,
        random_rotation=random_rotation,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader.indices = train_idx
    val_loader.indices = val_idx
    test_loader.indices = test_idx
    return train_loader, val_loader, test_loader

def load_test_dataloader(
    model,
    image_size=(224, 224),
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    batch_size=32,
    num_workers=2,
    pin_memory=True,
):
    _, _, test_idx = load_train_val_test_indices(model)
    test_loader = get_test_dataloader_from_indices(
        test_idx=test_idx,
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader.indices = test_idx
    return test_loader