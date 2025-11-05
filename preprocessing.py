from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets

PROJECT_ROOT = Path.cwd().resolve()
DATA_DIR = PROJECT_ROOT / "data"

def get_train_test_validation_indices(rng=42):
    base_ds = datasets.ImageFolder(root=DATA_DIR)
    folder_index = np.array([index for _, index in base_ds.samples])
    # Split data into train and test/val sets
    # https://scikit-learn.org/stable/modules/cross_validation.html#stratified-shuffle-split
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

    # In an image each pixels RGB values is a feature, for the sake of splitting in 
    # this classification problem we only need to consider the classification of the image itself first
    dummy_features = np.zeros_like(folder_index)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=rng)
    train_index, test_and_val_index = next(sss1.split(dummy_features, folder_index))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=rng)
    test_and_val_lables = folder_index[test_and_val_index]
    dummy_features = np.zeros_like(test_and_val_lables)
    test_index_rel, val_index_rel = next(sss2.split(dummy_features, test_and_val_lables))

    # Second SSS splits relative to the index of test_and_val lables so these need
    # to be mapped back to original indices
    test_index = test_and_val_index[test_index_rel]
    val_index  = test_and_val_index[val_index_rel]
    return train_index, test_index, val_index

print(get_train_test_validation_indices())