import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y)
        self.progress.close()

    # (TODO) Grow the decision tree and return it
    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth : int = 0):
        min_samples_split = 10
        num_samples_per_class = np.bincount(y)
        predicted_class = np.argmax(num_samples_per_class)
        if depth >= self.max_depth or len(set(y)) == 1 or X.empty or len(y) < min_samples_split:
            return {'type': 'leaf', 'class': predicted_class}
        if depth >= self.max_depth or len(set(y)) == 1 or X.empty:
            return {'type': 'leaf', 'class': predicted_class}
        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return {'type': 'leaf', 'class': predicted_class}
        left_X, left_y, right_X, right_y = self._split_data(X, y, feature_index, threshold)
        self.progress.update(1)
        return {
            'type': 'node',
            'feature_index': feature_index,
            'threshold': threshold,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }

    # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
    def predict(self, X: pd.DataFrame)->np.ndarray:
        if isinstance(X, list):
            X = pd.DataFrame([x.flatten() for x in X])
        id_pred_map = {}
        for idx, x in X.iterrows():
            pred = self._predict_tree(x, self.tree)
            id_pred_map[idx] = pred
        return [id_pred_map[idx] for idx in X.index]

    # (TODO) Recursive function to traverse the decision tree
    def _predict_tree(self, x, tree_node):
        if tree_node['type'] == 'leaf':
            return tree_node['class']
        if x[tree_node['feature_index']] <= tree_node['threshold']:
            return self._predict_tree(x, tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])

    # (TODO) split one node into left and right node 
    def _split_data(self, X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        feature = X.iloc[:, feature_index]
        left_mask = feature <= threshold
        right_mask = ~left_mask
        left_dataset_X = X[left_mask]
        left_dataset_y = y[left_mask]
        right_dataset_X = X[right_mask]
        right_dataset_y = y[right_mask]
        return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

    # (TODO) Use Information Gain to find the best split for a dataset
    def _best_split(self, X: pd.DataFrame, y: np.ndarray):
        best_gain = -1
        best_feature = None
        best_thresh = None
        current_entropy = self._entropy(y)
        for feature_index in range(X.shape[1]):
            values = X.iloc[:, feature_index].values
            thresholds = np.percentile(values, [25, 50, 75])
            for threshold in thresholds:
                left_y = y[values <= threshold]
                right_y = y[values > threshold]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                left_entropy = self._entropy(left_y)
                right_entropy = self._entropy(right_y)
                weighted_entropy = (len(left_y) * left_entropy + len(right_y) * right_entropy) / len(y)
                info_gain = current_entropy - weighted_entropy
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature_index
                    best_thresh = threshold
        return best_feature, best_thresh

    # (TODO) Return the entropy
    def _entropy(self, y: np.ndarray)->float:
        unique_classes, class_counts = np.unique(y, return_counts=True)
        probabilities = class_counts / len(y)
        entropy_value = -np.sum(probabilities * np.log2(probabilities))
        return entropy_value

# (TODO) Use the model to extract features from the dataloader, return the features and labels
def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            outputs = model(images)
            features.extend(outputs.cpu().numpy())
            labels.extend(lbls.numpy())
    features_df = pd.DataFrame(features)
    labels_np = np.array(labels)
    return features_df, labels_np

# (TODO) Use the model to extract features from the dataloader, return the features and path of the images
def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    model.eval()
    features = []
    paths = []
    with torch.no_grad():
        for images, image_names in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            outputs = model(images)
            features.extend(outputs.cpu().numpy())
            paths.extend(image_names)
    sorted_indices = sorted(range(len(paths)), key=lambda i: int(paths[i]))
    sorted_features = [features[i] for i in sorted_indices]
    sorted_paths = [paths[i] for i in sorted_indices]
    features_df = pd.DataFrame(sorted_features)
    return features_df, sorted_paths