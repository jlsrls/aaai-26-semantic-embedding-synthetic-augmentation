"""Utility functions for data transformation, analysis, and model evaluation.

This module provides helper functions for the training pipeline including:
- Data format conversions (DataFrame <-> DataLoader, ID transformers)
- Training log consolidation and analysis
- Model performance evaluation metrics
- Model naming and file management utilities
"""
from __future__ import annotations
import os
import re
import pickle
from typing import Any

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from random import randint
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


"""
BATCH TRAINING LOG FORMAT

This module uses a specific DataFrame format for batch-based training and testing logs.
These logs are generated during training and testing, written to disk per-epoch, then
consolidated into single files for easier analysis.

Training Log Format (from train_epoch):
- batch_size: Number of samples in this batch
- batch_pred: NumPy array of model predictions for the batch
- batch_true: NumPy array of true labels for the batch

Test Log Format (from test_dataloader):
- batch_size: Number of samples in this batch  
- batch_data_indiv: Individual IDs from input features [:, 0]
- batch_data_year: Year/wave IDs from input features [:, 1] 
- batch_data_question: Question IDs from input features [:, 2]
- batch_pred: NumPy array of model predictions for the batch
- batch_true: NumPy array of true labels for the batch

These DataFrames are saved as parquet files with naming convention:
{model_name}-ep{epoch}-logs-{type}.parquet where type is 'train', 'test', or 'test-on-train'

Consolidation combines all epoch files into:
{model_name}-full-logs-{type}.parquet for analysis functions.
"""

def parse_into_array(dat_frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a melted survey DataFrame into feature and label arrays for PyTorch.
    
    Extracts the three ID columns (individual, wave, question) as features and
    the value column as labels, converting to appropriate NumPy arrays for model input.
    
    Args:
        dat_frame (pd.DataFrame): Melted survey data with columns 'indiv_id', 'wave_id', 
                                  'question_id', and 'value'
    
    Returns:
        tuple: (features, labels) where:
            - features: NumPy array of shape (N, 3) containing [indiv_id, wave_id, question_id]
            - labels: NumPy array of shape (N,) containing binary response values as float32
    """
    features = torch.from_numpy(np.array((dat_frame['indiv_id'], dat_frame['wave_id'], dat_frame['question_id'])).T).to("cuda")
    labels = torch.from_numpy(np.array(dat_frame['value'], dtype=np.float32)).to("cuda") # needs to be a float32 to match outputs from the imputation model
    return features, labels

def frame_to_dataloader(dat_frame: pd.DataFrame, *dl_args: Any, **dl_kwargs: Any) -> torch.utils.data.DataLoader:
    """
    Converts a melted survey DataFrame into a PyTorch DataLoader.
    
    Transforms survey data through parse_into_array and creates a DataLoader
    suitable for batch-based training or testing.
    
    Args:
        dat_frame (pd.DataFrame): Melted survey data with required columns
        *dl_args: Positional arguments passed to torch.utils.data.DataLoader
        **dl_kwargs: Keyword arguments passed to torch.utils.data.DataLoader
                     (e.g., batch_size, shuffle)
    
    Returns:
        torch.utils.data.DataLoader: DataLoader yielding (features, labels) tuples
                                     where features are shape (batch_size, 3)
    """
    features, labels = parse_into_array(dat_frame)

    # we zip the two arrays together into an N-element list of 2-element tuples,
    # where the first element in each tuple is an array containing features w/ shape (3,) and the second is the label
    zipped_list = [i for i in zip(features, labels)]

    # then we turn the zipped list into a dataloader object for throwing at the training data
    return torch.utils.data.DataLoader(
        zipped_list, *dl_args, **dl_kwargs # type: ignore
    )

def get_model_name() -> str:
    """
    Generates a random hexadecimal model name for experiment tracking.
    
    Creates a unique identifier by generating a random 32-bit integer
    and converting to hexadecimal (without '0x' prefix).
    
    Returns:
        str: Random hexadecimal string (e.g., 'a3b2c1d4')
    """
    return hex(randint(0, (2**32)-1))[2:]


### log consolidation

def check_consolidated(model_name: str, files: list[str]) -> bool:
    """
    Checks if log files have already been consolidated for a model.
    
    Looks for the existence of consolidated log files to avoid redundant processing.
    
    Args:
        model_name (str): Model identifier
        files (list): List of filenames in the model's directory
    
    Returns:
        bool: True if all three consolidated log files exist
              ({model_name}-full-logs-train.parquet, -test.parquet, -test-on-train.parquet)
    """
    prefix = f"{model_name}-full-logs"
    if (f"{prefix}-train.parquet" in files) and (f"{prefix}-test.parquet" in files) and (f"{prefix}-test-on-train.parquet" in files):
        return True
    return False

def consolidate_folder_logs(model_name: str, save_dir: str = "./saved_models") -> None:
    """
    Consolidates per-epoch log files into single consolidated files for a model.
    
    Combines all epoch-specific log files (e.g., model-ep1-logs-train.parquet,
    model-ep2-logs-train.parquet) into consolidated files (model-full-logs-train.parquet)
    with epoch and batch indexing. Removes individual epoch files after consolidation.
    
    Args:
        model_name (str): Model identifier to consolidate logs for
        save_dir (str, optional): Base directory containing model folders. 
                                  Defaults to "./saved_models".
    
    Returns:
        None: Files are written to disk and epoch files are deleted
    """
    files = list(os.walk(f"{save_dir}/{model_name}"))[0][2]
    if check_consolidated(model_name, files):
        return

    suffixes = [
        "train",
        "test",
        "test-on-train"
    ]

    for suff in suffixes:
        master_file_name = f"{model_name}-full-logs-{suff}.parquet"
        patt = re.compile(
            f"{model_name}-ep(\\d+)-logs-{suff}\\.parquet"
        )
        matching_files = [match for file_name in files for match in re.finditer(patt, file_name)]
        if len(matching_files) == 0:
            continue

        concatted = pd.concat( # type: ignore
            [
                pd.read_parquet(f"{save_dir}/{model_name}/{match.string}")
                for match in matching_files
            ],
            keys = [int(match.group(1)) for match in matching_files],
            names = ("epoch", "batch_index") # type: ignore
        )
        concatted.to_parquet(f"{save_dir}/{model_name}/{master_file_name}")

        for match in matching_files:
            os.remove(f"{save_dir}/{model_name}/{match.string}")


def consolidate_all_logs(save_dir: str = "./saved_models") -> None:
    """
    Consolidates log files for all models in the save directory.
    
    Iterates through all model subdirectories and calls consolidate_folder_logs
    for each one. Used for batch processing after training multiple models.
    
    Args:
        save_dir (str, optional): Base directory containing model folders.
                                  Defaults to "./saved_models".
    
    Returns:
        None: All model logs are consolidated in-place
    """
    subdir_names = list(os.walk(save_dir))

    for model_name in subdir_names[0][1]:
        consolidate_folder_logs(model_name, save_dir)



### log parsing

def get_model_test_logs(model_name: str, save_dir: str = "./saved_models") -> pd.DataFrame:
    """
    Loads consolidated test logs for a specific model.
    
    Reads the consolidated test log parquet file created by consolidate_folder_logs.
    
    Args:
        model_name (str): Model identifier
        save_dir (str, optional): Base directory containing model folders.
                                  Defaults to "./saved_models".
    
    Returns:
        pd.DataFrame: Consolidated test logs with multi-index (epoch, batch_index)
                      and columns as defined in the batch log format documentation
    """
    return pd.read_parquet(f"{save_dir}/{model_name}/{model_name}-full-logs-test.parquet")

def analyze_test_frame(test_frame: pd.DataFrame) -> tuple[float, float, float, float, float]:
    """
    Analyzes test results by computing classification metrics.

    Extracts predictions and true labels from batch log format and computes
    standard binary classification metrics including loss functions and
    performance measures.

    Args:
        test_frame (pd.DataFrame): Test log DataFrame with 'batch_pred' and 'batch_true' columns
                                   containing model predictions and true labels as arrays

    Returns:
        tuple[float, float, float, float, float]: Classification metrics in order:
            (cross_entropy_loss, mse_loss, auc_score, accuracy, f1_score)
    """
    predictions = np.asarray(test_frame['batch_pred'].explode(), dtype=np.float32)
    true_vals = np.asarray(test_frame['batch_true'].explode(), dtype=np.float32)

    with torch.no_grad():
        cross_loss = nn.functional.binary_cross_entropy(torch.from_numpy(predictions), torch.from_numpy(true_vals), reduction='mean').item()
        mse_loss = nn.functional.mse_loss(torch.from_numpy(predictions), torch.from_numpy(true_vals), reduction='mean').item()
    auc = roc_auc_score(true_vals, predictions)
    acc = accuracy_score(true_vals, np.asarray(predictions >= 0.5, dtype = np.float32))
    f1 = f1_score(true_vals, np.asarray(predictions >= 0.5, dtype = np.float32))

    return cross_loss, mse_loss, auc, acc, f1


def analyze_best_epoch_from_logs(test_logs_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Analyzes consolidated test logs to find the best performing epoch for each metric.

    Computes test metrics for each epoch and identifies which epoch performed
    best for each of the five classification metrics. Uses inverted metrics
    (1 - metric) for AUC, accuracy, and F1 to find minimum values consistently.

    Args:
        test_logs_frame (pd.DataFrame): Consolidated test logs with multi-index (epoch, batch_index)
                                       from consolidate_folder_logs

    Returns:
        tuple[np.ndarray, np.ndarray]: (best_epoch_ids, best_epoch_stats) where:
            - best_epoch_ids: Array of epoch numbers (1-indexed) that achieved best performance
                             for each metric [cross_entropy, mse, auc, accuracy, f1]
            - best_epoch_stats: Array of the actual best metric values achieved
                               in the same order as the metrics
    """
    # list of stats by epoch
    epoch_stats = np.asarray([
        analyze_test_frame(test_logs_frame.loc[test_logs_frame.index.get_level_values(0) == ep]) for ep in (test_logs_frame.index.get_level_values(0).drop_duplicates())
    ])
    # note that this is 0-indexed, whereas our files and the spreadsheet itself 1-index our epochs. so we return the value plus one
    part_inv_epoch_stats = epoch_stats.copy()
    part_inv_epoch_stats[:, 2:] = 1 - part_inv_epoch_stats[:, 2:]

    best_epoch_id = np.argmin(part_inv_epoch_stats, axis=0)

    return best_epoch_id+1, epoch_stats[best_epoch_id]





def create_id_transformers(frame: pd.DataFrame) -> tuple[LabelEncoder, LabelEncoder, LabelEncoder]:
    """
    Creates LabelEncoder transformers for converting categorical IDs to numerical values.

    Builds three separate encoders for individual IDs, wave names, and question identifiers.
    Question IDs are created by concatenating variable_name and variable_label with "X@X".

    Args:
        frame (pd.DataFrame): Melted survey data with 'prim_key', 'wave_name',
                             'variable_name', and 'variable_label' columns

    Returns:
        tuple[LabelEncoder, LabelEncoder, LabelEncoder]: (individual_encoder, wave_encoder, question_encoder)
                                                        fitted to the data
    """
    i_transformer = LabelEncoder().fit(frame['prim_key'])
    w_transformer = LabelEncoder().fit(frame['wave_name'])
    q_transformer = LabelEncoder().fit(frame['variable_name'] + "X@X" + frame['variable_label'])

    return i_transformer, w_transformer, q_transformer

def save_transformers(i_transformer: LabelEncoder, w_transformer: LabelEncoder, q_transformer: LabelEncoder, base_path: str) -> None:
    """
    Saves fitted LabelEncoder transformers to disk as pickle files.

    Args:
        i_transformer (LabelEncoder): Fitted individual ID transformer
        w_transformer (LabelEncoder): Fitted wave name transformer
        q_transformer (LabelEncoder): Fitted question ID transformer
        base_path (str): Base path for files (will append suffixes: -individual.pkl, -wave.pkl, -question.pkl)

    Returns:
        None: Files are written to disk
    """
    pickle.dump(i_transformer, open(f"{base_path}-individual.pkl", "xb"))
    pickle.dump(w_transformer, open(f"{base_path}-wave.pkl", "xb"))
    pickle.dump(q_transformer, open(f"{base_path}-question.pkl", "xb"))

def load_transformers(base_path: str) -> tuple[LabelEncoder, LabelEncoder, LabelEncoder]:
    """
    Loads LabelEncoder transformers from pickle files.

    Args:
        base_path (str): Base path for files (expects suffixes: -individual.pkl, -wave.pkl, -question.pkl)

    Returns:
        tuple[LabelEncoder, LabelEncoder, LabelEncoder]: (individual_encoder, wave_encoder, question_encoder)
                                                        loaded from disk
    """
    i_transformer = pickle.load(open(f"{base_path}-individual.pkl", "rb"))
    w_transformer = pickle.load(open(f"{base_path}-wave.pkl", "rb"))
    q_transformer = pickle.load(open(f"{base_path}-question.pkl", "rb"))
    return i_transformer, w_transformer, q_transformer

def use_id_transformers_on_melt(frame: pd.DataFrame, i_transformer: LabelEncoder, w_transformer: LabelEncoder, q_transformer: LabelEncoder) -> pd.DataFrame:
    """
    Applies fitted LabelEncoder transformers to melted survey data.

    Converts categorical IDs to numerical values suitable for model input,
    adding new columns for the numerical IDs while preserving original data.

    Args:
        frame (pd.DataFrame): Melted survey data with categorical ID columns
        i_transformer (LabelEncoder): Fitted individual ID transformer
        w_transformer (LabelEncoder): Fitted wave name transformer
        q_transformer (LabelEncoder): Fitted question ID transformer

    Returns:
        pd.DataFrame: Copy of input frame with added numerical ID columns:
                     'indiv_id', 'wave_id', 'question_id'
    """
    new_frame = frame.copy()
    new_frame['indiv_id'] = i_transformer.transform(frame['prim_key'])
    new_frame['wave_id'] = w_transformer.transform(frame['wave_name'])
    new_frame['question_id'] = q_transformer.transform(frame['variable_name'] + "X@X" + frame['variable_label'])
    return new_frame

