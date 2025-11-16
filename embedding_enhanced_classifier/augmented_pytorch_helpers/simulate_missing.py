"""Missing data simulation patterns for training and evaluation.

This module provides functions to simulate different missing data patterns in
longitudinal survey data:
- MCAR (Missing Completely at Random): Random cell-level missingness
- Retrodiction: Wave-specific column-level missingness
- UOP (Unobserved Opinion Prediction): Full column removal across all waves

These simulation functions create realistic training scenarios where the model
learns to impute missing responses under different missingness mechanisms.
"""
from __future__ import annotations

import re
import torch
import random
import pandas as pd
import numpy as np
from random import randint
import sklearn.model_selection as model_selection

from .parse_sanitize_data import *
from .configs import *

WAVE_COL = "wave_name"
PART_ID_COL = "prim_key"


def gen_merged_train_frame(master_frame: pd.DataFrame, train_info: TrainDataInfo) -> pd.DataFrame:
    """
    Generates a training frame with missing data pattern based on configuration.

    Applies the specified missing data simulation pattern (MCAR or retrodiction)
    to create a training dataset with missing values that will be imputed.

    Args:
        master_frame (pd.DataFrame): Complete merged survey data in wide format
        train_info (TrainDataInfo): Tuple of (DataMode, proportion) specifying
                                   missing data pattern and proportion

    Returns:
        pd.DataFrame: Training frame with missing data pattern applied

    Raises:
        Exception: If an unimplemented DataMode is specified
    """
    mode = train_info[0]
    prop = train_info[1]

    if mode == DataMode.MCAR_MISSING:
        return remove_cell_proportion(
            master_frame,
            prop,
            list(set(master_frame.columns.to_list()) - set([PART_ID_COL, WAVE_COL]))
        )
    elif mode == DataMode.RETRO_MISSING:
        return merged_to_retrodiction_missing(master_frame, prop)
    else:
        raise Exception("Unimplemented training data type!")

def get_mask_complement(master_frame: pd.DataFrame, mask_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Given a master frame and a frame masked with NA, returns a frame with values 
    masked in the opposite manner.
    
    Args:
        master_frame (pd.DataFrame): Original dataframe in wide form
        mask_frame (pd.DataFrame): DataFrame with NA values indicating masking pattern
    
    Returns:
        pd.DataFrame: Frame with values masked opposite to mask_frame, intended 
                      for extracting test sets from a masked training set
    """
    new_frame = master_frame.copy()
    aff_cols = list(set(master_frame.columns.to_list()) - set([WAVE_COL, PART_ID_COL]))
    new_frame[aff_cols] = master_frame[aff_cols].mask(mask_frame[aff_cols].notna(), other = pd.NA, inplace = False)
    return new_frame



def remove_cell_proportion(master_frame: pd.DataFrame, proportion_missing: float, affected_cols: list[str]) -> pd.DataFrame:
    """
    Randomly removes cells from specified columns based on a proportion.
    
    Args:
        master_frame (pd.DataFrame): Original dataframe
        proportion_missing (float): Proportion of cells to mark as missing (0.0 to 1.0)
        affected_cols (list): List of column names to apply missing data simulation to
    
    Returns:
        pd.DataFrame: DataFrame with randomly missing cells in affected columns
    """
    missing_vals = np.random.uniform(size = master_frame[affected_cols].shape)
    
    missing_matters = np.where(master_frame[affected_cols].isna().to_numpy(), np.array([-1.0]), missing_vals).flatten()

    new_array = []
    for i in missing_matters:
        if i >= 0:
            new_array.append(i)
    new_array = np.array(new_array)
    threshold = np.percentile(new_array, proportion_missing * 100)

    missingness = missing_vals <= threshold
    named_missingness = pd.DataFrame(missingness, columns = affected_cols)
    copied_missingness = master_frame.isna().assign(**dict([(col, named_missingness[col]) for col in affected_cols]))

    masked = master_frame.mask(copied_missingness, other = pd.NA, inplace = False)
    return masked

def remove_column_proportion(master_frame: pd.DataFrame, proportion_missing: float, affected_cols: list[str]) -> pd.DataFrame:
    """
    Removes entire columns based on a target proportion of missing data.
    
    Selects columns to remove entirely until the cumulative size reaches
    the desired proportion of missing data.
    
    Args:
        master_frame (pd.DataFrame): Original dataframe
        proportion_missing (float): Target proportion of data to be missing (0.0 to 1.0)
        affected_cols (list): List of column names eligible for removal
    
    Returns:
        pd.DataFrame: DataFrame with selected columns completely removed (set to NA)
    """
    col_size_tuples = []
    for col in affected_cols:
        rows_with_col = master_frame.loc[master_frame[col].notna()]
        col_size_tuples.append(
            (col, len(rows_with_col))
        )
    
    column_frame = pd.DataFrame(col_size_tuples, columns = ["col_name", "size"])
    total_size = column_frame["size"].sum()
    desired_size = np.ceil(total_size * proportion_missing)

    shuffled_col_frame = column_frame.sample(frac = 1, replace = False, ignore_index = True, random_state = np.random.default_rng())
    sum_list = shuffled_col_frame["size"].cumsum().to_list()

    idx = 0
    while sum_list[idx] < desired_size:
        idx += 1
    
    selected_missing_cols = shuffled_col_frame.head(idx + 1)

    masked_frame = master_frame.copy()
    for info_row in selected_missing_cols.iterrows():
        masked_frame[info_row[1]["col_name"]] = pd.NA
    return masked_frame



def merged_to_mcar_missing(wave_data_merged: pd.DataFrame, proportion_missing: float = 0.5) -> pd.DataFrame:
    """
    Applies Missing Completely at Random (MCAR) pattern to merged wave data.
    
    Args:
        wave_data_merged (pd.DataFrame): Merged dataframe containing wave data
        proportion_missing (float, optional): Proportion of cells to make missing. Defaults to 0.5.
    
    Returns:
        pd.DataFrame: DataFrame with MCAR missing pattern applied to all columns 
                      except PART_ID_COL and WAVE_COL
    """
    affected_columns = list(set(wave_data_merged.columns.to_list()) - set([PART_ID_COL, WAVE_COL]))
    return remove_cell_proportion(wave_data_merged, proportion_missing, affected_columns)

def merged_to_retrodiction_missing(wave_data_merged: pd.DataFrame, proportion_missing: float = 0.5) -> pd.DataFrame:
    """
    Applies retrodiction missing pattern to merged wave data.
    
    Splits merged data by wave, applies column-wise missing pattern to each wave
    separately, then merges back together.
    
    Args:
        wave_data_merged (pd.DataFrame): Merged dataframe containing wave data
        proportion_missing (float, optional): Target proportion of missing data. Defaults to 0.5.
    
    Returns:
        pd.DataFrame: DataFrame with retrodiction missing pattern applied
    """
    wave_data_dict = split_merged_to_wave_dict(wave_data_merged)

    new_dict = {}
    for wave, frame in wave_data_dict.items():
        affected_columns = list(set(frame.columns.to_list()) - set([PART_ID_COL, WAVE_COL]))
        new_dict[wave] = remove_column_proportion(frame, proportion_missing, affected_columns)

    return wave_dict_to_merged(new_dict)

def merged_to_uop_missing(wave_data_merged: pd.DataFrame, proportion_missing: float = 0.5) -> pd.DataFrame:
    """
    Applies "unobserved opinion prediction" missing pattern to merged wave data.
    
    Removes entire columns from the merged dataset to achieve the target
    proportion of missing data.
    
    Args:
        wave_data_merged (pd.DataFrame): Merged dataframe containing wave data
        proportion_missing (float, optional): Target proportion of missing data. Defaults to 0.5.
    
    Returns:
        pd.DataFrame: DataFrame with UOP missing pattern applied (entire columns removed)
    """
    affected_columns = list(set(wave_data_merged.columns.to_list()) - set([PART_ID_COL, WAVE_COL]))
    return remove_column_proportion(wave_data_merged, proportion_missing, affected_columns)