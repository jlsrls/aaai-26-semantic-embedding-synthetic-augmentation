"""Data loading, sanitization, and transformation for survey data.

This module provides functions to load and process longitudinal survey data
including metadata, raw responses, and LLM-generated synthetic data. It handles
encoding issues, validates data integrity, and performs transformations needed
for neural network training (binarization, melting, merging across waves).
"""
from __future__ import annotations
from typing import Any, Union

import pandas as pd
import numpy as np
import os
import re
import math

def get_sane_dicts(
        metadata_dir: str = "./data_manip/metadata",
        raw_data_dir: str = "./data_manip/raw_data"
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Main entry point to load and sanitize both metadata and raw survey data.
    
    Loads metadata and raw data from specified directories, then performs
    second-pass sanitization by cross-referencing metadata with raw data.
    
    Args:
        metadata_dir (str, optional): Path to metadata CSV files. Defaults to "./data_manip/metadata".
        raw_data_dir (str, optional): Path to raw survey data CSV files. Defaults to "./data_manip/raw_data".
    
    Returns:
        tuple: (metadata_dict, data_dict) where each is a dictionary keyed by wave names
               containing DataFrames of sanitized metadata and raw data respectively
    """
    metadata_dict = get_sane_metadata(metadata_dir)
    data_dict = get_sane_raw_data(raw_data_dir)
    for wave in metadata_dict.keys():
        second_pass_sanitize_data(data_dict[wave], metadata_dict[wave])
    return metadata_dict, data_dict

def get_synth_compat_dicts(
        metadata_dir: str = "./data_manip/metadata",
        raw_data_dir: str = "./data_manip/raw_data",
        synth_data_dirs: dict[str, str] = {
            "gpt-4o": "./data_manip/synth_data/gen_data_gpt_4o",
            "gpt-5": "./data_manip/synth_data/gen_data_gpt_5",
            "sonnet-4": "./data_manip/synth_data/gen_data_sonnet_4"
        },
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    """
    Loads metadata, real data, and synthetic data with cross-compatibility.

    Similar to get_sane_dicts() but also loads LLM-generated synthetic data and
    ensures metadata compatibility across all data sources. Uses a multi-pass
    sanitization strategy: synthetic data informs metadata, which then sanitizes
    both real and synthetic data to ensure consistent column sets.

    Args:
        metadata_dir: Path to metadata CSV files. Defaults to "./data_manip/metadata".
        raw_data_dir: Path to raw survey data CSV files. Defaults to "./data_manip/raw_data".
        synth_data_dirs: Dictionary mapping model names to their synthetic data directories.
                        Defaults to GPT-4o, GPT-5, and Sonnet-4 directories.

    Returns:
        tuple: (metadata_dict, data_dict, synth_dicts) where:
               - metadata_dict: Dictionary of metadata DataFrames keyed by wave
               - data_dict: Dictionary of real data DataFrames keyed by wave
               - synth_dicts: Nested dictionary [model_name][wave] -> DataFrame of synthetic data
    """
    metadata_dict = get_sane_metadata(metadata_dir)
    data_dict = get_sane_raw_data(raw_data_dir)
    syn_dicts = dict([(k, get_sane_synth_data(synth_data_dirs[k], k)) for k in synth_data_dirs.keys()])
    for wave in metadata_dict.keys():
        for syn_dict in syn_dicts.keys():
            first_pass_inform_metadata(syn_dicts[syn_dict][wave], metadata_dict[wave])
        
        second_pass_sanitize_data(data_dict[wave], metadata_dict[wave], fill_possible=True)

        for syn_dict in syn_dicts.keys():
            second_pass_sanitize_data(syn_dicts[syn_dict][wave], metadata_dict[wave], fill_possible=False)
    
    return metadata_dict, data_dict, syn_dicts


def get_sane_metadata(metadata_dir: str) -> dict[str, pd.DataFrame]:
    """
    Loads and sanitizes metadata CSV files from the specified directory.
    
    Processes metadata files that describe survey questions, response types,
    missing value codes, and binarization rules. Handles cp1252 encoding
    (for smartquotes) and converts to UTF-8.
    
    Args:
        metadata_dir (str): Path to directory containing metadata CSV files
    
    Returns:
        dict: Dictionary keyed by wave names (e.g., 'wave1', 'wave2') containing
              DataFrames with metadata for each survey wave
    
    Raises:
        Exception: If duplicate variable names are found within a wave
    """
    dirinfo = next(os.walk(metadata_dir))
    csvs = dirinfo[2]

    wave_frame_dict = {}
    for name in csvs:
        if name == "Labels Glossary.csv":
            continue
        
        # the source of the encoding errors: the survey questions use smartquotes and a non-utf encoding.
        # cp1252 should be the correct decoding (thanks google). later we'll convert this to utf-8 and remove all the weird apostrophes
        # later we'll convert everything to ascii
        data = pd.read_csv(f"{metadata_dir}/{name}", encoding="cp1252")

        # remove duplicate rows, then check for more vexing duplicates
        data = data.drop_duplicates()
        if len(data) != len(data["variable_name"].drop_duplicates()):
            raise Exception(f"Questions in wave {name} are not uniquely identified by variable_name: {len(data)} vs. {len(data[["variable_name"]].drop_duplicates())}")

        wave_str = "wave" + next(re.finditer("\\d+[ab]?", name)).group(0)
        wave_frame_dict[wave_str] = data

    return wave_frame_dict

def get_sane_raw_data(raw_data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Loads and validates raw survey data CSV files from the specified directory.
    
    Performs basic validation to ensure data integrity including checking
    for duplicate primary keys and duplicate column names.
    
    Args:
        raw_data_dir (str): Path to directory containing raw survey data CSV files
    
    Returns:
        dict: Dictionary keyed by wave names (e.g., 'wave1', 'wave2') containing
              DataFrames with raw survey responses for each wave
    
    Raises:
        Exception: If duplicate primary keys or column names are found
    """
    dirinfo = next(os.walk(raw_data_dir))
    csvs = dirinfo[2]

    wave_frame_dict = {}
    for name in csvs:
        data = pd.read_csv(f"{raw_data_dir}/{name}")
        if len(data) != len(data["prim_key"].drop_duplicates()):
            raise Exception(f"Duplicate primary-key responses in {name}!")
        if len(data.columns) != len(data.columns.drop_duplicates()):
            raise Exception(f"Duplicate columns in {name}!")

        wave_str = next(re.finditer("wave\\d+[ab]?", name)).group(0)
        wave_frame_dict[wave_str] = data

    return wave_frame_dict

def get_sane_synth_data(synth_data_dir: str, model_name: str) -> dict[str, pd.DataFrame]:
    """
    Loads and validates LLM-generated synthetic survey data from the specified directory.

    Processes synthetic response data generated by language models, removes extraneous
    columns, creates unique primary keys, and validates data integrity.

    Args:
        synth_data_dir: Path to directory containing synthetic data CSV files
        model_name: Name of the LLM that generated the data (e.g., 'gpt-4o', 'sonnet-4')

    Returns:
        dict: Dictionary keyed by wave names (e.g., 'wave1', 'wave2') containing
              DataFrames with synthetic survey responses for each wave

    Raises:
        Exception: If duplicate primary keys or column names are found
    """
    dirinfo = next(os.walk(synth_data_dir))

    csvs = dirinfo[2]

    wave_frame_dict = {}
    for name in csvs:
        data = pd.read_csv(f"{synth_data_dir}/{name}")
        data = data[list(set(data.columns.to_list()) - set(["Unnamed: 0", "prev_cluster", "cluster"]))]

        wave_str = next(re.finditer("gen_data_(wave\\d+[ab]?)", name)).group(1)

        data["prim_key"] = data.index.astype('string') + f"-{wave_str}-{model_name}"
        if len(data) != len(data["prim_key"].drop_duplicates()):
            raise Exception(f"Duplicate primary-key responses in {name}!")
        if len(data.columns) != len(data.columns.drop_duplicates()):
            raise Exception(f"Duplicate columns in {name}!")

        wave_frame_dict[wave_str] = data

    return wave_frame_dict

def first_pass_inform_metadata(synth_data: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """
    Prunes metadata to match columns present in synthetic data.

    Part of multi-pass sanitization: removes metadata entries for questions that
    aren't present in the synthetic data, ensuring compatibility between all data
    sources. Modifies metadata DataFrame in-place.

    Args:
        synth_data: DataFrame containing LLM-generated synthetic survey responses
        metadata: DataFrame containing question metadata (modified in-place)

    Returns:
        None: Modifies metadata DataFrame in-place
    """
    synth_cols = set(synth_data.columns.to_list()) - set(["prim_key"])
    # drop rows in the metadata which are not columns in the synthetic data
    metadata.drop(metadata[~metadata['variable_name'].isin(synth_cols)].index, inplace=True)


def second_pass_sanitize_data(raw_data: pd.DataFrame, metadata: pd.DataFrame, fill_possible = True) -> None:
    """
    Performs cross-referenced sanitization of raw data using metadata.
    
    Removes columns not described in metadata, applies missing value masks
    based on metadata specifications, and adds a 'possible_values' column
    to metadata containing the unique non-missing values for each variable.
    
    Args:
        raw_data (pd.DataFrame): Raw survey data (modified in-place)
        metadata (pd.DataFrame): Metadata describing variables (modified in-place)
    
    Returns:
        None: Modifies input DataFrames in-place
    """
    data_cols = metadata["variable_name"].drop_duplicates().to_list()
    drop_cols = list(set(raw_data.columns.to_list()) - (set(data_cols) | set(["prim_key"])))
    raw_data.drop(columns=drop_cols, inplace = True)

    unique_vals_running_list = []
    # now we need to do some operations on the columns by cross-referencing between the metadata and raw data
    for col_ind in range(len(metadata)):        
        col_name = metadata.iloc[col_ind]["variable_name"]
        col_missing_val = metadata.iloc[col_ind]["value_for_missing"]

        # first we mask out cells that correspond to missing answers
        if not pd.isna(col_missing_val):
            raw_data[col_name] = raw_data[col_name].mask((raw_data[col_name] == col_missing_val), other = pd.NA)

        col_vals = raw_data[col_name]
        unique_vals = np.array(col_vals.dropna().drop_duplicates().to_list())
        unique_vals_running_list.append(unique_vals)
    
    if fill_possible:
        metadata.insert(len(metadata.columns), "possible_values", unique_vals_running_list)
    

def wave_dict_to_merged(wave_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Converts a dictionary of wave-specific DataFrames into a single merged DataFrame.
    
    Adds a 'wave_name' column to identify the source wave for each row,
    then concatenates all waves into a single long-format DataFrame.
    
    Args:
        wave_dict (dict): Dictionary keyed by wave names containing DataFrames
    
    Returns:
        pd.DataFrame: Merged DataFrame with all waves combined and 'wave_name' column added
    """
    exp_waves = []
    for wave in wave_dict.keys():
        frame = wave_dict[wave]

        frame_copy = frame.copy()
        frame_copy["wave_name"] = wave
        exp_waves.append(frame_copy)

    merged = pd.concat(exp_waves)
    return merged

def split_merged_to_wave_dict(merged_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Splits a merged DataFrame back into wave-specific DataFrames.
    
    Groups by 'wave_name' column and creates separate DataFrames for each wave,
    removing the wave_name column and any completely empty columns.
    
    Args:
        merged_frame (pd.DataFrame): Merged DataFrame with 'wave_name' column
    
    Returns:
        dict: Dictionary keyed by wave names containing wave-specific DataFrames
    """
    wave_tuples = {}
    for a,b in merged_frame.groupby(merged_frame["wave_name"]):
        cols = list(set(b.columns.to_list()) - set(["wave_name"]))
        wave_tuples[a] = b[cols].dropna(how='all', axis='columns')
    return wave_tuples



def melt_merged(merged_data: pd.DataFrame, metadata_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Melts merged survey data from wide format to long format with metadata joined.
    
    Converts wide-format survey data to long format where each row represents
    one question response, then joins with metadata to include variable labels.
    
    Args:
        merged_data (pd.DataFrame): Wide-format merged survey data
        metadata_dict (dict): Dictionary of metadata DataFrames keyed by wave names
    
    Returns:
        pd.DataFrame: Long-format DataFrame with columns: prim_key, wave_name,
                      variable_name, value, variable_label
    """
    data_dict = split_merged_to_wave_dict(merged_data)
    melted_dict = dict(
        (w_n, melt_filter_join_frame(data_dict[w_n], metadata_dict[w_n])) for w_n in data_dict.keys()
    )
    return wave_dict_to_merged(melted_dict)

def melt_filter_join_frame(frame: pd.DataFrame, metadata: pd.DataFrame, id_vars: str = "prim_key") -> pd.DataFrame:
    """
    Melts a single wave's data and joins with its metadata.
    
    Converts from wide to long format, filters out missing values,
    and joins with metadata to include variable labels.
    
    Args:
        frame (pd.DataFrame): Wide-format survey data for one wave
        metadata (pd.DataFrame): Metadata for this wave
        id_vars (str, optional): Column to use as identifier. Defaults to "prim_key".
    
    Returns:
        pd.DataFrame: Long-format DataFrame with variable labels joined
    """
    unfiltered = pd.melt(frame, id_vars=id_vars, var_name="variable_name", value_name="value")
    filtered = unfiltered.loc[pd.notnull(unfiltered['value'])]
    return pd.merge(filtered, metadata[['variable_name', 'variable_label']], how="inner")



def binarize_dict(data_dict: dict[str, pd.DataFrame], metadata_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Binarizes all waves in a data dictionary using corresponding metadata.
    
    Applies binarization to each wave's data based on the 'yes_list' specifications
    in the metadata, converting multi-valued survey responses to binary (0/1) values.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by wave names
        metadata_dict (dict): Dictionary of metadata DataFrames keyed by wave names
    
    Returns:
        dict: Dictionary of binarized DataFrames keyed by wave names
    """
    return dict([
        (
            w_n, get_binarized_wave(data_dict[w_n], metadata_dict[w_n])
        ) for w_n in data_dict.keys()
    ])

def get_binarized_wave(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Binarizes survey responses for a single wave based on metadata specifications.
    
    Converts survey responses to binary values (0/1) based on whether the response
    appears in the 'yes_list' defined in the metadata. Preserves prim_key and
    wave_name columns unchanged.
    
    Args:
        data (pd.DataFrame): Survey data for one wave
        metadata (pd.DataFrame): Metadata describing binarization rules
    
    Returns:
        pd.DataFrame: Binarized version of the input data with same structure
                      but binary response values
    """
    binarized_col_dict = {}
    binarized_col_dict["prim_key"] = data["prim_key"]
    if "wave_name" in data.columns.to_list():
        binarized_col_dict["wave_name"] = data["wave_name"]

    # binarized_col_dict["weight"] = data["weight"].to_list()

    for col_name in data.columns:
        col_vals = data[col_name]
        meta_row = metadata.loc[metadata["variable_name"] == col_name]

        if len(meta_row) == 0:
            continue

        # for one-hot-encoded features, we don't consider any submissions "missing"
        # this isn't quite correct -- if somebody submits no response in any column, it's still missing --
        # but this should do well enough for now.
        nan_override = int(meta_row["one_hot"].to_list()[0]) != 0

        yes_list = process_yes_list(meta_row["yes_list"].to_list()[0])

        semibin_col = [response_in_yes_list(x, yes_list, nan_override) for x in col_vals]
        binarized_col_dict[col_name] = semibin_col

    df = pd.DataFrame(binarized_col_dict)
    return df

def response_in_yes_list(resp: Any, y_list: list[Any], na_override: bool = False) -> Union[int, float]:
    """
    Determines if a survey response should be coded as 1 (yes) or 0 (no/missing).
    
    Checks if a response value appears in the 'yes_list' for binarization.
    Handles missing values according to na_override setting and performs
    type conversion as needed.
    
    Args:
        resp: Survey response value (can be various types or missing)
        y_list (list): List of values that should be coded as 1 ("yes")
        na_override (bool, optional): If True, missing values become 0 instead of NA.
                                    Defaults to False.
    
    Returns:
        np.int64 or pd.NA: 1 if response is in yes_list, 0 if not in yes_list
                          (or if missing and na_override=True), pd.NA if missing
                          and na_override=False
    
    Raises:
        Exception: If response type doesn't match expected type from yes_list
    """
    if (resp is pd.NA) or (isinstance(resp, str) and resp == "") or ((not isinstance(resp, str)) and (math.isnan(resp) or np.isnan(resp))):
        if na_override:
            return np.int64(0)
        else:
            return pd.NA
    
    resp_type = type(resp)
    list_type = type(y_list[0])
    if resp_type != list_type:
        if isinstance(y_list[0], np.int64):
            resp = np.int64(resp)
        else:
            raise Exception(f"Cell value mismatch: {resp} has type {resp_type} when {list_type} expected!")

    return np.int64(1) if resp in y_list else np.int64(0)

def process_yes_list(_str: str) -> list[str]:
    """
    Parses 'yes_list' strings from metadata into lists of values for binarization.
    
    Handles various formats:
    - Single numbers: "1" -> [1]
    - Comma-separated: "1,2" -> [1, 2] 
    - Ranges: "1 to 3" -> [1, 2, 3]
    - Special case: "a to b" -> ["a", "b"]
    
    Args:
        _str (str): String specification from metadata 'yes_list' column
    
    Returns:
        list: List of values that should be coded as 1 in binarization
    
    Raises:
        Exception: If string format is not recognized
    """
    nums = re.findall("\\d+", _str)
    if len(nums) == 2:
        n_list = [np.int64(i) for i in nums]
        if "," in _str:
            return n_list
        elif "to" in _str:
            return [np.int64(i) for i in list(range(min(n_list), max(n_list)+1))]
        else:
            raise Exception(f"Unknown yes_list: {_str}")
    elif len(nums) == 1:
        return [np.int64(nums[0])]
    elif "a to b" in _str:
        return ["a", "b"]
    else:
        raise Exception(f"Unknown yes_list: {_str}")
