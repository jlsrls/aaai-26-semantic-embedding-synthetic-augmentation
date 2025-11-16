"""Training pipeline orchestration and epoch management.

This module contains the main training loop implementation including:
- Epoch-level training and evaluation functions
- Model initialization with proper embedding configuration
- Data simulation and train/test split management
- Integration of synthetic data (when configured)
- Model checkpointing and logging to disk
"""
from __future__ import annotations
from typing import Any, Callable
import math

import torch
import pandas as pd
import numpy as np
import os
import torch
from tqdm.auto import tqdm
import torch.nn as nn

from .helpers import *
from .configs import TrainConfig, ImputeConfig, ImputeExpConfig, EmbeddingMode, EpochMode
from .parse_sanitize_data import *
from .simulate_missing import *
from .embeddings import *
from .impute_model import *


def train_epoch(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        num_steps: int = -1,
        model_name: str = "",
        device: str = "cpu",
    ) -> pd.DataFrame:
    """
    Trains the model for one epoch (according to the DataLoader) and returns batch-level training logs.
    
    Performs standard PyTorch training loop with gradient updates and scheduler steps.
    Collects predictions and true labels for each batch to create a training log DataFrame.
    
    Args:
        dataloader (torch.utils.data.DataLoader): Training data loader
        model (torch.nn.Module): Model to train
        loss_fn: Loss function (typically BCELoss for binary classification)
        optimizer: PyTorch optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        num_steps: The number of training steps to run for. May end a training epoch "early". Ignored if -1.
        model_name (str, optional): Name for progress bar description. Defaults to "".
        device (str, optional): Device to run on ('cpu' or 'cuda'). Defaults to "cpu".
    
    Returns:
        pd.DataFrame: Training log with columns ['batch_size', 'batch_pred', 'batch_true']
                      Each row represents one batch's results.
    """
    model.train()

    batch_data = []

    pbar = tqdm(total=len(dataloader), desc=f"{model_name} train epoch progress")
    for step, (X, y) in enumerate(dataloader):
        if num_steps != -1 and step >= num_steps:
            break 

        # X = X.to(device)
        # y = y.to(device)

        optimizer.zero_grad()

        pred = model(X).flatten()
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            batch_data.append({
                'batch_size': len(X),
                'batch_pred': pred.cpu().numpy(),
                'batch_true': y.cpu().numpy(),
            })
        pbar.update(1)


    pbar.close()

    epoch_frame = pd.DataFrame(batch_data)
    return epoch_frame


def test_dataloader(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, model_name: str = "", device: str = "cpu") -> pd.DataFrame:
    """
    Evaluates the model on a dataset and returns detailed batch-level test logs.
    
    Runs model in evaluation mode without gradient computation. Collects predictions,
    true labels, and input feature data for comprehensive testing analysis.
    
    Args:
        dataloader (torch.utils.data.DataLoader): Test/validation data loader
        model (torch.nn.Module): Model to evaluate
        model_name (str, optional): Name for progress bar description. Defaults to "".
        device (str, optional): Device to run on ('cpu' or 'cuda'). Defaults to "cpu".
    
    Returns:
        pd.DataFrame: Test log with columns ['batch_size', 'batch_data_indiv', 
                      'batch_data_year', 'batch_data_question', 'batch_pred', 'batch_true']
                      Each row represents one batch's results with input feature breakdown.
    """
    model.eval()

    test_data = []

    pbar = tqdm(total=len(dataloader), desc=f"{model_name} test progress")
    with torch.no_grad():
        for X, y in dataloader:
            # X = X.to(device)
            # y = y.to(device)
            # onehot = nn.functional.one_hot(y, num_classes = len(class_vals)).float()
            # onehot = onehot.to(device)

            pred = model(X).flatten()

            raw_data = X.cpu().numpy()
            test_data.append({
                'batch_size': len(X),
                'batch_data_indiv': raw_data[:, 0],
                'batch_data_year': raw_data[:, 1],
                'batch_data_question': raw_data[:, 2],
                'batch_pred': pred.cpu().numpy(),
                'batch_true': y.cpu().numpy()
            })
            pbar.update(1)
    pbar.close()


    epoch_frame = pd.DataFrame(test_data)
    return epoch_frame




def create_and_train(
        train_conf: TrainConfig,
        metadata_dict: dict[str, pd.DataFrame],
        bin_data_dict: dict[str, pd.DataFrame],
        synth_data_dicts: dict[str, dict[str, pd.DataFrame]] | None,
        device: str = "cpu"
    ) -> tuple[nn.Module, tuple[float, float, float, float, float]]:
    """
    Creates and trains an imputation model based on the provided configuration.

    Orchestrates the complete training pipeline including data preparation,
    model initialization, training loop, and results saving. Supports different
    missing data patterns (MCAR, retrodiction) and embedding types.

    Args:
        train_conf (TrainConfig): Complete training configuration including model type,
                                 hyperparameters, optimizer settings, and data specs
        metadata_dict (dict[str, pd.DataFrame]): Metadata for each wave describing
                                               survey questions and binarization rules
        bin_data_dict (dict[str, pd.DataFrame]): Binarized survey data for each wave
        device (str, optional): Device for training ('cpu' or 'cuda'). Defaults to "cpu".

    Returns:
        tuple: (trained_model, test_metrics) where:
            - trained_model: Final trained nn.Module
            - test_metrics: Tuple of (cross_entropy_loss, mse_loss, auc, accuracy, f1_score)
                           from the final epoch's test evaluation
    """

    # Step 1: Prepare base data and create train/test split
    # Merge all waves into wide format (one row per individual per wave)
    base_merged = wave_dict_to_merged(bin_data_dict)

    # Apply missing data pattern (MCAR or retrodiction) to create training set
    base_train_merged = gen_merged_train_frame(base_merged, train_conf["train_data_info"])

    # Test set is the complement: cells that are NOT missing in training
    # This ensures no leakage between train and test
    test_merged = get_mask_complement(base_merged, base_train_merged)

    # Step 2: Optionally augment with LLM-generated synthetic data
    syn_mode = train_conf["synth_data_mode"]
    if (synth_data_dicts != None) and (syn_mode != SynthDataMode.SYNTH_DATA_NONE_COMPATIBLE):
        # Select which synthetic data source(s) to use
        if syn_mode == SynthDataMode.SYNTH_DATA_GPT_4O:
            _synth_merged = wave_dict_to_merged(synth_data_dicts["gpt-4o"])
        elif syn_mode == SynthDataMode.SYNTH_DATA_GPT_5:
            _synth_merged = wave_dict_to_merged(synth_data_dicts["gpt-5"])
        elif syn_mode == SynthDataMode.SYNTH_DATA_SONNET_4:
            _synth_merged = wave_dict_to_merged(synth_data_dicts["sonnet-4"])
        else:  # SYNTH_DATA_ALL: concatenate all sources
            _synth_merged = pd.concat([
                wave_dict_to_merged(synth_data_dicts["gpt-4o"]),
                wave_dict_to_merged(synth_data_dicts["gpt-5"]),
                wave_dict_to_merged(synth_data_dicts["sonnet-4"])
            ], ignore_index = True)

        # Optionally apply same missing pattern to synthetic data
        # (if False, synthetic data is fully observed)
        if train_conf["apply_missing_to_synth_data"]:
            synth_merged = gen_merged_train_frame(_synth_merged, train_conf["train_data_info"])
        else:
            synth_merged = _synth_merged

        # Combine real and synthetic data for training
        train_merged = pd.concat([base_train_merged, synth_merged], ignore_index=True)
        master_merged = pd.concat([base_merged, synth_merged], ignore_index=True)
    else:
        # No synthetic data: use only real data
        train_merged = base_train_merged
        master_merged = base_merged

    # Step 3: Transform from wide to long format (one row per response)
    # This creates (individual, wave, question, value) tuples
    master_melt = melt_merged(master_merged, metadata_dict)
    train_melt = melt_merged(train_merged, metadata_dict)
    test_melt = melt_merged(test_merged, metadata_dict)

    # Step 4: Create label encoders to convert categorical IDs to integer indices
    # These map: individual names -> [0, n_indiv), wave names -> [0, n_waves), etc.
    indiv_id_transform, wave_id_transform, question_id_transform = create_id_transformers(master_melt)

    # Step 5: Apply transformers to convert all categorical IDs to integers
    # Final format: DataFrame with integer columns [indiv_id, wave_id, question_id, value]
    numerical_master = use_id_transformers_on_melt(
        master_melt,
        indiv_id_transform,
        wave_id_transform,
        question_id_transform
    )
    numerical_train = use_id_transformers_on_melt(
        train_melt,
        indiv_id_transform,
        wave_id_transform,
        question_id_transform
    )
    numerical_test = use_id_transformers_on_melt(
        test_melt,
        indiv_id_transform,
        wave_id_transform,
        question_id_transform
    )

    # Step 6: Determine vocabulary sizes from data
    # These define the embedding table dimensions
    n_indiv = numerical_master['indiv_id'].nunique()
    n_years = numerical_master['wave_id'].nunique()
    n_questions = numerical_master['question_id'].nunique()

    # Step 7: Build complete model configuration
    # Combine experiment config with runtime-computed values (vocab sizes)
    model_conf = ImputeConfig(
        n_indiv = n_indiv,
        n_years = n_years,
        n_questions = n_questions,
        **train_conf["model_config"]
    )

    # If using pre-trained embeddings (e.g., OpenAI), generate them now
    # This fetches or loads cached embeddings for all questions
    if model_conf["q_embedding_type"] == EmbeddingMode.PRESET_FROZEN:
        model_conf["q_embedding_values"] = get_embeddings(
            numerical_master,
            embedding_size = model_conf["q_embedding_size"]
        )

    # Step 8: Initialize model, optimizer, scheduler, and loss function
    model = train_conf["model_type"](model_conf)

    optimizer = train_conf["opt_type"](
        model.parameters(),
        **(train_conf["opt_kwargs"])
    )
    scheduler = train_conf["sched_type"](
        optimizer,
        **(train_conf["sched_kwargs"])
    )
    loss_func = nn.BCELoss()

    # Step 9: Create save directory and persist configurations
    # Save config, initial weights, data splits, and ID transformers
    model_name = train_conf["model_name"]
    save_dir = f"./saved_models/{model_name}"
    os.makedirs(save_dir)
    with open(f"{save_dir}/{model_name}-config.txt", 'x') as configfile:
        configfile.write(repr(train_conf) + "\n")
        configfile.write(repr(model_conf))

    torch.save(model, f"{save_dir}/{model_name}-ep0-model-weights.pt")
    master_merged.to_parquet(f"{save_dir}/{model_name}-data-train.parquet", index=False)
    test_merged.to_parquet(f"{save_dir}/{model_name}-data-test.parquet", index=False)
    save_transformers(
        indiv_id_transform,
        wave_id_transform,
        question_id_transform,
        f"{save_dir}/{model_name}-transformer"
    )

    # Move model to GPU (if available)
    model.to(device)

    # Step 10: Create PyTorch DataLoaders for efficient batching
    batch_size = train_conf["batch_size"]
    train_size = len(numerical_train)
    steps_per_epoch = math.ceil(train_size / batch_size)

    train_dl = frame_to_dataloader(numerical_train, batch_size=batch_size, shuffle=True)
    test_dl = frame_to_dataloader(numerical_test, batch_size=len(numerical_test), shuffle=False)


    epoch_mode, tr_len = train_conf['epoch_info']
    if epoch_mode == EpochMode.FIXED_EPOCHS:
        rem_steps: int = tr_len * steps_per_epoch
    else:
        rem_steps: int = tr_len


    # epoch id. used for naming, but does not control the loop in any way; rem_steps does that.
    e_id = 0
    while rem_steps > 0:
        e_id += 1
        # if synth_data_dict != None:
            # model_epoch_name = f"{model_name}-S-ep{str(e_id)}"
        # else:
        model_epoch_name = f"{model_name}-ep{str(e_id)}"

        el_steps = min(rem_steps, steps_per_epoch)
        train_epoch_frame = train_epoch(
            train_dl,
            model,
            loss_func,
            optimizer,
            scheduler,
            num_steps=-1 if el_steps == steps_per_epoch else el_steps,
            model_name=model_epoch_name,
            device=device
        )
        rem_steps -= el_steps

        torch.save(model, f"{save_dir}/{model_epoch_name}-model-weights.pt")
        train_epoch_frame.to_parquet(f"{save_dir}/{model_epoch_name}-logs-train.parquet")


        if ((e_id % train_conf["epochs_per_assess"]) == 0) or (rem_steps <= 0):
            test_epoch_frame = test_dataloader(
                test_dl,
                model,
                model_name=model_epoch_name,
                device=device
            )
            test_on_train_epoch_frame = test_dataloader(
                train_dl,
                model,
                model_name=model_epoch_name,
                device=device
            )
            test_epoch_frame.to_parquet(f"{save_dir}/{model_epoch_name}-logs-test.parquet")
            test_on_train_epoch_frame.to_parquet(f"{save_dir}/{model_epoch_name}-logs-test-on-train.parquet")

            # we smuggle these out of the loop for the return value
            test_vals = analyze_test_frame(test_epoch_frame)
            print(f"{model_epoch_name} test values:          {str(test_vals)}")
            print(f"{model_epoch_name} test-on-train values: {str(analyze_test_frame(test_on_train_epoch_frame))}")
        print(f"{model_epoch_name} training average:     {str(analyze_test_frame(train_epoch_frame))}")

    print("training complete")
    return (model, test_vals) # type: ignore
