"""Main training script for embedding-enhanced survey response imputation.

This script orchestrates the complete training pipeline for Deep Cross Network
models that predict missing binary survey responses. It configures multiple
training experiments, loads data, generates OpenAI embeddings (if needed), trains
models, and analyzes results. All training configurations are defined as TrainConfig
TypedDicts within the script.

Usage:
    python new_imp.py

Requirements:
    - OPENAI_API_KEY environment variable must be set
    - CUDA-capable GPU (configured for device="cuda")
    - Survey data in ./data_manip/ directory structure
"""
from __future__ import annotations

import pandas as pd
import random
import os
import numpy as np
from tqdm.auto import tqdm
from typing import Any, Optional
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from openai import OpenAI

from augmented_pytorch_helpers.helpers import *
from augmented_pytorch_helpers.impute_model import *
from augmented_pytorch_helpers.training import *
from augmented_pytorch_helpers.configs import *
from augmented_pytorch_helpers.parse_sanitize_data import *

import augmented_pytorch_helpers.embeddings


device: str = "cuda"
tqdm.pandas()


if __name__ == "__main__":
    # Initialize OpenAI client for generating embeddings
    # API key should be set in OPENAI_API_KEY environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it or create a .env file.")

    augmented_pytorch_helpers.embeddings.client = OpenAI(
        api_key = openai_api_key
    )
    
    bin_synth_dict = None
    

    full_metadata_dict, full_data_dict = get_sane_dicts()
    full_bin_data_dict = binarize_dict(full_data_dict, full_metadata_dict)

    # this is dumb and redundant, but it runs fast enough, so it doesn't really matter. looks hacky though
    synth_data_dicts = {}
    syn_comp_metadata_dict, syn_comp_data_dict, synth_data_dicts = get_synth_compat_dicts()

    syn_comp_bin_data_dict = binarize_dict(syn_comp_data_dict, syn_comp_metadata_dict)


    run_param_list: list[TrainConfig] = [
        TrainConfig(
            model_name = get_model_name(),
            train_data_info = (DataMode.MCAR_MISSING, 0.5),
            synth_data_mode = SynthDataMode.SYNTH_DATA_NONE_INCOMPATIBLE,
            apply_missing_to_synth_data = False,
            
            model_type = DCNImputeModelAdv,
            model_config = ImputeExpConfig(
                n_dim = 150,
                depth = 6,
                dropout = 0.25,
                q_embedding_type = EmbeddingMode.PRESET_FROZEN,
                q_embedding_size = 1536,
            ),

            epoch_info = (EpochMode.FIXED_STEPS, 92680),
            epochs_per_assess = 10,
            batch_size = 128,
            
            opt_type = optim.AdamW,
            opt_kwargs = {"lr": 7e-5, "weight_decay": 1/100},
            sched_type = optim.lr_scheduler.CosineAnnealingWarmRestarts,
            sched_kwargs = {"T_0": 23175}
        ),
        TrainConfig(
            model_name = get_model_name(),
            train_data_info = (DataMode.MCAR_MISSING, 0.5),
            synth_data_mode = SynthDataMode.SYNTH_DATA_NONE_COMPATIBLE,
            apply_missing_to_synth_data = False,
            
            model_type = DCNImputeModelAdv,
            model_config = ImputeExpConfig(
                n_dim = 150,
                depth = 6,
                dropout = 0.25,
                q_embedding_type = EmbeddingMode.PRESET_FROZEN,
                q_embedding_size = 1536,
            ),

            epoch_info = (EpochMode.FIXED_STEPS, 92680),
            epochs_per_assess = 10,
            batch_size = 128,
            
            opt_type = optim.AdamW,
            opt_kwargs = {"lr": 7e-5, "weight_decay": 1/100},
            sched_type = optim.lr_scheduler.CosineAnnealingWarmRestarts,
            sched_kwargs = {"T_0": 23175}
        ),
        TrainConfig(
            model_name = get_model_name(),
            train_data_info = (DataMode.MCAR_MISSING, 0.5),
            synth_data_mode = SynthDataMode.SYNTH_DATA_SONNET_4,
            apply_missing_to_synth_data = False,
            
            model_type = DCNImputeModelAdv,
            model_config = ImputeExpConfig(
                n_dim = 150,
                depth = 6,
                dropout = 0.25,
                q_embedding_type = EmbeddingMode.PRESET_FROZEN,
                q_embedding_size = 1536,
            ),

            epoch_info = (EpochMode.FIXED_STEPS, 92680),
            epochs_per_assess = 10,
            batch_size = 128,
            
            opt_type = optim.AdamW,
            opt_kwargs = {"lr": 7e-5, "weight_decay": 1/100},
            sched_type = optim.lr_scheduler.CosineAnnealingWarmRestarts,
            sched_kwargs = {"T_0": 23175}
        ),
    ]

    # Set random seeds for reproducible results
    os.environ['PYTHONHASHSEED'] = str(413)
    random.seed(413)
    np.random.seed(413)
    torch.manual_seed(413)

    # Train each model configuration
    for i in range(len(run_param_list)):
        params = run_param_list[i]

        print(f"running experiment with dataset {params["train_data_info"]}, synth {params["synth_data_mode"]}, name: {params["model_name"]}")

        if params["synth_data_mode"] == SynthDataMode.SYNTH_DATA_NONE_INCOMPATIBLE:
            create_and_train(
                params,
                full_metadata_dict,
                full_bin_data_dict,
                None,
                device=device
            )
        else:
            create_and_train(
                params,
                syn_comp_metadata_dict,
                syn_comp_bin_data_dict,
                synth_data_dicts,
                device=device
            )

    # Consolidate training logs from all models
    print("consolidating logs")
    consolidate_all_logs()
    print("consolidation finished")
    
    # Analyze and display results for each trained model
    for name in [x["model_name"] for x in run_param_list]:
        log = get_model_test_logs(name)
        results = analyze_best_epoch_from_logs(log)
        print(f"{name}: {results[0]}\n{results[1]}")
