"""Configuration types and enums for neural network training.

This module defines TypedDict configurations and Enum types used throughout
the embedding-enhanced classifier system. It provides type-safe configuration
for model architecture, training parameters, and data processing modes.
"""
from __future__ import annotations
from typing import Any, TypedDict, NotRequired, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from enum import Enum, EnumType

from .impute_model import *

class EmbeddingMode(Enum):
    """Question embedding initialization strategies.

    Defines how question embeddings are initialized and whether they are
    trainable during model optimization.

    Attributes:
        RAND_TRAINABLE: Random initialization with gradient updates enabled
        RAND_FROZEN: Random initialization frozen during training
        PRESET_FROZEN: Pre-trained embeddings (e.g., from OpenAI) frozen during training
    """
    RAND_TRAINABLE = 1
    RAND_FROZEN = 2
    PRESET_FROZEN = 3 # currently corresponds only to LLM embeddings


# # should probably have an "explicit" mode and an "implicit" mode
# # but then, in what form do we pass in data during the "implicit" mode?
# # hm.
# class DataMode(Enum):



class DataMode(Enum):
    """Missing data simulation patterns for training.

    Defines the strategy used to simulate missing data in the training set,
    which determines what the model learns to impute.

    Attributes:
        MCAR_MISSING: Missing Completely at Random - cells randomly removed
        RETRO_MISSING: Retrodiction - entire columns removed per wave
    """
    MCAR_MISSING = 1
    RETRO_MISSING = 2

class SynthDataMode(Enum):
    """Synthetic data integration modes for training augmentation.

    Controls whether and which LLM-generated synthetic survey responses are
    used to augment the training data.

    Attributes:
        SYNTH_DATA_NONE_INCOMPATIBLE: No synthetic data; the training data is not pruned to match
        SYNTH_DATA_NONE_COMPATIBLE: No synthetic data; training data is pruned to match synth. data
        SYNTH_DATA_GPT_4O: Use GPT-4o generated responses
        SYNTH_DATA_GPT_5: Use GPT-5 generated responses
        SYNTH_DATA_SONNET_4: Use Claude Sonnet 4 generated responses
        SYNTH_DATA_ALL: Use all available synthetic data sources
    """
    SYNTH_DATA_NONE_INCOMPATIBLE = 0
    SYNTH_DATA_NONE_COMPATIBLE = 1
    SYNTH_DATA_GPT_4O = 2
    SYNTH_DATA_GPT_5 = 3
    SYNTH_DATA_SONNET_4 = 4
    SYNTH_DATA_ALL = 5

type TrainDataInfo = tuple[DataMode, float]


class EpochMode(Enum):
    """Training duration control modes.

    Specifies whether training length is measured in complete epochs through
    the dataset or in total number of batch steps.

    Attributes:
        FIXED_EPOCHS: Train for a specified number of complete dataset passes
        FIXED_STEPS: Train for a specified number of batch updates
    """
    FIXED_EPOCHS = 1
    FIXED_STEPS = 2

type EpochInfo = tuple[EpochMode, int]


class TrainConfig(TypedDict):
    """Configuration for training neural network models.
    
    Well-typed mapping containing all hyperparameters and settings needed
    for model training, including architecture, training parameters, optimizer
    settings, and learning rate scheduler configuration.
    """
    model_name: str
    train_data_info: TrainDataInfo
    synth_data_mode: SynthDataMode
    apply_missing_to_synth_data: bool

    model_type: type[DCNImputeModelAdv]  # Type/class of the model to be trained
    model_config: ImputeExpConfig

    epoch_info: EpochInfo        # Controls length of training run; passed either fixed epochs or fixed steps (batches)
    epochs_per_assess: int           # Number of epochs between evaluation on the test set
    batch_size: int              # Number of samples per training batch

    opt_type: type[optim.Optimizer]  # Optimizer class/type
    opt_kwargs: dict[str, Any]       # Keyword arguments for optimizer initialization
    sched_type: type[LRScheduler]    # Learning rate scheduler class/type
    sched_kwargs: dict[str, Any]     # Keyword arguments for scheduler initialization


class ImputeExpConfig(TypedDict):
    """Experiment-facing configuration for imputation models.

    Does not include certain variables important for model initialization."""    
    n_dim: int
    depth: int
    dropout: float

    q_embedding_type: EmbeddingMode
    q_embedding_size: int

class ImputeConfig(ImputeExpConfig):
    """Configuration for imputation model architecture.
    
    Well-typed mapping defining the structure and hyperparameters for deep
    learning models used to predict missing survey responses. The model uses
    embeddings for individuals, time periods, and questions.
    """
    n_indiv: int  # Number of unique individuals in the dataset
    n_years: int  # Number of unique time periods/waves in the dataset
    n_questions: int  # Number of unique survey questions

    q_embedding_values: NotRequired[np.ndarray]