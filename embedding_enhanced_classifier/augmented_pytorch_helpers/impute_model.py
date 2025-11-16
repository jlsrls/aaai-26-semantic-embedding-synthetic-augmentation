"""Neural network architectures for survey response imputation.

This module defines PyTorch model architectures based on Deep & Cross Networks
(DCN) for predicting missing binary survey responses. Models use embeddings for
individuals, time periods, and questions, with support for pre-trained language
model embeddings (e.g., from OpenAI).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from augmented_pytorch_helpers.deepctr_torch.layers.interaction import CrossNet
from augmented_pytorch_helpers.configs import ImputeConfig, EmbeddingMode



class Square(nn.Module):
    """
    Custom activation layer that applies element-wise square operation.

    Simple PyTorch module that computes the square of input tensors.
    Used in advanced model variants for additional non-linearity.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply element-wise square to input tensor.

        Args:
            x: Input tensor of any shape

        Returns:
            Tensor with same shape as input, with all values squared
        """
        return torch.square(x)

class DCNImputeModelAdv(nn.Module):
    """
    Advanced Deep & Cross Network imputation model with architectural improvements.
    
    Enhanced version of DCNImputeModel with architectural modifications that
    further improve performance beyond the base methodology from Kim and Lee [1].
    Includes better organization of components, improved activation functions,
    and normalization layers.
    
    Key improvements over DCNImputeModel:
    - Modular organization with ModuleDict for embeddings and body components
    - Enhanced question embedding processing with ReLU and Square activation
    - RMSNorm layers for better gradient flow
    - Square activation functions for additional non-linearity
    
    Architecture:
    - Embeddings: Individual, year, and question embeddings with improved processing
    - Body: CrossNet and enhanced linear layers with normalization
    - Head: Final linear layer with sigmoid activation for binary output
    
    Input format: [individual_id, year_id, question_id]
    Output: Binary probability (0-1) for survey response
    
    References:
    [1] J. Kim and B. Lee, "AI-Augmented Surveys: Leveraging Large Language Models
        and Surveys for Opinion Prediction," arXiv:2305.09620, 2023.
    """
    def __init__(self, conf: ImputeConfig):
        """Initialize the DCN imputation model with specified configuration.

        Args:
            conf: Configuration dictionary containing architecture parameters including:
                  - n_indiv: Number of unique individuals
                  - n_years: Number of time periods/waves
                  - n_questions: Number of survey questions
                  - n_dim: Embedding dimension for person/year/question
                  - depth: Number of CrossNet and linear layers
                  - dropout: Dropout probability
                  - q_embedding_type: How to initialize question embeddings
                  - q_embedding_size: Dimension of question embeddings
                  - q_embedding_values: Pre-trained embeddings (if PRESET_FROZEN mode)
        """
        super(DCNImputeModelAdv, self).__init__()

        self.conf = conf

        # Initialize individual and time period embeddings
        # Question embeddings handled separately below based on embedding mode
        self.embeddings = nn.ModuleDict({
            "pers_embed": nn.Embedding(conf["n_indiv"], conf["n_dim"]),
            "year_embed": nn.Embedding(conf["n_years"], conf["n_dim"]),
        })

        # Build deep network: stack of (RMSNorm -> Linear -> Dropout -> ReLU -> Square)
        # Each layer processes concatenated embeddings (n_dim * 3)
        lin_modules = []
        for i in range(conf["depth"]):
            lin_modules.append(nn.modules.normalization.RMSNorm(conf["n_dim"] * 3))
            lin_modules.append(nn.Linear(conf["n_dim"] * 3, conf["n_dim"] * 3))
            lin_modules.append(nn.Dropout(conf["dropout"]))
            lin_modules.append(nn.ReLU())
            lin_modules.append(Square())

        # Body contains two parallel paths: CrossNet for explicit feature crossing
        # and deep network for implicit interactions
        self.body = nn.ModuleDict({
            "crossnets": nn.Sequential(
                CrossNet(conf["n_dim"] * 3, conf["depth"] * 2),
                nn.Dropout(conf["dropout"]),
            ),
            "linear_layers": nn.Sequential(*lin_modules)
        })

        # Configure question embeddings based on the specified mode
        # Three modes: trainable random, frozen random, or pre-trained (from OpenAI)
        if conf["q_embedding_type"] == EmbeddingMode.RAND_TRAINABLE:
            # Standard trainable embeddings
            self.embeddings.update({"q_embed": nn.Embedding(
                conf["n_questions"],
                conf["q_embedding_size"],
                _freeze = False
            )})
        else:
            # Frozen embeddings (random or pre-trained)
            if conf["q_embedding_type"] == EmbeddingMode.RAND_FROZEN:
                self.embeddings.update({"q_embed": nn.Embedding(
                    conf["n_questions"],
                    conf["q_embedding_size"],
                    _freeze = True
                )})
            elif conf["q_embedding_type"] == EmbeddingMode.PRESET_FROZEN:
                # Load pre-trained embeddings (e.g., from OpenAI text-embedding models)
                self.embeddings.update({"q_embed": nn.Embedding.from_pretrained(
                    torch.from_numpy(conf["q_embedding_values"]), # type: ignore
                    freeze = True
                )})

            q_embed_size = conf["q_embedding_size"]

            # For frozen embeddings, add a condenser network to project to n_dim
            # This allows different embedding sizes while maintaining consistent architecture
            self.body.update({"q_embed_condense": nn.Sequential(
                nn.Linear(q_embed_size, conf["n_dim"]),
                nn.ReLU(),
                Square()
            )})

        # Final prediction head: concatenate both paths (n_dim * 6) -> binary probability
        self.head = nn.Sequential(
            nn.Linear(conf["n_dim"] * 6, 1),
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DCN imputation model.

        Args:
            x: Input tensor of shape (batch_size, 3) containing:
               - Column 0: Individual IDs (integer indices)
               - Column 1: Year/wave IDs (integer indices)
               - Column 2: Question IDs (integer indices)

        Returns:
            Tensor of shape (batch_size, 1) with predicted probabilities (0-1)
            for binary survey responses
        """
        # Embed each input feature: person, year, and question
        # type: ignore comments are for ModuleDict dynamic key access
        x_pers_embed = self.embeddings.pers_embed(x[:, 0]) # type: ignore
        x_year_embed = self.embeddings.year_embed(x[:, 1]) # type: ignore

        # Normalize question embeddings to unit vectors
        # This helps stabilize training when using different embedding sizes
        x_q_embed = nn.functional.normalize(self.embeddings.q_embed(x[:, 2])) # type: ignore

        # If using frozen embeddings (random or pre-trained), project to n_dim
        if "q_embed_condense" in self.body.keys():
            x_q_embed = self.body.q_embed_condense(x_q_embed) # type: ignore

        # Concatenate all embeddings: (batch, n_dim * 3)
        # Order: question, person, year
        x = torch.cat((x_q_embed, x_pers_embed, x_year_embed), dim = -1)

        # Parallel processing through two network paths:
        # - CrossNet (x_a): Learns explicit feature interactions
        # - Deep network (x_b): Learns implicit patterns
        x_a = self.body.crossnets(x) # type: ignore
        x_b = self.body.linear_layers(x) # type: ignore

        # Concatenate both paths: (batch, n_dim * 6)
        x = torch.cat((x_a, x_b), dim = -1)

        # Final prediction: linear layer + sigmoid for binary probability
        x = self.head(x)

        return x
