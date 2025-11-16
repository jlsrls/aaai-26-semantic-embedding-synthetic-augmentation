"""OpenAI embedding generation and caching for question text.

This module provides functionality to generate text embeddings for survey questions
using OpenAI's embedding models. Embeddings are cached to disk to avoid redundant
API calls and normalized to standard normal distribution for compatibility with
PyTorch's embedding initialization conventions.
"""
from openai import OpenAI
import os
import pickle
from typing import Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


client: OpenAI


# embedding-3-small max dims: 1536
# embedding-3-large max dims: 3072
def get_embeddings(
        master_frame: pd.DataFrame, 
        openai_model_name: str = "text-embedding-3-large", 
        embedding_size: int = 3072
    ) -> np.ndarray:
    """
    Generates OpenAI embeddings for question labels in the dataset.
    
    Args:
        master_frame: DataFrame containing question data with 'question_id' and 'variable_label' columns
        openai_model_name: OpenAI model to use for embeddings (default: "text-embedding-3-large")
        embedding_size: Dimension size of the embeddings (default: 3072)
        
    Returns:
        numpy.ndarray: Array of embeddings with shape (n_questions, embedding_size)
    """
    # keeping them sorted by ID ensures they'll match up with the order of embedding layer weights
    question_list = master_frame[['question_id', 'variable_label']].drop_duplicates().sort_values('question_id')['variable_label'].tolist()
    
    n_embeddings = len(question_list)
    pickle_name = f"./embeddings-{n_embeddings}-{openai_model_name}-{embedding_size}.pkl"

    if os.path.isfile(pickle_name):
        return pickle.load(open(pickle_name, "rb"))

    embedding_list = []
    failure = False
    for q in tqdm(question_list):
        try:
            embed = client.embeddings.create(
                input = [q.replace("\n", " ")],
                model = openai_model_name,
                dimensions = embedding_size
            ).data[0].embedding
            embedding_list.append(np.array(embed).reshape(-1))
        except Exception as e:
            failure = True
            print(f"Error processing question: {q[:50]}... Error: {e}")
            embedding_list.append(np.random.randn(embedding_size))

    embedding_weights = np.asarray(embedding_list)

    if not failure:
        # convert to a standard normal distribution
        # pytorch expects embeddings to be sampled from a standard normal distribution, so this ensures their magnitude
        # matches that of the other embeddings. it's also generally good practice

        embedding_weights = ((embedding_weights - np.mean(embedding_weights, axis=0)) / np.std(embedding_weights, axis=0)).astype(np.float32)
        pickle.dump(embedding_weights, open(pickle_name, "xb"))

    return embedding_weights.astype(np.float32)
