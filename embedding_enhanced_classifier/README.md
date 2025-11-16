# Embedding-Enhanced Classifier

A neural network-based system for data imputation using deep learning and embeddings, designed for longitudinal survey data with missing values.

## Overview

This project implements a Deep Cross Network (DCN) architecture for imputing missing values in longitudinal datasets. The system can utilize OpenAI embeddings to enhance question representation and improve imputation accuracy. Random hexadecimal names are used to store experiment data and must be tracked manually by the user. Experiment design may be configured via the modification of the code in `new_imp.py` which controls dataset selection and training run parameters.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)
- OpenAI API key (for embedding generation)

### Dependencies
Install required packages:
```bash
pip install torch pandas scikit-learn numpy scipy tqdm openai
```

## Running the Primary File

### Basic Usage

1. **Set up your OpenAI API key**: 
   Replace the OpenAI API key in `new_imp.py` with your actual OpenAI API key.

2. **Run the training script**:
   ```bash
   python new_imp.py
   ```

### Configuration

The training is configured through the `TrainConfig` object in `new_imp.py`. Key parameters include:

- `exp_tag`: Experiment identifier (affects which dataset is used)
- `model_type`: Network architecture (DCNImputeModelAdv)
- `model_dim`: Hidden dimension size
- `model_depth`: Number of network layers
- `use_llm_embeddings`: Whether to use OpenAI embeddings or random embeddings
- `embedding_dim`: Size of question embeddings
- `batch_size`: Training batch size
- `num_epochs`: Total training epochs

### Data Requirements

The system expects:
- Metadata CSVs in the `./data_manip/metadata` folder with filenames `metadataN_ego.csv`, where `N` is the wave identifier (3, 8a, 10, etc.). These metadata files are expected to be encoded in the Microsoft CP-1252 text encoding format, as the original metadata files used for this analysis were. Our analysis requires the presence of `value_for_missing`, `yes_list` and `one_hot` columns within these metadata files; these attributes were assembled manually as part of our research.
- Raw data CSVs with wave-based survey responses in the `./data_manip/raw_data` folder with filenames `waveN.csv`, where `N` is the wave identifier (3, 8a, 10, etc.). These files are expected to be encoded with UTF-8, the default text codec.

### Output

The system will:
1. Generate embeddings for survey questions
2. Train the imputation models
3. Save model weights and configurations to `./saved_models/`
4. Generate training and testing logs
5. Print basic final performance metrics

### File Structure After Running

```
./saved_models/[model_name]/
├── [model_name]-config.txt          # Model configuration
├── [model_name]-ep[X]-model-weights.pt  # Model weights by epoch
├── [model_name]-data-train.parquet  # Training data
├── [model_name]-data-test.parquet   # Test data
├── [model_name]-transformer-*.pkl   # Data transformers, used to assign IDs to datapoints
└── [model_name]-logs-*.parquet      # Training logs
```

## Key Functions

- `create_and_train()`: Main training function that creates and trains the imputation model
- `get_embeddings()`: Generates OpenAI embeddings for survey questions
- `create_id_transformers()`: Creates label encoders for categorical variables

## Hardware Requirements

- GPU with CUDA support (recommended)
- Sufficient RAM for dataset size
- Storage space for model outputs and logs

## Notes

- The system automatically sets random seeds for reproducible results
- Model training progress is logged and can be monitored
- Pre-computed embeddings are cached to avoid redundant API calls