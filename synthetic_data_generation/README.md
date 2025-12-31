# Synthetic Data Generation Methodology

To address data sparsity constraints originating from item and unit gaps, we employ a cluster-guided LLM approach to generate synthetic survey data. We performed cluster analysis and identified two behavioral groups—one corresponding to generally positive attitudes towards vaccination, and the other to generally negative attitudes. We then performed a three-stage prompting method, inspired by Wang, Wang, and Sun (2024), using a few-shot approach to generate synthetic data for each wave. The main script to run the pipeline is `run_data_gen_pipeline.py`.

In order to evaluate two separate approaches of synthetic data generation, we compiled two datasets. For the first datasets, the LLM generated synthetic data only for three questions highly associated with cluster separation, as well as age, gender, and cluster labels. This was performed for waves 1, 3, 5 and 7 of the FluPaths study. For the second datasets, the LLM produced full responses for every wave in the FluPaths and COVIDPaths surveys, reconstructing all major variables that had sufficient response density in the original data. Cluster labels are included as well. This was performed for all Waves 1 to 14 of the FluPaths and COVIDPaths studies. For both of them, we apply the methodology we detail here, inspired by Wang et al. (2024).




## Step 1: Data preparation and generation of cluster descriptions

We start by cleaning missing data and selecting ego-related questions from the dataset. Then, we apply clustering analysis on the selected variables (questions). 

For the first dataset from a smaller set of questions, we carry out a cross-wave analysis of the clustering results, and for each wave, we select three variables corresponding to question items that are the most explicative of individual trajectories across waves, as well as the demographic attributes sex and age. For each wave, we select 4 random samples from each of the two clusters. For Waves 3, 5, and 7, we also include the vaccination outcome variable not present in Wave 1. The resulting dataset and samples are dense, with very low or zero degree of missingness.

For the second dataset from a larger set of questions (20-30), given context length constraints, we select all major variables that had sufficient response density in the original dataset. For each wave, given the larger number of variables, we select 30 random samples from each of the two clusters. It must be noted that the resulting dataset and samples still possess a significant degree of sparsity and missing not-at-random values, as opposed to the first dataset.

Finally, we prompt the LLM with summary statistics of the selected variables to obtain natural-language descriptions of the distributions of these variables in each cluster. We ask the LLM to provide the ``skewness'', ``spread'' and ``shape'' of the distribution of each variable in purely qualitative terms. Additionally, for the second dataset, given the complex dependency of the sparsity between variables, complexity that is not present in the first dataset, we ask the LLM to provide the NA percentage (in natural language terms) of each variable.


## Step 2: Formatted metadata generation
From the original metadata from the raw dataset, which contains the original survey questions, variable descriptions and the range of possible responses for each variable, we prompt the LLM using a metadata-driven approach similar to Wang et al. (2024). Dataset information is reformatted into concise natural-language summaries describing the dataset’s context, target variable, and feature meanings before being standardized into a structured format for downstream use.


## Step 3: Synthetic data prompting
We prompt the LLM incorporating the natural-language cluster descriptions, samples from each cluster, and the formatted metadata for synthetic data generation. In the first dataset, the LLM fills out every question given the dense features it is provided with. In the second dataset, the LLM is able to fill out responses with NA, given the sparsity in the features it is provided with.

In all steps, a structured output format is specified via a Pydantic model. All such models must be specified in the `pydantic_models` folder.


## Running the pipeline

Running the pipeline requires the libraries
- Pydantic: specifying output formats
- Instructor: prompting LLMs with Pydantic models
- Tenacity: API error handling
- Pandas: tabular data manipulation.

The main script, `run_data_gen_pipeline.py`, is run from the command line, using the format
`python run_data_gen_pipeline.py wave_num model model_dir_name API_KEY`
as specified in the file.

To run the pipeline, the following is required:

- API key, label for the desired model, and folder name. These only need to be specified at the start of the `main()` function in `run_data_gen_pipeline.py`. The model label needs to have the format `{provider}/{model_name}` (e.g., `anthropic/claude-sonnet-4-20250514`), as per the Instructor library functionality. The folder name will be used for writing the generated synthetic data to file, as well as logging outputs from Steps 1 and 2 (specified below).

- Number of data points to be generated in each synthetic dataset. These are specified at the start of the `main()` function in `run_data_gen_pipeline.py` through the parameters `n_outputs` and `n_batches` in `run_data_gen_pipeline.py`. The total number of points in each synthetic dataset will be `n_outputs * n_batches`. See `run_data_gen_pipeline.py` for more information.

- Number of samples to be selected from each cluster, specified at the start of the `main()` function in `run_data_gen_pipeline.py`

- Wave data files. These must contain the cluster labels (1 or 2) for each observation in a column named "cluster". This codebase assumes only two clusters in each wave. It is assumed that the labels 1 and 2 have the same meaning across waves. For Waves 3, 5, and 7, each observation must also contain the cluster label it had during the previous wave. These are in the `Original data` folder.

- Raw wave metadata as a JSON file, for each wave of interest. This file can be created from a CSV file containing in each row, a variable name from the wave data, the question text it corresponds to, and the possible responses to such question. This should only contain descriptions for the selected subset of variables in each wave. In this project, they are in the `Dictionaries/json_metadata` folder.
    - The same variables in the metadata file must be specified at the start of the `main()` function of `run_data_gen_pipeline.py`. These must be included in a dictionary that specifies the type of each variable (this is for creating a prompt for the LLM). Example for Wave 1:
    ```
    {
        "howlongagovaccine": "an ordinal", 
        "howoftenvaccine": "an ordinal", 
        "vaccimportant": "an ordinal"
    }
    ```
- Short descriptions of each wave's purpose and time of collection, specifying the target variable for the dataset and the wave conducted previous to it. These must be specified in the `Dictionaries/wave_contexts` folder.

- Dictionaries with the types of each variable, for each wave, to be inserted in the data generation prompt. These must be specified in the `Dictionaries/var_types` folder.

- Output folder for generated data. `run_data_gen_pipeline.py` will create a folder for the outputs from each tested model. In this project, this is the `Final_gen_data` folder.

- Logging directories to track outputs from Steps 1 and 2. These won't be read at any step during `run_data_gen_pipeline.py`, so if desired, the lines in the `data_gen_pipeline` function corresponding to logging can be commented out. Within each of these folders, `run_data_gen_pipeline.py` will create a folder for the outputs from each tested model. In this project, the directories are
    - `Dictionaries/cluster_descriptions` for cluster descriptions
    - `Examples` for cluster samples
    - `Gen_metadata` for formatted metadata


## Output

For each model, the synthetic datasets for each wave are stored in the specified output folder for generated data (i.e., `Final_gen_data`) within its respective folder. Thus, the output folder structure will resemble the following:

```
Final_gen_data
├── claude-sonnet-4
│   └── gen_data_wave1.csv
│   └── gen_data_wave2.csv
│   └── gen_data_wave3.csv
│     
└── gpt-5
    └── gen_data_wave1.csv
    └── gen_data_wave2.csv
    └── gen_data_wave3.csv
```

The logging output file structure will resemble the following:

For cluster descriptions:

```
Dictionaries
├── cluster_descriptions
    ├── claude-sonnet-4
    │   └── Wave 1 cluster descriptions.txt
    │   └── Wave 2 cluster descriptions.txt
    │   └── Wave 3 cluster descriptions.txt
    │     
    └── gpt-5
        └── Wave 1 cluster descriptions.txt
        └── Wave 2 cluster descriptions.txt
        └── Wave 3 cluster descriptions.txt

```

For cluster samples:
```
Examples
    ├── claude-sonnet-4
    │   └── Wave 1 examples.txt
    │   └── Wave 2 examples.txt
    │   └── Wave 3 examples.txt
    │     
    └── gpt-5
        └── Wave 1 examples.txt
        └── Wave 2 examples.txt
        └── Wave 3 examples.txt

```

For formatted metadata:
```
Gen_metadata
    ├── claude-sonnet-4
    │   └── Wave 1 generated metadata.txt
    │   └── Wave 2 generated metadata.txt
    │   └── Wave 3 generated metadata.txt
    │     
    └── gpt-5
        └── Wave 1 generated metadata.txt
        └── Wave 2 generated metadata.txt
        └── Wave 3 generated metadata.txt

```

## Evaluation

See the Evaluation folder for further details.

## Further references
R. Wang, Z. Wang, and J. Sun, UniPredict: Large Language Models are Universal Tabular Classifiers, 2024. arxiv.org/abs/2310.03266