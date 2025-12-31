"""Synthetic Data Generation Pipeline

This script allows the user to generate synthetic data from each wave
of a longitudinal survey. We based this methodology on the FluPaths and
COVIDPaths surveys conducted by the RAND Corporation.

Details for using the pipeline are specified on the data_gen_pipeline
function and the README file on this folder. The main() function serves 
the purpose of setting up data directories with the pathlib library and 
calling the data_gen_pipeline function.

Running the pipeline requires:
- Pydantic: specifying output formats
- Instructor: prompting LLMs with Pydantic models
- Pandas: tabular data manipulation.

The Instructor library may require the jsonref library for Google
Gemini models.
"""

from pathlib import Path
# Must be created depending on the structure of the data
from pydantic_models.pydantic_models import *
from helpers import *
from typing import Type
from pydantic import BaseModel
from tenacity import RetryError
from sys import argv
import json


def data_gen_pipeline(wave_context_file: Path, wave_dict_file: Path,
                      cluster_desc_out: Path, examples_out: Path, 
                      gen_meta_out: Path, backup_out: Path, wave_df: pd.DataFrame,
                      vars_for_desc: dict[str, str], 
                      format_model: Type[BaseModel], model: str, api_key: str, 
                      n_examples=30, n_outputs=10, n_batches=70, random_state=42
                      ) -> list[dict]:
    """
    Pipeline for synthetic data generation. The function outputs intermediate
    steps, i.e., cluster descriptions, cluster examples and formatted
    metadata, to TXT files for logging. The input wave dataset must have
    the cluster labels for each observation.

    Step 1: generating cluster descriptions and extracting cluster samples.

    Step 2: generating formatted, natural-language metadata.

    Step 3: prompting for data generation. As the LLM can sometimes generate
    an inconsistent number of data points, we first follow Steps 1 and 2,
    and then prompt the model n_batches times, with the same exact prompt,
    for generating n_outputs points. The resulting datasets are stored in a
    list for concatenation in the main() function.


    Parameters
    ----------

    wave_context_file: Path
        Path to the wave context TXT file. This file should contain a short
        description of the wave dataset and its purpose.

    wave_dict_file: Path
        Path to the wave metadata JSON file. This file is created from 
        a CSV file containing in each row, a variable name from the wave data,
        the question text it corresponds to, and the possible responses to
        such question.

    cluster_desc_out: Path
        Path to the output folder for cluster descriptions. 

    examples_out: Path
        Path to the output folder for cluster examples.

    gen_meta_out: Path
        Path to the output folder for formatted metadata.

    backup_out: Path
        Path to the backup file to dump processed outputs in case of a RetryError.

    wave_df: pd.DataFrame
        Wave data imported as a Pandas DataFrame. Must contain the cluster
        labels for each observation in a column named \"cluster\".

    vars_for_desc: dict
        Dictionary containing the variables to be selected as keys,
        whose values specify their types, i.e., ordinal, continuous,
        or categorical.

    format_model: inherits from BaseModel
        Format model inheriting from Pydantic's BaseModel class. This
        specifies the desired format for each generated wave dataset.

    model: str
        The LLM to be called for prompting. Needs to have the format
        {provider}/{model_name}, as per the Instructor library 
        functionality.

    api_key: str
        API key for the model being prompted.

    n_examples: int
        Number of examples to be selected from each cluster. Defaults to 4.

    n_outputs: int
        Number of data points to be generated for each prompt. Defaults to 10.

    n_batches: int
        Number of batches in which to generate data points. Defaults to 10.

    random_state: int
        Random seed used in cluster sample selection.

    
    Returns
    ----------

    list[dict]
        A list with n_batches synthetic datasets, each with n_outputs
        synthetic data points. These datasets are Python dictionaries
        deserialized from JSON, able to be exported to a Pandas DataFrame.

    Raises
    ----------

    RetryError
        if the rate limit for the given model is reached.
    """
    
    print("##### Start of pipeline #####")

    ##### Step 1 #####

    ### Get cluster descriptions ###
    print("Generating cluster descriptions...")
    wave_prompts = get_cluster_prompts_from_wave(wave_df, vars_for_desc)
    wave_cluster_desc = prompt_model_for_cluster_descriptions(wave_prompts, model, api_key)


    ### Select examples ###
    print("Selecting examples...")
    vars_for_desc_list = list(vars_for_desc.keys())
    # The function returns a tuple with examples for cluster 1 in the first entry,
    # and for cluster 2 in the second
    wave_examples = select_examples_from_wave(wave_df, n_examples, vars_for_desc_list, random_state)

    # Output to file
    try:
        with examples_out.open('w', encoding='utf-8') as f:
            f.write("Cluster 1 examples: \n")
            f.write(wave_examples[0])
            f.write("\nCluster 2 examples: \n")
            f.write(wave_examples[1])
    except FileNotFoundError:
        with examples_out.open('x', encoding='utf-8') as f:
            f.write("Cluster 1 examples: \n")
            f.write(wave_examples[0])
            f.write("\nCluster 2 examples: \n")
            f.write(wave_examples[1])

    # Output to file
    try: # Write to existing file
        with cluster_desc_out.open('w', encoding='utf-8') as f:
            f.write(wave_cluster_desc)
    except FileNotFoundError: # Output into new file
        with cluster_desc_out.open('x', encoding='utf-8') as f:
            f.write(wave_cluster_desc)

    
    ##### Step 2 #####

    ### Generate metadata descriptions ###
    print("Generating metadata...")
    wave_gen_meta = get_wave_metadata(wave_context_file=wave_context_file,
                                       wave_dict_file=wave_dict_file,
                                       model=model,
                                       api_key=api_key)
    
    # Output to file
    try:
        with gen_meta_out.open('w', encoding='utf-8') as f:
            f.write(wave_gen_meta)
    except FileNotFoundError:
        with gen_meta_out.open('x', encoding='utf-8') as f:
            f.write(wave_gen_meta)
    

    ##### Step 3 #####

    ### Generate data ###
    print("Generating data...")
    # List of n_batches datasets to be generated
    gen_data_list = []
    # For each batch, generate a dataset with n_output data points,
    # and append to list
    for i in range(n_batches):
        print(f"Batch {i+1} of {n_batches}")
        try:
            wave_gen_data = get_wave_gen_data(wave_gen_meta, wave_cluster_desc,
                                               wave_examples[0], wave_examples[1],
                                               format_model, model, api_key, 
                                               n_outputs=n_outputs)
        except RetryError as e:
            # Dump everything so far into file
            with backup_out.open('a+', encoding='utf-8') as backup:
                backup.write(str(gen_data_list))
            raise e

        wave_data_dict = json.loads(wave_gen_data)
        gen_data_list.append(wave_data_dict)

    print("##### End of pipeline #####")
    return gen_data_list


def main():

    ##### Setup #####

    # LLM settings
    wave_num = argv[1]
    model = argv[2]
    model_dir_name = argv[3] # The folder name used for the above model
    API_KEY = argv[4] # Insert API key here

    # Validation
    print("Validating...")
    print("Wave: " + wave_num)
    print("Model: " + model)
    print("Directory name: " + model_dir_name)
    validation = input("Continue? y/N ")
    if validation != "y":
        raise ValueError("Please validate correctly.")
    
    # Prompt settings
    n_examples = 30 # Number of samples from each cluster to be selected
    n_outputs = 10 # Number of synthetic samples generated in each batch
    n_batches = 70 # Number of batches to generate



    # Input directories
    metadata_dir = Path.cwd() / "Dictionaries" / "json_metadata"
    data_dir = Path.cwd() / "Original data"
    wave_context_dir = Path.cwd() / "Dictionaries" / "wave_contexts"
    types_dir = Path.cwd() / "Dictionaries" / "var_types"
    # Dictionary containing all Pydantic models for each dataset
    # Must have the structure like the following:
    format_models = {"1": Wave1Data, "2": Wave2Data, "3": Wave3Data, "4": Wave4Data, 
                    "5": Wave5Data, "6": Wave6Data, "7": Wave7Data, "8a": Wave8aData, 
                    "8b": Wave8bData, "9": Wave9Data, "10": Wave10Data, "11":Wave11Data, 
                    "12": Wave12Data, "13": Wave13Data, "14": Wave14Data}

    # Logging directories
    gen_metadata_dir = Path.cwd() / "Gen_metadata" / model_dir_name
    cluster_desc_dir = Path.cwd() / "Dictionaries" / "cluster_descriptions" / model_dir_name
    example_dir = Path.cwd() / "Examples" / model_dir_name

    # Output directory for synthetic data
    output_dir = Path().cwd() / "Final_gen_data" / model_dir_name
    backup_path = output_dir / "backup.txt"

    
    # Wave files
    wave_id = f"wave_{wave_num}"
    wave_format = format_models[wave_num]
    wave_file = data_dir / f"wave{wave_num}.csv"
    meta_wave_file = metadata_dir / f"metadata_wave{wave_num}.json"
    context_wave_file = wave_context_dir / f"wave{wave_num}_context.txt"
    types_wave_file = types_dir / f"types_wave{wave_num}.json"

    output_cluster_desc_wave = cluster_desc_dir / f"Wave {wave_num} cluster descriptions.txt"
    output_examples_wave = example_dir / f"Wave {wave_num} examples.txt"
    output_gen_meta_wave = gen_metadata_dir / f"Wave {wave_num} generated metadata.txt"
    output_wave_file = output_dir / f"gen_data_wave{wave_num}.csv"

    # Selected variables for wave
    with types_wave_file.open('r', encoding='utf-8') as f:
        vars_for_desc_wave = json.load(f)

    wave = pd.read_csv(wave_file)

    ##### Run pipeline for each wave #####
    wave_data_list = data_gen_pipeline(wave_context_file=context_wave_file,
                                        wave_dict_file=meta_wave_file,
                                        cluster_desc_out=output_cluster_desc_wave,
                                        examples_out=output_examples_wave,
                                        gen_meta_out=output_gen_meta_wave,
                                        backup_out=backup_path,
                                        wave_df=wave,
                                        vars_for_desc=vars_for_desc_wave,
                                        format_model=wave_format,
                                        model=model,
                                        api_key=API_KEY,
                                        n_examples=n_examples,
                                        n_outputs=n_outputs,
                                        n_batches=n_batches)
    
    ##### Output to file #####

    # Convert all elements in each output list to dataframes
    wave_data_list = [pd.DataFrame(data=wave_data_list[i][wave_id]) for i in range(n_batches)]

    # Concatenate all dataframes in each list
    wave_gen_df = pd.concat(wave_data_list)

    # Output as CSV
    wave_gen_df.to_csv(output_wave_file)



if __name__ == "__main__":
    main()