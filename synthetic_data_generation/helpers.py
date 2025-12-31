import numpy as np
import pandas as pd
import json
import instructor # This library accesses the required APIs
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt
from pydantic import BaseModel
from pydantic_models.pydantic_models import *
from typing import Type
from pathlib import Path


@retry(retry=retry_if_exception_type(instructor.exceptions.InstructorRetryException), 
       wait=wait_exponential(multiplier=3, min=20, max=60), 
       stop=stop_after_attempt(5))
def prompt_model(sys_prompt: str, prompt: str, format_model: Type[BaseModel], 
                 model:str, api_key:str) -> str:
    """
    descriptions the selected from the supplied system and user prompt strings. 
    Uses the Instructor library for standardized API calls to all tested LLMs.
    Uses the Tenacity library with exponential backoff to deal with rate
    limit errors. If such an error occurs, tries up to 5 times, and raises
    a RetryError if the prompt is not successful in 5 attempts.

    Parameters
    ----------

    sys_prompt: str
        System prompt for the model. Sets the LLM's role in the prompt,
        i.e., "You are an epidemiologist who..."

    prompt: str
        User prompt for the model.

    format_model: inherits from BaseModel
        Format model inheriting from Pydantic's BaseModel class. This
        specifies the desired format for the generated response. Used
        in this project to guarantee standardized output formats from LLMs.

    model: str
        The LLM to be called for prompting. Needs to have the format
        {provider}/{model_name}, as per the Instructor library 
        functionality.

    api_key: str
        API key for the model being prompted.


    Returns
    ----------

    str
        Output of the LLM prompt, in stringified JSON format.

    
    Raises
    ----------
    instructor.exceptions.InstructorRetryException
        If the LLM being prompted reaches the rate limit for the supplied
        API key.
    
    RetryError
        If the LLM being prompted reaches the rate limit for the supplied
        API key after the specified limit of attempts (5 in this case).


    """

    # Standard client API from Instructor library
    client = instructor.from_provider(model, api_key=api_key)

    print(f"Prompting {model}...")

    # Prompt LLM
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            response_model=format_model
        )
    # If rate limit reached and number of attempts is less than 5,
    # try again
    except instructor.exceptions.InstructorRetryException as rate_limit_e:
        print(rate_limit_e)
        print("Rate limit reached, retrying...")
        raise rate_limit_e

    print("Prompt successful!")
    
    # Return prompt output as stringified JSON
    return response.model_dump_json(indent=2)


def get_var_desc_prompt(df_column: pd.Series, var_type: str) -> str:
    """
    Uses summary statistics of a given variable in a dataset to create a
    prompt requesting qualitative information for such column. 

    Used summary statistics:
    - Minimum
    - First quartile (pd.Series.quantile(0.25))
    - Median
    - Mean
    - Third quartile (pd.Series.quantile(0.25))
    - Maximum
    - Variance
    - NA percentage

    Parameters
    ----------

    df_column: pd.Series
        Vector representing observations of a given variable in a dataset.

    var_type: str
        Type of the variable, i.e., binary, categorical, ordinal, or
        continuous.


    Returns
    ----------

    str
        User prompt requesting qualitative information for the given variable.
        Concretely, requests information about the skew, spread, and shape of
        the distribution of this variable.


    """
    if var_type == "a categorical":
        a_freq = len(df_column[df_column == "a"]) / len(df_column)
        b_freq = len(df_column[df_column == "b"]) / len(df_column)
        c_freq = len(df_column[df_column == "c"]) / len(df_column)
        d_freq = len(df_column[df_column == "d"]) / len(df_column)
        e_freq = len(df_column[df_column == "e"]) / len(df_column)
        f_freq = len(df_column[df_column == "f"]) / len(df_column)
        na_freq = df_column.isna().sum() / len(df_column)

        prompt = f"""Please describe the distribution of a categorical variable that
        takes values 'a', 'b', 'c', 'd', 'e', 'f' and has the following summary statistics:
        Frequency of 'a': {a_freq: .2f}
        Frequency of 'b': {b_freq: .2f}
        Frequency of 'c': {c_freq: .2f}
        Frequency of 'd': {d_freq: .2f}
        Frequency of 'e': {e_freq: .2f}
        Frequency of 'f': {f_freq: .2f}
        NA percentage: {na_freq: .2f}
        Do not use any numbers in your answer. Respond, in at most one sentence, 
        to the questions below, and only those questions. Be succinct in your answers.
        Frequency of 'a':
        Frequency of 'b':
        Frequency of 'c':
        Frequency of 'd':
        Frequency of 'e':
        Frequency of 'f':
        NA percentage:"""
        
    else:
        # Stats to compute: min, 1q, median, mean, 3q, max, variance
        col_min = df_column.min()
        col_1q = df_column.quantile(0.25)
        col_median = df_column.median()
        col_mean = df_column.mean()
        col_3q = df_column.quantile(0.75)
        col_max = df_column.max()
        col_var = df_column.var()
        col_na = df_column.isna().sum() / len(df_column)

        prompt = f"""Please describe the distribution of {var_type} variable that takes 
        values from {col_min: .0f} to {col_max: .0f} and has the following summary statistics: 
        Min: {col_min: .0f}
        First quartile: {col_1q: .0f}
        Median: {col_median}
        Mean: {col_mean: .2f}
        Third quartile: {col_3q: .0f}
        Max: {col_max: .0f}
        Variance: {col_var: .2f} 
        NA percentage: {col_na: .2f}
        Do not use any numbers in your answer. Respond, in at most one sentence, 
        to the questions below, and only those questions. Be succinct in your answers.
        Skew:
        Spread:
        Shape:
        NA frequency:"""

    return prompt


def get_cluster_prompts_from_wave(merged_wave_df: pd.DataFrame,
                                  vars_for_description_dict: dict[str, str]
                                  ) -> dict[str, dict[str, str]]:
    """
    For the given wave dataset, outputs a dictionary containing prompts 
    for each specified variable, in each cluster. These prompts are the 
    output of calling get_var_desc_prompt on each variable, conditioned 
    on each cluster. Variables are specified through the 
    vars_for_description argument.

    Parameters
    ----------
    
    merged_wave_df: pd.DataFrame
        Wave dataset with cluster labels included for each individual,
        in a column labeled \"cluster\".

    vars_for_description_dict: dict
        Dictionary containing the variables to be selected as keys,
        whose values specify their types, i.e., ordinal, continuous,
        or categorical.


    Returns
    ----------

    dict
        A dictionary with the cluster labels as the keys. Each cluster has
        as value a dictionary with prompts for each (variable, prompt) pair 
        in vars_for_description_dict.

    Raises
    ----------

    ValueError
        If merged_wave_df does not contain the column labels in a column 
        named \"cluster\".

    KeyError
        If vars_for_description_dict contains variables not in
        merged_wave_df.

    """

    if "cluster" not in merged_wave_df.columns:
        raise ValueError("merged_wave_df must contain the column labels \
                         in a column named \"cluster\"")

    # Get list of variables to select
    vars_to_select = list(vars_for_description_dict.keys())
    vars_to_select.append("cluster")

    # Select specified variables
    try:
        reduced_wave = merged_wave_df.loc[:, vars_to_select]
    except KeyError:
        raise KeyError("vars_for_description contains variables not \
                       in the given dataframe")

    # Split df in two and remove the cluster attribute
    wave_cluster1 = reduced_wave[reduced_wave["cluster"] == 1]\
                    .drop(columns=["cluster"])
    wave_cluster2 = reduced_wave[reduced_wave["cluster"] == 2]\
                    .drop(columns=["cluster"])
    
    # Initialize dictionary containing prompts
    prompts = {"cluster_1": {}, "cluster_2": {}}

    # For each selected variable in cluster 1, create prompt and
    # append to prompts dictionary
    for col_name in wave_cluster1.columns:
        var_type = vars_for_description_dict[col_name]
        var_prompt = get_var_desc_prompt(wave_cluster1[col_name], var_type)
        prompts["cluster_1"][col_name] = var_prompt

    # For each selected variable in cluster 1, create prompt and
    # append to prompts dictionary
    for col_name in wave_cluster2.columns:
        var_type = vars_for_description_dict[col_name]
        var_prompt = get_var_desc_prompt(wave_cluster2[col_name], var_type)
        prompts["cluster_2"][col_name] = var_prompt

    return prompts


def build_metadata_prompt(wave_context: str, wave_dict: str) -> str:
    """
    Builds a prompt from the supplied wave context (string) and wave metadata (as JSON).
    Outputs a prompt incorporating this information.

    Parameters
    ----------

    wave_context: str
        Short description of the wave dataset and its purpose.

    wave_dict: str
        A Python dictionary serialized into a JSON string. This dictionary is
        created from a CSV file containing in each row, a variable name 
        from the wave data, the question text it corresponds to, and the 
        possible responses to that question.


    Returns
    ----------

    str
        User prompt for formatted metadata generation.

    """

    prompt = f"""The following is the metadata of a tabular dataset. {wave_context}
    Column meanings are as follows: {wave_dict}

    Instructions:

    Generate a description of the given dataset containing the following:
    1. The dataset description, in one sentence.
    2. The target of the dataset. If no target exists, choose one from the column as
    target for the dataset to classify.
    3. The features and their explanations.
    """

    return prompt


def build_gen_data_prompt(wave_meta: str, cluster_desc: str, 
                          examples_cluster1: str, examples_cluster2: str, 
                          n_outputs: int) -> str:
    """
    Builds a prompt from the supplied wave metadata string, cluster description string,
    and 2 examples from that wave's first and second clusters, in stringified JSON.
    Outputs a string with the above information.

    Parameters
    ----------
    
    wave_meta: str
        Natural-language description of wave metadata. In this project,
        this is the output of the function generate_metadata.

    cluster_desc: str
        Description of clusters in the wave dataset. In this project,
        this is the output of the function generate_cluster_desc.

    examples_cluster1: str
        Samples from cluster 1 in the wave dataset. These should be
        provided in stringified JSON format.

    examples_cluster2: str
        Samples from cluster 1 in the wave dataset. These should be
        provided in stringified JSON format.

    n_outputs: int
        Number of data points to be generated.


    Returns
    ----------

    str
        User prompt for synthetic data generation.
    """

    prompt = f"""
    Below you can find the metadata of a dataset, which contains the context, target and feature meanings. {wave_meta}
    Clusters were found in the dataset. The distribution of the variables in each cluster can be described
    as follows: {cluster_desc}

    Below are two examples from the first cluster: {examples_cluster1}
    Below are two examples from the second cluster: {examples_cluster2}

    INSTRUCTIONS:
    1. Generate data for exactly {n_outputs} individuals conforming to the given metadata and examples.
    2. Use the provided examples and cluster descriptions to recreate those patterns to the best of your ability.
    3. Consider real-world factors, including, but not limited to: vaccine hesitancy, changing attitudes, seasonal patterns, demographics, and health status.
    """

    return prompt


def prompt_model_for_cluster_descriptions(wave_prompts: dict[str, dict[str, str]], 
                                          model: str, api_key: str) -> str:
    """
    Prompts the given LLM to obtain qualitative descriptions for each variable
    in the wave_prompts dictionary, for each cluster. This function uses
    a Pydantic model (the class ClusterDescription) to specify the format
    of the prompt's output. For more details, see the pydantic_models
    documentation in this project.

    Parameters
    ----------

    wave_prompts: dict
        A dictionary with the cluster labels as the keys. Each cluster has
        as value a dictionary with prompts for each (variable, prompt) pair 
        in vars_for_description_dict. In this project, this is the output of
        the helper function get_cluster_prompts_from_wave. The prompt for each
        variable uses summary statistics conditioned to each cluster, and asks 
        the LLM to obtain the skew, spread, and shape of this conditional
        distribution.

    model: str
        The LLM to be called for prompting. Needs to have the format
        {provider}/{model_name}, as per the Instructor library 
        functionality.

    api_key: str
        API key for the model being prompted.


    Returns
    ----------

    str
        A dictionary of cluster descriptions, formatted as a JSON string.
        The keys are the cluster labels, and the values are dictionaries
        where each variable is mapped to its qualitative description, which
        consists of skew, spread, and shape.

        
    Raises
    ----------
    ValueError
        If wave_prompts does not have \"cluster_1\" or \"cluster_2\" as keys.


    """

    if "cluster_1" not in wave_prompts.keys() or \
        "cluster_2" not in wave_prompts.keys():
        raise ValueError("The dictionary must contain prompts for both clusters")
    
    sys_prompt = "You are an expert at giving short qualitative descriptions \
                  of variable distributions."
    
    # Cluster 1: prompt for all variables
    cluster1_dict = {}
    # Iterate over all variables and their respective prompts
    for variable, prompt in wave_prompts["cluster_1"].items():
        # Get the variable's qualitative description
        response = prompt_model(sys_prompt, prompt, ClusterDescription, model, api_key)
        # Deserialize into dictionary
        processed_response = json.loads(response) 
        # Record pair (variable, dictionary)
        cluster1_dict[variable] = processed_response


    # Cluster 2: prompt for all variables
    cluster2_dict = {}
    for variable, prompt in wave_prompts["cluster_2"].items():
        response = prompt_model(sys_prompt, prompt, ClusterDescription, model, api_key)
        processed_response = json.loads(response) # loads as a dict
        cluster2_dict[variable] = processed_response


    # Build output dictionary as serialized JSON string
    wave_cluster_desc = {"cluster_1": cluster1_dict, "cluster_2": cluster2_dict}
    str_wave_cluster_desc = json.dumps(wave_cluster_desc, indent=2)

    return str_wave_cluster_desc


def select_examples_from_wave(wave_df: pd.DataFrame, 
                              n_examples_per_cluster: int, 
                              vars_for_description_list: list[str], 
                              random_state: int) -> tuple[str, str]:
    """
    Selects at random the defined number of examples per cluster in the 
    given wave dataframe, with the specified attributes in
    vars_for_description_list. The dataframe must contain the cluster labels 
    in a column named "cluster". It is assumed that wave_df already contains 
    only the desired variables to keep. Returns a tuple, whose i-th entry is a 
    JSON string representing a dictionary of examples for cluster i.

    Parameters
    ----------

    wave_df: pd.DataFrame
        Wave data imported as a Pandas DataFrame. Must contain the cluster
        labels for each observation in a column named \"cluster\".

    n_examples_per_cluster: int
        Number of samples from wave_df to select in each cluster.

    vars_for_description_list: list
        List containing the variables to be selected from wave_df.

    random_state: int
        Random seed to reproduce random sampling.

        
    Returns
    ----------

    tuple
        A tuple, whose i-th entry is a JSON string representing a dictionary 
        of n_examples_per_cluster samples from wave_df for cluster i.

    """
    if "cluster" not in wave_df.columns:
        raise ValueError("wave_df must contain the column labels in a column \
                         named \"cluster\"")
    
    # Set random state
    np.random.seed(random_state)
    # Get the datapoints for each cluster
    cluster1_data = wave_df[wave_df["cluster"] == 1]
    cluster2_data = wave_df[wave_df["cluster"] == 2]

    # Get indices to select in each cluster
    indices_cluster1 = np.random.choice(len(cluster1_data), 
                                        size=n_examples_per_cluster, 
                                        replace=False)
    indices_cluster2 = np.random.choice(len(cluster2_data), 
                                        size=n_examples_per_cluster, 
                                        replace=False)

    # Select examples in each cluster with desired variables
    examples_cluster1 = wave_df.iloc[indices_cluster1]\
                        .loc[:, vars_for_description_list]
    examples_cluster2 = wave_df.iloc[indices_cluster2]\
                        .loc[:, vars_for_description_list]

    return (examples_cluster1.to_json(orient='index', indent=2), 
            examples_cluster2.to_json(orient='index', indent=2))


def get_wave_metadata(wave_context_file: Path, wave_dict_file: Path,
                      model: str, api_key: str) -> str:
    """
    Prompts the given LLM for wave metadata generation. Generated metadata 
    consists of three components: wave dataset context, the dataset's target
    variable (already_vaccinated for waves 3, 5, 7, vaccimportant for wave 1)
    and variable meanings for each variable in the supplied JSON metadata file.
    This function uses a Pydantic model (the class WaveDescription) to
    specify the prompt's output format. For more details, see the 
    pydantic_models documentation in this project.

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

    model: str
        The LLM to be called for prompting. Needs to have the format
        {provider}/{model_name}, as per the Instructor library 
        functionality.

    api_key: str
        API key for the model being prompted.

    Returns
    ----------

    str
        Formatted metadata as a JSON string.
    """

    # Get wave context and raw metadata from files
    with open(wave_context_file, 'r') as context_file:
        wave_context = context_file.read()

    with open(wave_dict_file, 'r') as dict_file:
        wave_dict = json.load(dict_file)
    wave_dict = json.dumps(wave_dict)
    
    # Prompt building
    sys_prompt = "You are an expert at describing datasets succinctly."
    user_prompt = build_metadata_prompt(wave_context=wave_context, wave_dict=wave_dict)

    # Prompting the model
    response = prompt_model(sys_prompt=sys_prompt, 
                            prompt=user_prompt, 
                            format_model=WaveDescription, 
                            model=model, 
                            api_key=api_key)

    return response


def get_wave_gen_data(wave_gen_meta: str, cluster_desc: str,
                      ex_cluster1: str, ex_cluster2: str,
                      format_model: Type[BaseModel], 
                      model: str, api_key: str, n_outputs):
    """
    Queries the given LLM for wave data generation. Outputs a synthetic
    dataset in the form of a string that can be parsed as JSON.
    This function uses a Pydantic model (the classes Wave1Data, Wave3Data, 
    etc.) to specify the prompt's output format. For more details, see the 
    pydantic_models documentation in this project.

    Parameters
    ----------
    wave_gen_meta: str
        Natural-language description of wave metadata. In this project,
        this is the output of the function generate_metadata.
    
    cluster_desc: str
        Description of clusters in the wave dataset. In this project,
        this is the output of the function generate_cluster_desc.

    ex_cluster1: str
        Samples from cluster 1 in the wave dataset. These should be
        provided in stringified JSON format.

    ex_cluster2: str
        Samples from cluster 1 in the wave dataset. These should be
        provided in stringified JSON format.

    format_model: inherits from BaseModel
        Format model inheriting from Pydantic's BaseModel class. This
        specifies the desired format for the generated wave data.

    model: str
        The LLM to be called for prompting. Needs to have the format
        {provider}/{model_name}, as per the Instructor library 
        functionality.

    api_key: str
        API key for the model being prompted.

    n_outputs: int
        Number of data points to be generated for each prompt.



    Returns
    ----------
    str
        String containing the generated dataset as stringified JSON.

    """

    # Get prompts
    sys_prompt = "You are an epidemiological data simulation expert \
                  that generates realistic data in JSON format."
    
    user_prompt = build_gen_data_prompt(wave_meta=wave_gen_meta, 
                                        cluster_desc=cluster_desc, 
                                        examples_cluster1=ex_cluster1, 
                                        examples_cluster2=ex_cluster2, 
                                        n_outputs=n_outputs)

    # Get data as stringified JSON
    response = prompt_model(sys_prompt=sys_prompt, 
                            prompt=user_prompt, 
                            format_model=format_model, 
                            model=model, 
                            api_key=api_key)

    return response