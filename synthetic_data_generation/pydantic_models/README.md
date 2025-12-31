# Prompt output format models

In this methodology we use the Pydantic and Instructor libraries to specify the format of each of our prompts' outputs. In each prompt, we want the LLM to output a JSON string with uniform attribute names, which is the motivation for using the Pydantic library. The file pydantic_models.py must contain the format models for the cluster description, wave metadata and data generation format models.

For example, if we specify the following Pydantic model as format model for a prompt,

```
class ClusterDescription(BaseModel):
    skew: str
    spread: str
    shape: str
```

the result will have the form

```
{
    "skew": ...,
    "spread": ...,
    "shape": ...,
}
```

In particular, this allows the output of the data generation prompt to return a dataset with the same variables that are supplied to it. See the Pydantic and Instructor libraries' documentations for more details on functionality.

This folder must also include a file named __init__.py to allow proper initialization. For further details about the specific structure of the models, please contact the authors.