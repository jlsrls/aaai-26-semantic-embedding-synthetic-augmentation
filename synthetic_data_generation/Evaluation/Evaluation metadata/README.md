This folder contains the evaluation metadata for each wave. These must have the attributes "column_names" and "metadata". "column_names" must be mapped to the desired subset of variables for the selected wave, specified previously in the raw metadata JSON files that serve as input to the data generation pipeline. "metadata" must contain a dictionary with number i as key, whose value is the data type of the i-th element in "column_names". Acceptable data types are "numerical" and "categorical". An example is as follows:

```
{
    "column_names": ["var1", "var2", "var3"],
    "metadata": {
        "columns": {
            "0": {
                "sdtype": "numerical"
            },
            "1": {
                "sdtype": "categorical"
            },
            "2": {
                "sdtype": "categorical"
            }
        }
    }
}
```