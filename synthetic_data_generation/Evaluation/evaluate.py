from copy import deepcopy
import pandas as pd
import json
from pathlib import Path
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import LogisticDetection
from sdmetrics.visualization import get_column_plot
import plotly.io as pio
from PIL import Image
from io import BytesIO
from tqdm import tqdm


class Evaluator(object):
    """
    A class with utilities to evaluate the quality of a synthetic dataset 
    centered around sdmetrics's QualityReport output information. The
    central methods of the class are plot_density() and evaluate(). 
    Computations are carried out and explained in the class's static methods.
    
    Attributes
    ----------
    real_data_path: Path | str
        Path to the real data file. Should be a CSV file able to be opened
        by pandas.read_csv.
    syn_data_path: Path | str
        Path to the synthetic data file. Should be a CSV file able to be 
        opened by pandas.read_csv. Should have the same columns and
        column names as real_data_path.
    info_path: Path | str
        Path to the metadata file. Should be a JSON file, able to be parsed
        into a Python dictionary by json.load(). Must have two keys: 
        "column_names" and "metadata". "column_names" should be mapped to a
        list of variable names from the real dataset, and "metadata" should
        be mapped to another dictionary containing information about the type
        of such variables. This type must be either "categorical" or 
        "numerical" as per the sdmetrics documentation. Use this link to 
        format the metadata: 
        https://docs.sdv.dev/sdmetrics/getting-started/metadata/single-table-metadata

    Methods
    ----------
    @staticmethod
    self.evaluate_density(real_data_path, syn_data_path, info_path):
        Computes column shape, column trend and overall similarity scores. 
        Outputs two dictionaries. The first contains the aforementioned scores
        for the given synthetic datasets. The second contains Pandas
        DataFrames with more details about these metrics. For example, the 
        details for 'Column Shapes' shows the name of each individual 
        column, the metric that was used to compute it and the overall 
        score for that column.

    @staticmethod
    self.evaluate_c2st(real_data_path, syn_data_path, info_path):
        Computes the C2ST score for the given synthetic dataset.
        Outputs a dictionary with one key, "c2st", mapped to one value,
        the C2ST score.

    @staticmethod
    self.plot_density_single_column(real_data, syn_data, info_dict,
                                    num_per_row = 3):
        Outputs an image showing, for each variable, a plot comparing 
        the distributions of all variables in the real and synthetic 
        datasets. This uses the get_column_plot function from sdmetrics, 
        and the Image module to paste plots together. The num_per_row 
        parameter specifies how many plots should be shown in each line.

    self.plot_density():
        Imports the real and synthetic datasets with which the Evaluator
        object was initialized, and calls plot_density_single_column.

    self.evaluate():
        Imports the real and synthetic datasets with which the Evaluator
        object was initialized, and calls evaluate_density and
        evaluate_c2st. Returns a list whose first element is the output 
        of evaluate_density, and whose second element is the output of 
        evaluate_c2st.


    Technical notes
    ----------
    - Column shape similarity score: for each variable X, the distributions
    of X in the real and synthetic data are compared. If X is a continuous 
    variable (such as age), the Kolmogorov-Smirnov statistic KS of both 
    distributions is computed, and the quantity 1 - KS is taken as the 
    score of X. Otherwise, if X is a categorical or ordinal variable,
    the total variation distance (TVD) between both distributions is
    computed, and the quantity 1 - TVD is taken as the score of X.
    Finally, the scores for each variable are averaged to obtain a final 
    score for the given experimental run. In each case, a score of 1 
    indicates a perfect match between the real and synthetic columns, 
    while a score of 0 indicates no relationship between them.

    - Column trend similarity score: for each pair of variables (X, Y), 
    the joint distribution of (X, Y) in the real and synthetic data 
    are compared. If both X and Y are categorical or ordinal 
    variables, a contingency table is computed for both the real and 
    synthetic data, and then the difference between them is computed 
    via the TVD. If one of X or Y is continuous and the other is 
    categorical or ordinal, the continuous variable is discretized 
    and the score is computed as before. If both are continuous, the 
    Pearson or Spearman correlation coefficient is computed. Finally, 
    the scores for each variable are averaged to obtain a final score 
    for the given experimental run. As before, a score of 1 indicates 
    a perfect match between the trends in the real and synthetic columns, 
    while a score of 0 indicates no relationship.

    - Classifier 2-sample test score (C2ST): computes a metric that 
    indicates how difficult it is to tell apart the synthetic data from 
    the real data. The real and synthetic data are pooled and then split 
    into training and validation sets, and then a classifier (such as a 
    logistic regression model) is trained to predict whether a given row 
    is real or synthetic. The classifier is tested on the validation set, 
    and the aforementioned steps are repeated several times. The final 
    score is based on the AUC-ROC score of the classifier across all 
    splits. A score of 1 indicates that the model does not distinguish 
    at all between the real and synthetic rows (i.e., does not do better 
    than chance), while a score of 0 indicates that it perfectly 
    distinguishes the real from the synthetic data.

    - Overall similarity score: the average of the column shape and column 
    pair trends score. As before, a score of 1 indicates a perfect match 
    between the trends in the real and synthetic columns, while a score 
    of 0 indicates no relationship.
    

    """
    # Constructor: real_data_path, synthetic_data_path, info_path
    def __init__(self, real_data_path: Path | str, syn_data_path: Path | str, info_path: Path | str) -> None:
        """
        Evaluator constructor method.

        Parameters
        ----------

        real_data_path: Path | str
            Path to the real data file. Should be a CSV file able to be opened
        by pandas.read_csv.

        syn_data_path: Path | str
            Path to the synthetic data file. Should be a CSV file able to be 
        opened by pandas.read_csv. Should have the same columns and
        column names as real_data_path.

        info_path: Path | str
            Path to the metadata file. Should be a JSON file, able to be parsed
        into a Python dictionary by json.load(). Must have two keys: 
        "column_names" and "metadata". "column_names" should be mapped to a
        list of variable names from the real dataset, and "metadata" should
        be mapped to another dictionary containing information about the type
        of such variables. This type must be either "categorical" or 
        "numerical" as per the sdmetrics documentation. Use this link to 
        format the metadata: 
        https://docs.sdv.dev/sdmetrics/getting-started/metadata/single-table-metadata

        """
        self.real_data_path = real_data_path
        self.syn_data_path = syn_data_path
        self.info_path = info_path
        

    @staticmethod
    def evaluate_density(real_data: pd.DataFrame, syn_data: pd.DataFrame, 
                         info_dict: dict) -> tuple[dict, dict]:
        """
        Computes column shape, column trend and overall similarity scores. 
        Outputs two dictionaries. The first contains the aforementioned scores
        for the given synthetic datasets. The second contains Pandas
        DataFrames with more details about these metrics. For example, the 
        details for 'Column Shapes' shows the name of each individual 
        column, the metric that was used to compute it and the overall 
        score for that column.

        Parameters
        ----------

        real_data: pd.DataFrame
            Real dataset in pd.DataFrame format.
        syn_data: pd.DataFrame
            Synthetic dataset in pd.DataFrame format.
        info_dict: dict
            Python dictionary representing the imported metadata from 
            info_path.

        Returns
        ----------
        tuple[dict, dict]
            A tuple with two dictionaries. The first contains the 
            aforementioned scores for the given synthetic dataset. 
            The second contains Pandas DataFrames with more details 
            about these metrics. For example, the details for 'Column Shapes'
            shows the name of each individual column, the metric that was 
            used to compute it and the overall score for that column.

        """

        # Create a copy of the imported metadata
        info = deepcopy(info_dict)
        metadata = info['metadata'] # Metadata: tells the algorithm the 
        # response type of each column --> decide the appropriate evaluation metrics

        metadata_col_names = info["column_names"]
        # Transform metadata to format required by sdmetrics
        metadata['columns'] = {metadata_col_names[int(key)]: value for key, value in metadata['columns'].items()} 


        # Evaluate the metrics with sdmetrics's QualityReport
        qual_report = QualityReport()
        qual_report.generate(real_data, syn_data, metadata)

        quality = qual_report.get_properties() # Retrieve the evaluated values on Column Shapes and Column Pair Trends

        Shape = quality['Score'][0] # The evaluated value for Column Shape
        Trend = quality['Score'][1] # The evaluated value for Column Trend

        Overall = (Shape + Trend) /2 # Calculate the overall similarity score

         # Get details on the methods being used to evaluate Column Shape on each column
        shape_details = qual_report.get_details(property_name = 'Column Shapes')
        # Get details on methods being used to evaluate Column Trend on each column
        trend_details = qual_report.get_details(property_name = 'Column Pair Trends')

        # Evaluated values for Shape, Trend and overall scores
        shape_metrics = {
            'density/Shape': round(Shape, 4), 
            'density/Trend': round(Trend, 4), 
            'density/Overall': round(Overall, 4)
        }

        # Detailed scores for each column
        shape_details = {
            'shapes': shape_details, 
            'trends': trend_details
        }


        return shape_metrics, shape_details
    
    @staticmethod
    def evaluate_c2st(real_data, syn_data, info_dict) -> dict:
        """
        Computes the Classifier 2-Sample Test (C2ST) score for 
        the given synthetic dataset. See technical notes for more information.

        Parameters
        ----------

        real_data: pd.DataFrame
            Real dataset in pd.DataFrame format.
        syn_data: pd.DataFrame
            Synthetic dataset in pd.DataFrame format.
        info_dict: dict
            Python dictionary representing the imported metadata from 
            info_path.

        Returns
        ----------
        dict
            A dictionary with one key, "c2st", mapped to one value,the 
            C2ST score.

        """
        # Create a copy of the imported metadata
        info = deepcopy(info_dict)

        # Transform metadata into required format by sdmetrics
        metadata = info['metadata']
        metadata_col_names = info["column_names"]
        metadata['columns'] = {metadata_col_names[int(key)]: value for key, value in metadata['columns'].items()}

        # Compute C2ST score with sdmetrics's LogisticDetection module
        score = LogisticDetection.compute(
            real_data = real_data, 
            synthetic_data = syn_data, 
            metadata = metadata
        )

        out_metrics = {"c2st": score}
        return out_metrics
    

    @staticmethod
    def plot_density_single_column(real_data, syn_data, info_dict, num_per_row = 3) -> Image.Image:
        """
        Creates plots visualizing distributions of individual variables in
        real and synthetic datasets. This uses the get_column_plot function 
        from sdmetrics, and the Image module to paste plots together. 

        Parameters
        ----------

        real_data: pd.DataFrame
            Real dataset in pd.DataFrame format.
        syn_data: pd.DataFrame
            Synthetic dataset in pd.DataFrame format.
        info_dict: dict
            Python dictionary representing the imported metadata from 
            info_path.
        num_per_row: int
            Specifies how many plots should be shown in each line. Set to 3
            by default.

        Returns
        ----------
        Image.Image
            An Image showing, for each variable, a plot comparing the 
            distributions of all variables in the real and synthetic 
            datasets.
        """

        # Create copies of metadata and synthetic data
        syn_data_cp = deepcopy(syn_data)
        info = deepcopy(info_dict)

        # Get column names for plot titles
        column_names = info['column_names']
        num_cat = len(column_names)
        num_col = num_per_row # Initialize the number column for the plots 
        num_row = (num_cat - 1) // num_col + 1 # Initialize the number of rows for the plots

        # Initialize list of images for pasting later on
        imgs = []
        # Iterate over all attributes in the real/synthetic datasets
        for i, col in tqdm(enumerate(column_names), total = len(column_names)): 
            # Plot the variable distribution in both datasets.
            # Bar plot if the data type is 'categorical': usually our case
            # Density plot if the data type is 'numerical', only used for 'age'

            plot_type = 'bar' if info['metadata']['columns'][str(i)]['sdtype'] == 'categorical' else 'distplot'
            fig = get_column_plot(
                real_data = real_data,
                synthetic_data = syn_data_cp,
                column_name = col,
                plot_type = plot_type
            )
            
            # Append to list of images
            img_bytes = pio.to_image(fig, format='png')
            img = Image.open(BytesIO(img_bytes))
            imgs.append(img)

        # Paste all generated plots into one image
        width, height = imgs[0].size
        big_img = Image.new('RGB', (width * num_col, height * num_row))
        for i, img in enumerate(imgs):
            coordinate = (i%num_col * width, i//num_col * height)
            big_img.paste(img, coordinate)
        
        # Return the image with all plots
        return big_img


    def plot_density(self) -> Image.Image:
        """
        Imports the real and synthetic datasets with which the Evaluator
        object was initialized, and calls plot_density_single_column.

        Returns
        ----------
        Image.Image
            An Image showing, for each variable, a plot comparing the 
            distributions of all variables in the real and synthetic 
            datasets.

        """
        # Import real and synthetic datasets, and variable metadata
        real_data = pd.read_csv(self.real_data_path)
        syn_data = pd.read_csv(self.syn_data_path)
        with open(self.info_path, 'r') as f:
            info = json.load(f)

        # Generate plots
        img = self.plot_density_single_column(real_data, syn_data, info)
        return img
        
    
    def evaluate(self) -> list:
        """
        Imports the real and synthetic datasets with which the Evaluator
        object was initialized, and calls evaluate_density and 
        evaluate_c2st.

        Returns
        ----------
        list
            A list whose first element is the output of evaluate_density, 
            and whose second element is the output of evaluate_c2st.

        """
        # Load the real data, synthetic data, and the info file
        real_data = pd.read_csv(self.real_data_path)
        syn_data = pd.read_csv(self.syn_data_path)
        with open(self.info_path, 'r') as f:
            info = json.load(f)

        # Select same columns from both datasets using the metadata file
        metadata_col_names = info["column_names"]
        real_data = real_data.loc[:, metadata_col_names]
        syn_data = syn_data.loc[:, metadata_col_names]

        # Call the evaluation functions
        return [self.evaluate_density(real_data, syn_data, info), self.evaluate_c2st(real_data, syn_data, info)]


def record_eval_metrics(eval_metrics: list) -> dict:
    """
    Helper function that computes column shape, column trend, overall
    and C2ST similarity scores calling Evaluator.evaluate(). 
    See Evaluator documentation for more information.

    Parameters
    ----------

    eval_metrics: list
        Output from Evaluator.evaluate().

    Returns
    ----------
    dict
        Dictionary containing the column shape, column trend, overall
        and C2ST similarity scores for the given eval_metrics object.

    """
    # Record the overall metrics for shape and trend
    overall_metrics = eval_metrics[0][0]

    # Compute individual shape, trend and overall scores
    density_shape = round(overall_metrics['density/Shape'], 3)
    density_trend = round(overall_metrics['density/Trend'], 3)
    density_overall = round(overall_metrics['density/Overall'], 3)

    # Compute C2ST score
    c2st = round(eval_metrics[1]['c2st'], 4)

    return {'density_shape': density_shape, 'density_trend': density_trend, 
            'density_overall': density_overall, 'c2st': c2st}