import os
from datetime import date
import pandas as pd

# output path
main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Isochrone_Archive/Coding_logs/"


def set_output_path(main_path: str = main):
    """
    Automatically sets the output directory to a folder named after the current date.

    :param main_path: Path to the directory containing the coding-logs
    :return: path-string
    """
    subdir = date.today()
    output_path = os.path.join(main_path, str(subdir))
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    output_path = output_path + "/"
    return output_path


def setup_HP(filepath_and_name: str, name_string: str = None):
    """
    Reading in hyperparameters from the HP file at the given location. Either accepts a
    custom name string, or the one saved as default for more flexibility.

    :param filepath_and_name: path to the HP file
    :param name_string: String of csv headings or column names that will be the keys of the HP dictionary
    :return: either reads the file into a dataframe or opens a new one where it writes the name-string
    """
    if name_string is None:
        name_string = "id,name,abs_mag,cax,score,std,dataset_id,C,epsilon,gamma,kernel"
    try:
        pd.read_csv(filepath_and_name)
    except FileNotFoundError:
        with open(filepath_and_name, "w") as f:
            f.write(name_string + "\n")