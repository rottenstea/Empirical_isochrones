import os
from datetime import date
import pandas as pd

# output paths
main = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Isochrone_Archive/Coding/"


def set_output_path(main_path: str = main):
    subdir = date.today()
    output_path = os.path.join(main_path, str(subdir))
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    output_path = output_path + "/"
    return output_path


def setup_HP(filepath_and_name: str, name_string: str = None):
    if name_string is None:
        name_string = "id,name,abs_mag,cax,score,std,catalog_id,C,epsilon,gamma,kernel"
    try:
        pd.read_csv(filepath_and_name)
    except FileNotFoundError:
        with open(filepath_and_name, "w") as f:
            f.write(name_string + "\n")
