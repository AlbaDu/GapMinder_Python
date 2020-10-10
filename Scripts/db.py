import pandas as pd
import numpy as np

def file2df(file_):
    """
    ---What it does---
        + Generates a dataframe from a files: ".csv", ".xlsx" or ".xls".
    ---What it needs---
        + file: data consisting in structured in rows and columns.

    IMPORTANT! the files need to be stored in the path specified by the function.    
    ---What it returns---
        + new_df: dataframe containing the data from the file.
    """
    path = "C:\\Users\\34609\Documents\\Repos Git\\GapMinder_Python\\Raw_data\\"
    
    if file_.endswith("csv"):
        name = str(path + file_)
        new_df = pd.read_csv(name)
        return new_df
    elif file_.endswith("xlsx") or file_.endswith("xls"): 
        name = str(path + file_)
        new_df = pd.read_excel(name)
        return new_df