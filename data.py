import pandas as pd
import numpy as np
import datetime

def file2df(file_):
    """
    ---What it does---
        + Generates a dataframe from a files: ".csv", ".xlsx" or ".xls".
    ---What it needs---
        + file_: data consisting in structured in rows and columns.
    IMPORTANT! the files need to be stored in the path specified by the function.    
    ---What it returns---
        + new_df: dataframe containing the data from the file.
    """
    #path = "F:\\Programacion\\1.BOOTCAMP\\Proyectos\\Gap_Minder\\Raw_data\\"
    path = "C:\\Users\\34609\\Documents\\Repos Git\\GapMinder_Python\\Raw_data\\"

    if file_.endswith("csv"):
        name = str(path + file_)
        new_df = pd.read_csv(name)
        return new_df
    elif file_.endswith("xlsx") or file_.endswith("xls"): 
        name = str(path + file_)
        new_df = pd.read_excel(name)
        return new_df

def melt_df(df, feature):
    """
    ---What it does---
        + Melts GapMinder raw dfs, to a df with 3 columns: country, year & feature.
    ---What it needs---
        + df: 1st column with countries & next columns with feature values for each country & year.
        + feature: string with the name of the feature.    
    ---What it returns---
        + new_df: dataframe with reordered data.
    """
    new_df = pd.melt(frame = df, id_vars = str(df.columns[0]), 
                     value_vars = df.columns[1:], var_name = "year", 
                     value_name = feature)
    return new_df

def df2file(df):
    """
    ---What it does---
        + Generates a ".csv" file from a df.
    ---What it needs---
        + df: containing data in rows and columns. 
    IMPORTANT! the files would be stored in the path specified by the function and the name should be given WITHOUT the .csv extension.
    """
    name = input("Name of the file WITHOUT .csv")
    #path = str("F:\\Programacion\\1.BOOTCAMP\\Proyectos\\Gap_Minder\\Processed_data\\" + name + ".csv")
    path = str("C:\\Users\\34609\Documents\\Repos Git\\GapMinder_Python\\Processed_data\\" + name + ".csv")
    df.to_csv(path, index = False)

def merge_all (feature_dict, keys):
    """
    ---What it does---
        + Using a dictionary of the csvs to merge, copies sed dictionary and deletes the first element.
          Then, using the dropped element from the original, uses the merger function in the loop to join all df into one using the keys provided.
          Lastly, drops all NaN values.
    ---What it needs---
        + feature_dict: A dictionary of dataframes ().
        + Acess to the merger function.
        + keys: A key or keys, can be string or list of strings.
    ---What it returns---
        + new_df
    """
    feature_dict_keys = list(feature_dict.keys())
    #print(f'Current dfs to merge: {feature_dict_keys}')
    feature_copy = feature_dict.copy()

    to_drop =  list(feature_dict.keys())[0]
    del feature_copy[to_drop]
    new_df = feature_dict[to_drop].copy()
        
    for e in feature_copy.values():
        new_df = merger (left_df = new_df, right_df = e, keys = keys)
    
    new_df = new_df.dropna()
    return new_df

def merger (left_df, right_df, keys):
    """
    ---What it does---
        + Merges two dfs on the selected keys and eliminates NaN values before returning the new df.
    ---What it needs---
        + left_df: A df to merge on the left.
        + right df: A df to merge on the rigth.
        + keys: A key or keys, can be string or list of strings.
    ---What it returns---
        + new_df
    """
    new_df = pd.merge(left = left_df, right = right_df, how = "outer", on = keys)
    return new_df