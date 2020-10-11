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
    path = "C:\\Users\\34609\Documents\\Repos Git\\GapMinder_Python\\Raw_data\\"
    
    if file_.endswith("csv"):
        name = str(path + file_)
        new_df = pd.read_csv(name)
        return new_df
    elif file_.endswith("xlsx") or file_.endswith("xls"): 
        name = str(path + file_)
        new_df = pd.read_excel(name)
        return new_df

def clean_df(df):
    """
    ---What it does---
        + Transforms GapMinder raw data into a df compatible with ML models.
    ---What it needs---
        + df: 1st column with countries & year/values for the feature at each column.   
    ---What it returns---
        + new_df: dataframe with reordered data.
    """
    countries = [] #extracting the countries which data are recorded at the dataframe
    for i in range(df.shape[0]):
        countries.append(df["country"][i])  
    df_transp = df.T
    df_transp.columns = countries #columns rename
    df_transp.drop(df_transp.index[0], inplace = True) #drop the frist row without information

    max_year = int(df.columns[-1]) #extracting the maximun year record available at the dataframe
    actual_year = datetime.datetime.now().year #obtain today's year
    new_df = df_transp[:-(max_year - actual_year)] #drop the information for years after current day
    new_df = new_df.astype(float) #transforming data into float type
    return new_df

def red_df(df):
    """
    ---What it does---
        + Eliminate the rows with NaN records from the given df.
    ---What it needs---
        + df.   
    ---What it returns---
        + new_df.
    """
    new_df = df.dropna()
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
    path = str("C:\\Users\\34609\Documents\\Repos Git\\GapMinder_Python\\Raw_data\\" + name + ".csv")
    df.to_csv(path, index = False)    