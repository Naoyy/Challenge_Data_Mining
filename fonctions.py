from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


_coefficient_variation= lambda series : series.std()/series.mean()

columns_to_treat=[]
related_to_type=[]

imputers={}

def _fill_missing_values(colonne : pd.Series) -> pd.Series:
    colname=colonne.name
    if colonne.dtype in ["float64"]:
        if _coefficient_variation(colonne) > 0.15 :
            imputers[colname]=SimpleImputer(missing_values=np.nan,strategy="median")
        else:
            imputers[colname]=SimpleImputer(missing_values=np.nan,strategy="mean")
    else:
        imputers[colname]=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    return pd.Series(imputers[colname].fit_transform(colonne.to_numpy().reshape(-1,1)).flatten())
    

def group_fuel_types(category: str):
    if category in ['PETROL/ELECTRIC', 'DIESEL/ELECTRIC']:
        return "HYBRID"
    elif category in ['NG-BIOMETHANE', 'HYDROGEN', 'NG','E85']:
        return "BIO-FUEL"
    else :
        return category

