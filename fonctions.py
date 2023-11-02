from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


_coefficient_variation= lambda series : series.std()/series.mean()

columns_to_treat=[]
related_to_type=[]

imputers={}

def fill_missing_values(colonne : pd.Series):
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

# TODO A supprimer nan ? Utilisé nulle part + no return

def fill_engine_capacity(colonne_category,colonne_engine_capacity):
    fill_missing_values(colonne_engine_capacity)
    if (colonne_category=="ELECTRIC") & (colonne_engine_capacity== np.nan):
        colonne_engine_capacity=0
    

class Dataset():
    
    def __init__(self,path) :
        self.path=path

    def load_data(self):
        return pd.read_csv(self.path, sep=",",low_memory=False)
    
    


from abc import abstractmethod
class Preprocessor():

    imputers={}
    encoders={}

    @abstractmethod
    def fill_missing_values(self):
        pass

    @abstractmethod
    def fill_engine_capicity(self):
        pass

# TODO Supprimer, il y est déjà 
    @abstractmethod
    def fill_engine_capacity(self):
        pass

# TODO Supprimer, il y est déjà
    @abstractmethod
    def fill_engine_capacity(self):
        pass
    
    @abstractmethod
    def fill_electric_consumption(self):
        pass
    
    @abstractmethod
    def fill_fuel_consumption(self):
        pass

    @abstractmethod
    def fill_electric_range(self):
        pass

    @abstractmethod
    def fill_type_approval_number(self):
        pass

    @abstractmethod
    def fill_category_type(self):
        pass

    @abstractmethod
    def fill_wheel_base(self):
        pass

    @abstractmethod
    def fill_At_1(self):
        pass
    
    @abstractmethod
    def fill_At_2(self):
        pass


class TrainPreprocessor(Preprocessor):
     
    def __init__(self,data : Dataset):
         self.data=data
         pass
     
    def fill_missing_values(self,colname: str):
        #colname=colonne.name
        if self.data[colname].dtype in ["float64"]:
            if _coefficient_variation(self.data[colname]) > 0.15 :
                imputers[colname]=SimpleImputer(missing_values=np.nan,strategy="median")
            else:
                imputers[colname]=SimpleImputer(missing_values=np.nan,strategy="mean")
        else:
            imputers[colname]=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        return pd.Series(imputers[colname].fit_transform(self.data[colname].to_numpy().reshape(-1,1)).flatten())
    
    def fill_engine_capacity(self):
        fill_missing_values(self.data["ec (cm3)"])
        self.data[(self.data["Ft"].apply(group_fuel_types)=="ELECTRIC") & (self.data["ec (cm3)"].isna())] = 0
        return pd.Series(imputers["ec (cm3)"].transform(self.data["ec (cm3)"].to_numpy().reshape(-1,1)).flatten())
    
    def fill_electric_consumption(self):
        fill_missing_values(self.data["z (Wh/km)"])
        self.data[~(self.data["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (self.data["z (Wh/km)"].isna())] = 0
        return pd.Series(imputers["z (Wh/km)"].transform(self.data["z (Wh/km)"].to_numpy().reshape(-1,1)).flatten())
    
    def fill_fuel_consumption(self):
        fill_missing_values(self.data["Fuel consumption "])
        self.data[(self.data["Ft"].apply(group_fuel_types) =="ELECTRIC") & (self.data["Fuel consumption "].isna())] = 0
        return pd.Series(imputers["Fuel consumption "].transform(self.data["Fuel consumption "].to_numpy().reshape(-1,1)).flatten())
    
    def fill_electric_range(self):
        fill_missing_values(self.data["Electric range (km)"])
        self.data[~(self.data["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (self.data["Electric range (km)"].isna())] = 0
        return pd.Series(imputers["Electric range (km)"].transform(self.data["Electric range (km)"].to_numpy().reshape(-1,1)).flatten())
    
    def fill_type_approval_number(self):
        fill_missing_values(self.data["Tan"])
        return pd.Series(imputers["Tan"].transform(self.data["Tan"].to_numpy().reshape(-1,1)).flatten())
    
    def fill_category_type(self):
        fill_missing_values(self.data["Ct"])
        return pd.Series(imputers["Ct"].transform(self.data["Ct"].to_numpy().reshape(-1,1)).flatten())
    
    def fill_wheel_base(self):
        fill_missing_values(self.data["W (mm)"])
        return pd.Series(imputers["W (mm)"].transform(self.data["W (mm)"].to_numpy().reshape(-1,1)).flatten())
    
    def fill_At_1(self):
        fill_missing_values(self.data["At1 (mm)"])
        return pd.Series(imputers["At1 (mm)"].transform(self.data["At1 (mm)"].to_numpy().reshape(-1,1)).flatten())
    
    def fill_At_2(self):
        fill_missing_values(self.data["At2 (mm)"])
        return pd.Series(imputers["At2 (mm)"].transform(self.data["At2 (mm)"].to_numpy().reshape(-1,1)).flatten())
    

class TestPreprocessor(Preprocessor):
    def __init__(self,data : Dataset):
         self.data=data
         pass
     
    def fill_missing_values(self,colname: str):
        #colname=colonne.name
        try :
            return pd.Series(imputers[colname].transform(self.data[colname].to_numpy().reshape(-1,1)).flatten())
        except:
            print("Imputer not fitted yet to the train")
    pass

    def fill_engine_capacity(self):
        self.data[(self.data["Ft"].apply(group_fuel_types)=="ELECTRIC") & (self.data["ec (cm3)"].isna())] = 0
        try :
            return pd.Series(imputers["ec (cm3)"].transform(self.data["ec (cm3)"].to_numpy().reshape(-1,1)).flatten())
        except:
            print("Imputer not fitted yet to the train")


    def fill_electric_consumption(self):
        self.data[~(self.data["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (self.data["z (Wh/km)"].isna())] = 0
        try :
            return pd.Series(imputers["z (Wh/km)"].transform(self.data["z (Wh/km)"].to_numpy().reshape(-1,1)).flatten())
        except:
            print("Imputer not fitted yet to the train")


    def fill_fuel_consumption(self):
        self.data[(self.data["Ft"].apply(group_fuel_types) =="ELECTRIC") & (self.data["Fuel consumption "].isna())] = 0
        try:
            return pd.Series(imputers["Fuel consumption "].transform(self.data["Fuel consumption "].to_numpy().reshape(-1,1)).flatten())
        except : 
            print("Imputer not fitted yet to the train")


    def fill_electric_range(self):
        self.data[~(self.data["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (self.data["Electric range (km)"].isna())] = 0
        try:
            return pd.Series(imputers["Electric range (km)"].transform(self.data["Electric range (km)"].to_numpy().reshape(-1,1)).flatten())
        except:
            print("Imputer not fitted yet to the train")

    def fill_type_approval_number(self):
        try:
            return pd.Series(imputers["Tan"].transform(self.data["Tan"].to_numpy().reshape(-1,1)).flatten())
        except:
            print("Imputer not fitted yet to the train")

    def fill_category_type(self):
        try:
            return pd.Series(imputers["Ct"].transform(self.data["Ct"].to_numpy().reshape(-1,1)).flatten())
        except:
            print("Imputer not fitted yet to the train")

    def fill_wheel_base(self):
        try:
            return pd.Series(imputers["W (mm)"].transform(self.data["W (mm)"].to_numpy().reshape(-1,1)).flatten())
        except:
            print("Imputer not fitted yet to the train")

    def fill_At_1(self):
        try:
            return pd.Series(imputers["At1 (mm)"].transform(self.data["At1 (mm)"].to_numpy().reshape(-1,1)).flatten())
        except:
            print("Imputer not fitted yet to the train")

    def fill_At_2(self):
        try:
            return pd.Series(imputers["At2 (mm)"].transform(self.data["At2 (mm)"].to_numpy().reshape(-1,1)).flatten())
        except:
            print("Imputer not fitted yet to the train")
