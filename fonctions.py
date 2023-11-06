import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

_coefficient_variation= lambda series : series.std()/series.mean()

columns_to_treat=[]
related_to_type=[]
columns_to_delete=['MMS', 'r', 'Ernedc (g/km)', 'De', 'Vf', 'Status','Va','Ve','Enedc (g/km)','IT','Country','Date of registration']
imputers={}
ohe_encoders={}

def _fill_missing_values(colonne : pd.Series) -> None:
    colname=colonne.name
    if colonne.dtype in ["float64"]:
        if _coefficient_variation(colonne) > 0.15 :
            imputers[colname]=SimpleImputer(missing_values=np.nan,strategy="median")
        else:
            imputers[colname]=SimpleImputer(missing_values=np.nan,strategy="mean")
    else:
        imputers[colname]=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputers[colname].fit(colonne.to_numpy().reshape(-1,1))
    pass

def group_fuel_types(category: str):
    if category in ['PETROL/ELECTRIC', 'DIESEL/ELECTRIC']:
        return "HYBRID"
    elif category in ['NG-BIOMETHANE', 'HYDROGEN', 'NG','E85']:
        return "BIO-FUEL"
    elif category in ['PETROL','LPG'] :
        return 'PETROL'
    else:
        return category   

def encode_that_var(colonne: pd.Series):
    column_name=colonne.name
    ohe_encoder = OneHotEncoder(sparse_output=False, drop='first')
    ohe_features = ohe_encoder.fit_transform(colonne.to_frame())
    ohe_encoders[column_name] = ohe_encoder  # Stocker l'encodeur 

    return pd.DataFrame(ohe_features, columns=ohe_encoder.get_feature_names_out([column_name]))


class Dataset():
    
    def __init__(self,path) :
        self.path=path

    def load_data(self):
        return pd.read_csv(self.path, sep=",",low_memory=False)

from abc import abstractmethod
class Preprocessor():

    imputers={}
    ohe_encoders={}

    @abstractmethod
    def last_step(self):
        pass

    @abstractmethod
    def fill_missing_values(self):
        pass

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
    @abstractmethod
    def fill_mass(self):
        pass
    @abstractmethod
    def fill_engine_power(self):
        pass

    @abstractmethod
    def encode_that_var(self):
        pass

class TrainPreprocessor(Preprocessor):
     
    def __init__(self,data : Dataset):
        self.data=data
        pass

    def last_step(self):
        self.data.drop(columns=columns_to_delete, inplace=True)
        selected_columns = [col for col in self.data.columns if col not in ['ID','Ewltp (g/km)','Date of registration']]
        self.data.drop_duplicates(subset=selected_columns, inplace=True)
        pass
    
    def fill_engine_capacity(self):
        _fill_missing_values(self.data["ec (cm3)"])
        self.data.loc[(self.data["Ft"].apply(group_fuel_types)=="ELECTRIC") & (self.data["ec (cm3)"].isna()),"ec (cm3)"] = 0
        self.data["ec (cm3)"]=pd.Series(imputers["ec (cm3)"].transform(self.data["ec (cm3)"].to_numpy().reshape(-1,1)).flatten())
        pass 
    

    def fill_electric_consumption(self):
        _fill_missing_values(self.data["z (Wh/km)"])
        self.data.loc[~(self.data["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (self.data["z (Wh/km)"].isna()),"z (Wh/km)"] = 0
        self.data["z (Wh/km)"]= pd.Series(imputers["z (Wh/km)"].transform(self.data["z (Wh/km)"].to_numpy().reshape(-1,1)).flatten())
        pass

    
    def fill_fuel_consumption(self):
        _fill_missing_values(self.data["Fuel consumption "])
        self.data.loc[(self.data["Ft"].apply(group_fuel_types) =="ELECTRIC") & (self.data["Fuel consumption "].isna()),"Fuel consumption "] = 0
        self.data["Fuel consumption "]= pd.Series(imputers["Fuel consumption "].transform(self.data["Fuel consumption "].to_numpy().reshape(-1,1)).flatten())
        pass
    
    def fill_electric_range(self):
        _fill_missing_values(self.data["Electric range (km)"])
        self.data.loc[~(self.data["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (self.data["Electric range (km)"].isna()),"Electric range (km)"] = 0
        self.data["Electric range (km)"]= pd.Series(imputers["Electric range (km)"].transform(self.data["Electric range (km)"].to_numpy().reshape(-1,1)).flatten())
        pass


    def fill_category_type(self):
        _fill_missing_values(self.data["Ct"])
        self.data["Ct"] = pd.Series(imputers["Ct"].transform(self.data["Ct"].to_numpy().reshape(-1,1)).flatten())
        pass
    
    def fill_wheel_base(self):
        _fill_missing_values(self.data["W (mm)"])
        self.data["W (mm)"] = pd.Series(imputers["W (mm)"].transform(self.data["W (mm)"].to_numpy().reshape(-1,1)).flatten())
        pass
    
    def fill_At_1(self):
        _fill_missing_values(self.data["At1 (mm)"])
        self.data["At1 (mm)"] = pd.Series(imputers["At1 (mm)"].transform(self.data["At1 (mm)"].to_numpy().reshape(-1,1)).flatten())
        pass
    
    def fill_At_2(self):
        _fill_missing_values(self.data["At2 (mm)"])
        self.data["At2 (mm)"] = pd.Series(imputers["At2 (mm)"].transform(self.data["At2 (mm)"].to_numpy().reshape(-1,1)).flatten())
        pass


    def fill_mass(self):
        _fill_missing_values(self.data["m (kg)"])
        self.data["m (kg)"] = pd.Series(imputers["m (kg)"].transform(self.data["m (kg)"].to_numpy().reshape(-1,1)).flatten())
        pass 

    def fill_engine_power(self):
        _fill_missing_values(self.data["ep (KW)"])
        self.data["ep (KW)"] = pd.Series(imputers["ep (KW)"].transform(self.data["ep (KW)"].to_numpy().reshape(-1,1)).flatten())
        pass 

    def encode_that_var(self,column_name:str):
        ohe_encoder = OneHotEncoder(sparse_output=False, drop='first')
        ohe_features = ohe_encoder.fit_transform(self.data[[column_name]])
        ohe_encoders[column_name] = ohe_encoder  # Stocker l'encodeur 

        ohe_features = pd.DataFrame(ohe_features, columns=ohe_encoder.get_feature_names_out([column_name]))

        self.data.drop(columns=column_name, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.data = pd.concat([self.data, ohe_features], axis=1)

        return self.data



class TestPreprocessor(Preprocessor):
    def __init__(self,data : Dataset):
         self.data=data
         pass    
    
    def last_step(self):
        self.data.drop(columns= columns_to_delete, inplace = True)
        pass

    def fill_engine_capacity(self):
        self.data.loc[(self.data["Ft"].apply(group_fuel_types)=="ELECTRIC") & (self.data["ec (cm3)"].isna()),"ec (cm3)"] = 0
        try :
            self.data["ec (cm3)"]=pd.Series(imputers["ec (cm3)"].transform(self.data["ec (cm3)"].to_numpy().reshape(-1,1)).flatten())
            pass        
        except:
            print("Imputer ec (cm3) not fitted yet to the train")


    def fill_electric_consumption(self):
        self.data.loc[~(self.data["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (self.data["z (Wh/km)"].isna()),"z (Wh/km)"] = 0
        try :
            self.data["z (Wh/km)"]=pd.Series(imputers["z (Wh/km)"].transform(self.data["z (Wh/km)"].to_numpy().reshape(-1,1)).flatten())
            pass
        except:
            print("Imputer z (Wh/km) not fitted yet to the train")


    def fill_fuel_consumption(self):
        self.data.loc[(self.data["Ft"].apply(group_fuel_types) =="ELECTRIC") & (self.data["Fuel consumption "].isna()),"Fuel consumption "] = 0
        try:
            self.data["Fuel consumption "]=pd.Series(imputers["Fuel consumption "].transform(self.data["Fuel consumption "].to_numpy().reshape(-1,1)).flatten())
            pass
        except : 
            print("Imputer Fuel consumption not fitted yet to the train")


    def fill_electric_range(self):
        self.data.loc[~(self.data["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (self.data["Electric range (km)"].isna()),"Electric range (km)"] = 0
        try:
            self.data["Electric range (km)"]=pd.Series(imputers["Electric range (km)"].transform(self.data["Electric range (km)"].to_numpy().reshape(-1,1)).flatten())
            pass
        except:
            print("Imputer Electric range (km) not fitted yet to the train")

    def fill_category_type(self):
        try:
            self.data["Ct"] =pd.Series(imputers["Ct"].transform(self.data["Ct"].to_numpy().reshape(-1,1)).flatten())
            pass
        except:
            print("Imputer Ct not fitted yet to the train")

    def fill_wheel_base(self):
        try:
            self.data["W (mm)"] =pd.Series(imputers["W (mm)"].transform(self.data["W (mm)"].to_numpy().reshape(-1,1)).flatten())
            pass
        except:
            print("Imputer W (mm) not fitted yet to the train")

    def fill_At_1(self):
        try:
            self.data["At1 (mm)"] =pd.Series(imputers["At1 (mm)"].transform(self.data["At1 (mm)"].to_numpy().reshape(-1,1)).flatten())
            pass
        except:
            print("Imputer At1 (mm)not fitted yet to the train")

    def fill_At_2(self):
        try:
            self.data["At2 (mm)"] =pd.Series(imputers["At2 (mm)"].transform(self.data["At2 (mm)"].to_numpy().reshape(-1,1)).flatten())
            pass
        except:
            print("Imputer At2 (mm) not fitted yet to the train")

    def fill_mass(self):
        try:
            self.data["m (kg)"] =pd.Series(imputers["m (kg)"].transform(self.data["m (kg)"].to_numpy().reshape(-1,1)).flatten())
            pass
        except:
            print("Imputer m (kg) not fitted yet to the train")
    def fill_engine_power(self):
        try:
            self.data["ep (KW)"] =pd.Series(imputers["ep (KW)"].transform(self.data["ep (KW)"].to_numpy().reshape(-1,1)).flatten())
            pass
        except:
            print("Imputer ep (KW) not fitted yet to the train")

    def encode_that_var(self,column_name:str):
        try:
            ohe_encoder = ohe_encoders[column_name]
            ohe_features = ohe_encoder.transform(self.data[[column_name]])
            ohe_features = pd.DataFrame(ohe_features, columns=ohe_encoder.get_feature_names_out([column_name]))

            self.data.drop(columns=column_name, inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            self.data = pd.concat([self.data, ohe_features], axis=1)
            return self.data
        except:
            print(f"no encoders try encoding {column_name} in train first")