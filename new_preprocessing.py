import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, RobustScaler

_coefficient_variation= lambda series : series.std()/series.mean()

columns_to_treat=[]
related_to_type=[]
columns_to_delete=['MMS', 'r', 'Ernedc (g/km)', 'De', 'Vf', 'Status','Va','Ve','Enedc (g/km)','IT','Date of registration'] #,'Country'
imputers={}
ohe_encoders={}
label_encoders={}
ordinal_encoders={}
boundaries={}
normalise= {}

def _is_outlier(colonne : pd.Series) -> pd.Series:
        Q1=colonne.quantile(q=0.25)
        Q3=colonne.quantile(q=0.75)
        IQR=Q3-Q1
        boundaries[colonne.name]=(Q1-1.5*IQR,Q3+1.5*IQR)
        out_col=colonne.apply(lambda x: 1 if (((x!=np.nan) & (x>boundaries[colonne.name][1]) | (x<boundaries[colonne.name][0]))) else 0)
        return out_col


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


class Preprocessor():

    def __init__(self,data : Dataset):
        self.data=data
        pass

    def last_step(self):
        self.data.drop(columns=columns_to_delete, inplace=True)
        #selected_columns = [col for col in self.data.columns if col not in ['ID','Ewltp (g/km)','Date of registration']]
        #self.data.drop_duplicates(subset=selected_columns, inplace=True)
        pass

    def fill_vehicule_family_number(self):
        self.data["VFN"]=self.data["VFN"].fillna("UNKNOWN")
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

    def fill_test_mass(self):
        _fill_missing_values(self.data["Mt"])
        self.data["Mt"] = pd.Series(imputers["Mt"].transform(self.data["Mt"].to_numpy().reshape(-1,1)).flatten())
        pass

    def fill_fuel_mode(self):
        _fill_missing_values(self.data["Fm"])
        self.data.loc[(self.data["Ft"].apply(group_fuel_types) =="ELECTRIC") & (self.data["Fm"].isna()),"Fm"] = "E"
        self.data.loc[(self.data["Ft"].apply(group_fuel_types) =="HYBRID") & (self.data["Fm"].isna()),"Fm"] = "P"
        self.data["Fm"]= pd.Series(imputers["Fm"].transform(self.data["Fm"].to_numpy().reshape(-1,1)).flatten())
        pass


    def ohe_that_var(self,column_name:str):
        ohe_encoder = OneHotEncoder(sparse_output=False, drop='first')
        ohe_features = ohe_encoder.fit_transform(self.data[[column_name]])
        ohe_encoders[column_name] = ohe_encoder  # Stocker l'encodeur 

        ohe_features = pd.DataFrame(ohe_features, columns=ohe_encoder.get_feature_names_out([column_name]))

        self.data.drop(columns=column_name, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.data = pd.concat([self.data, ohe_features], axis=1)

        return self.data
    
    def outlier_detection(self,colname:str):
          self.data[f"flag_{colname}"]=_is_outlier(self.data[colname])
          pass
    
    def encode_country(self):
        label_encoders["Country"]=LabelEncoder()
        self.data["Country"]=label_encoders["Country"].fit_transform(self.data["Country"])
        pass


    def encode_manufacture_pooling(self):
        label_encoders['Mp']=LabelEncoder()
        self.data["Mp"].fillna("UNKNOWN", inplace=True)
        self.data["Mp"]=label_encoders["Mp"].fit_transform(self.data['Mp'])
        pass

    def encode_fuel_mode(self):
        ordinal_encoders["Fm"]=OrdinalEncoder()
        self.data["Fm"]=ordinal_encoders["Fm"].fit_transform(self.data[["Fm"]])
        pass

    def encode_category_type(self):
        ordinal_encoders["Ct"]=OrdinalEncoder()
        self.data["Ct"]=ordinal_encoders["Ct"].fit_transform(self.data[["Ct"]])
        pass
    
    def encode_category_registered(self):
        ordinal_encoders["Cr"]=OrdinalEncoder()
        self.data["Cr"]=ordinal_encoders["Cr"].fit_transform(self.data[["Cr"]])
        pass

    def encode_fuel_type(self):
        ordinal_encoders["Ft"]=OrdinalEncoder()
        self.data["Ft"]=pd.Series(ordinal_encoders["Ft"].fit_transform(self.data["Ft"].apply(group_fuel_types).to_numpy().reshape(-1,1)).flatten()) 
        pass

    def encode_manufacturer_name(self):
        label_encoders["Man"]=LabelEncoder()
        self.data['Man']= label_encoders['Man'].fit_transform(self.data['Man'].astype(str))
        pass



    def robust_normalise(self,colname:str):
          normalise[colname]=RobustScaler()
          self.data[colname] = pd.Series(normalise[colname].fit_transform(self.data[colname].to_numpy().reshape(-1,1)).flatten())
          pass
    
    def encode_vehicule_family_number(self):
        label_encoders["VFN"]=LabelEncoder()
        self.data["VFN"]=label_encoders["VFN"].fit_transform(self.data["VFN"])
        pass
