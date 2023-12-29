
"""
26 Décembre: Préprocessing
"""
# 0.1- Imports (ya des trucs inutiles au preprocessing)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, RobustScaler, PolynomialFeatures, TargetEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from category_encoders import LeaveOneOutEncoder
from category_encoders.count import CountEncoder
from category_encoders.cat_boost import CatBoostEncoder
import joblib

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import EarlyStopping, LearningRateScheduler

#from pytorch_tabnet.tab_model import TabNetRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
sns.set_theme(style="ticks", palette="pastel")


# GLOBAL VARIABLES
valeurs_uniques = {}
nombre_val_unique={}
quantiles={}
imputers={}
encoders = {}
encoded_cols={}
useless_columns=[]


# GLOBAL FUNCTIONS
def group_fuel_types(category: str):
    if category in ['PETROL/ELECTRIC', 'DIESEL/ELECTRIC']:
        return "HYBRID"
    elif category in ['NG-BIOMETHANE', 'HYDROGEN', 'NG','E85']:
        return "BIO-FUEL"
    elif category in ['PETROL','LPG'] :
        return 'PETROL'
    else:
        return category  

_coefficient_variation= lambda series : series.std()/series.mean()

# 0.2 - Charger la donnée

class Dataset():
    
    def __init__(self,path) :
        self.path=path

    def load_data(self):
        return pd.read_csv(self.path, sep=",",low_memory=False)



from abc import abstractmethod
class Preprocessor():
    
    useless_columns=[]
    nombre_val_unique={}
    
    def __init__(self,data:Dataset,train:bool):
        self.data=data
        self.train=train
    
    # 1- Récupération d'observations
    def recup_electric(self):
        #  ec (cm3)
        self.data.loc[(self.data["Ft"].apply(group_fuel_types)=="ELECTRIC") & (self.data["ec (cm3)"].isna()),"ec (cm3)"] = 0
        #  Fm
        self.data.loc[(self.data["Ft"].apply(group_fuel_types) =="ELECTRIC") & (self.data["Fm"].isna()),"Fm"] = "E"
        self.data.loc[(self.data["Ft"].apply(group_fuel_types) =="HYBRID") & (self.data["Fm"].isna()),"Fm"] = "P"

        # Electric range (km)
        self.data.loc[~(self.data["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (self.data["Electric range (km)"].isna()),"Electric range (km)"] = 0

        #  Fuel consumption 
        self.data.loc[(self.data["Ft"].apply(group_fuel_types) =="ELECTRIC") & (self.data["Fuel consumption "].isna()),"Fuel consumption "] = 0

        #  z (Wh/km)
        self.data.loc[~(self.data["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (self.data["z (Wh/km)"].isna()),"z (Wh/km)"] = 0
        pass

 
    def delete_useless_columns(self):
         
        if self.train==True:
            for col in self.data.columns:
                valeurs_uniques[col]=self.data[col].unique().tolist()
                Preprocessor.nombre_val_unique[col]=self.data[col].nunique()
            
            to_drop1 = [element for element, val in Preprocessor.nombre_val_unique.items() if val <= 1] 
            Preprocessor.useless_columns.extend(to_drop1)

            to_drop2=[col for col in self.data.columns if self.data[col].isna().mean()> 0.5]
            Preprocessor.useless_columns.extend(to_drop2)
            Preprocessor.useless_columns=list(set(Preprocessor.useless_columns))
            self.data.drop(columns=Preprocessor.useless_columns,inplace=True)
            print(f"{Preprocessor.useless_columns} have been deleted on train data")
            
        else:
            try :
                self.data.drop(columns=Preprocessor.useless_columns,inplace=True)
                print(f"{Preprocessor.useless_columns} have been deleted on test data")
            except:
                print("Useless columns are not determined yet on train data.")


    # 3- Traitement des Outliers par Winsorization (on cap la donnée au 5e et 95e quantile)
    def winsorize_outliers(self,column_name:str,lower_percentile=5, upper_percentile=95):
        if self.train==True:
            Q1=np.percentile(self.data[column_name], lower_percentile)
            Q3=np.percentile(self.data[column_name], upper_percentile)
            IQR= Q3-Q1
            quantiles[column_name]= (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            self.data[column_name]=np.clip(self.data[column_name],quantiles[column_name][0],quantiles[column_name][1])
            pass
        else:
            try:
                self.data[column_name]=np.clip(self.data[column_name],quantiles[column_name][0],quantiles[column_name][1])
            except:
                print("Boundaries are not fixed yet.")
            
    def fill_missing_values(self,colname:str):
        
        if self.train==True:
            
            if self.data[colname].dtype in ["float64"]:
                if _coefficient_variation(self.data[colname]) > 0.15 :
                    imputers[colname]=SimpleImputer(missing_values=np.nan,strategy="median")
                else:
                    imputers[colname]=SimpleImputer(missing_values=np.nan,strategy="mean")
            else:
                imputers[colname]=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            
            self.data[colname]=pd.Series(imputers[colname].fit_transform(self.data[colname].to_numpy().reshape(-1,1)).flatten())
            pass
        
        else:
            try:
                self.data[colname]=pd.Series(imputers[colname].transform(self.data[colname].to_numpy().reshape(-1,1)).flatten())
            except:
                print("Imputers not fitted yet on train data.")
                
    def count_encoder(self,col):
        if self.train==True:
            encoders[col]=CountEncoder(handle_unknown='value')
            self.data[col]=encoders[col].fit_transform(self.data[[col]])
        else:
            try:
                self.data[col]=encoders[col].transform(self.data[[col]])
            except:
                print("Encoder not yet fitted")
        pass

    def ohe_encoder(self,col):
        if self.train==True:
            encoders[col] = OneHotEncoder(sparse_output=False, drop='first',handle_unknown='ignore') #sparse = false sinn jsais pas gérer
            encoders[col].fit(self.data[[col]])
            encoded_cols[col]=encoders[col].get_feature_names_out([col])
            self.data[encoded_cols[col]]=encoders[col].transform(self.data[[col]])
            self.data.drop(columns=col, inplace=True)
            pass
        else: 
            try: 
                self.data[encoded_cols[col]]=encoders[col].transform(self.data[[col]])
                self.data.drop(columns=col, inplace=True)
                pass
            except:
                print("Encoders not fitted yet")
