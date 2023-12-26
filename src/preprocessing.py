import pandas as pd

def display_missing_values(df:pd.DataFrame)->pd.DataFrame:
    """
    Afficher dans un dataframe par ordre décroissant les colonnes,
    le pourcentage et le nombre de valeurs manquantes qu'elles possèdent

    Parameters:
    -----------
    pd.DataFrame

    Returns:
    --------
    pd.DataFrame
        Data frame avec les NaN correctement trouvés
    """
    dico_nan = []

    for col in df.columns:
        dico_nan = dico_nan + [{"Colonne" : col,"pourcentage manquant" : df[col].isna().sum()/len(df)*100 ,"nombre" : df[col].isna().sum()}] 
    dico_nan.sort(key = lambda x: x.get('pourcentage manquant'),reverse = True)

    return pd.DataFrame(dico_nan)

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

from pytorch_tabnet.tab_model import TabNetRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
sns.set_theme(style="ticks", palette="pastel")


# 0.2 - Charger la donnée

data_train= pd.read_csv("data/train.csv",sep=",",low_memory=False)
data_test = pd.read_csv("data/test.csv",sep=",",low_memory=False)

data_train.name="data_train"
data_test.name="data_test"

# 1- Récupération d'observations

def group_fuel_types(category: str):
    if category in ['PETROL/ELECTRIC', 'DIESEL/ELECTRIC']:
        return "HYBRID"
    elif category in ['NG-BIOMETHANE', 'HYDROGEN', 'NG','E85']:
        return "BIO-FUEL"
    elif category in ['PETROL','LPG'] :
        return 'PETROL'
    else:
        return category   

def recup_electric(df):
    #  ec (cm3)
    df.loc[(df["Ft"].apply(group_fuel_types)=="ELECTRIC") & (df["ec (cm3)"].isna()),"ec (cm3)"] = 0
    #  Fm
    df.loc[(df["Ft"].apply(group_fuel_types) =="ELECTRIC") & (df["Fm"].isna()),"Fm"] = "E"
    df.loc[(df["Ft"].apply(group_fuel_types) =="HYBRID") & (df["Fm"].isna()),"Fm"] = "P"

    # Electric range (km)
    df.loc[~(df["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (df["Electric range (km)"].isna()),"Electric range (km)"] = 0

    #  Fuel consumption 
    df.loc[(df["Ft"].apply(group_fuel_types) =="ELECTRIC") & (df["Fuel consumption "].isna()),"Fuel consumption "] = 0

    #  z (Wh/km)
    df.loc[~(df["Ft"].apply(group_fuel_types).isin(["ELECTRIC", "HYBRID"])) & (df["z (Wh/km)"].isna()),"z (Wh/km)"] = 0
    pass

recup_electric(data_train)
recup_electric(data_test)

# 2- Delete useless columns 

valeurs_uniques = {}
nombre_val_unique={}
for col in data_train.columns:
    valeurs_uniques[col]=data_train[col].unique().tolist()
    nombre_val_unique[col]=data_train[col].nunique()

for element in nombre_val_unique:
    if nombre_val_unique[element]<=1:
        print(f"colonne supprimée: {element}")
        data_train.drop(columns=element, inplace=True)
        data_test.drop(columns=element, inplace=True)

for col in data_train.columns:
    if (data_train[col].isna().sum()/data_train.shape[0] > 0.5):
        print(f"colonne supprimée: {col}")
        data_train.drop(columns=col, inplace=True)
        data_test.drop(columns=col, inplace=True)

data_train.drop(columns=['Date of registration','ID'], inplace=True)
data_test.drop(columns='Date of registration', inplace=True)

print(f"colonne supprimée pour data_train: Date of registration, ID")
print(f"colonne supprimée pour data_test: Date of registration")

col_categoricals = data_test.select_dtypes(include="object").columns.tolist()
col_numericals = [col for col in data_test.columns if col not in col_categoricals]
col_numericals.remove("ID")

# 3- Traitement des Outliers par Winsorization (on cap la donnée au 5e et 95e quantile)

quantiles={}

def winsorize_outliers(data, column_name, lower_percentile=5, upper_percentile=95,train=True):
    """
    Detects and imputes outliers using winsorizing for a specific column in a DataFrame.

    Parameters:
    - data: Pandas DataFrame, input data
    - column_name: str, name of the column to be winsorized
    - lower_percentile: int, lower percentile for winsorizing (default: 5)
    - upper_percentile: int, upper percentile for winsorizing (default: 95)

    Returns:
    - winsorized_data: Pandas DataFrame, data with outliers winsorized for the specified column
    """

    column_data = data[column_name]
    if train:
        quantiles["q1"] = np.percentile(column_data, lower_percentile)
        quantiles["q3"] = np.percentile(column_data, upper_percentile)
        iqr = quantiles["q3"] - quantiles["q1"]
        quantiles["lower_bound"] = quantiles["q1"] - 1.5 * iqr
        quantiles["upper_bound"] = quantiles["q3"] + 1.5 * iqr

    data[column_name] = np.clip(column_data, quantiles["lower_bound"], quantiles["upper_bound"])

    return data

for col in col_numericals:
    data_train=winsorize_outliers(data_train,col)
    data_test =winsorize_outliers(data_test,col,train=False)

# 4- NaN imputation

imputers={}
_coefficient_variation= lambda series : series.std()/series.mean()

def fill_missing_values(colname : str,data:pd.DataFrame) -> None:
    
    if data[colname].dtype in ["float64"]:
        if _coefficient_variation(data[colname]) > 0.15 :
            imputers[colname]=SimpleImputer(missing_values=np.nan,strategy="median")
        else:
            imputers[colname]=SimpleImputer(missing_values=np.nan,strategy="mean")
    else:
        imputers[colname]=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputers[colname].fit(data[colname].to_numpy().reshape(-1,1))
    pass

for col in data_test.columns[1:]:
    fill_missing_values(col,data_train)
    data_train[col]=pd.Series(imputers[col].transform(data_train[col].to_numpy().reshape(-1,1)).flatten())
    data_test[col]=pd.Series(imputers[col].transform(data_test[col].to_numpy().reshape(-1,1)).flatten())

# 5- Custom Encoding 
"""
Mélange de Count Encoder (nunique >=15) et OHE (else).

Il y a les catboost et ordinal encoder avec si tu veux mais la combinaison que j'ai trouvé et plus efficace
""" 

encoders = {}

def cat_boost_encoder(col,df,train=True):
    if train:
        encoders[col]= CatBoostEncoder(random_state=42)
        df[col]=encoders[col].fit_transform(df[[col]],df[['Ewltp (g/km)']])
    else:
        df[col]=encoders[col].transform(df[[col]])
    pass

def ohe_encoder(col,df,train=True):
    if train:
        encoders[col] = OneHotEncoder(sparse_output=False, drop='first',handle_unknown='ignore') #sparse = false sinn jsais pas gérer
        ohe_features = encoders[col].fit_transform(df[[col]])
    else: 
        ohe_features = encoders[col].transform(df[[col]])

    ohe_features = pd.DataFrame(ohe_features, columns=encoders[col].get_feature_names_out([col]))

    df.drop(columns=col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, ohe_features], axis=1)
    return df

def count_encoder(col, df, train=True):
    if train:
        encoders[col]=CountEncoder(handle_unknown='value')
        df[col]=encoders[col].fit_transform(df[[col]])
    else:
        df[col]=encoders[col].transform(df[[col]])
    pass

def ordinal_encoder(colname:str,data:pd.DataFrame,train=True):
    if train:
        encoders[colname]=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        data[colname]=encoders[colname].fit_transform(data[[colname]])
    else:
        data[colname]=encoders[colname].transform(data[[colname]])
    pass

for col in col_categoricals:
    if nombre_val_unique[col]>=15: #eventually replace by catboost encoder but careful cuz of Target ! (do TTS first)
        count_encoder(col,data_train)
        count_encoder(col,data_test,False)
        print(f"encoding count : {col}")
    else:
        data_train=ohe_encoder(col,data_train) #reassign cuz you don't know how to do it...
        data_test=ohe_encoder(col,data_test, False)
        print(f"encoding OHE : {col}")

# 6- Split
        
train, test = train_test_split(data_train, test_size=0.33, random_state=42)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

X_train, y_train = train.drop(columns=["Ewltp (g/km)"]), train["Ewltp (g/km)"]
X_test, y_test = test.drop(columns=["Ewltp (g/km)"]), test["Ewltp (g/km)"]