import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

pd.set_option('display.max_columns', None)

mean_imputer= SimpleImputer(missing_values=np.nan, strategy='mean')

def basic_preprocessing_keep_numericals (df: pd.DataFrame,train=True)->pd.DataFrame:
    """
    Principle:
    ---------
    La fonction applique un préprocessing normal qu'on utilisera
    pour la régression linéaire. C'est le préprocessing basique. On 
    ne garde que les colonnes numériques

    On effectue: 
    - Analyse des doublons purs
    - drop colonnes inutiles `["MMS","r","Enedc (g/km)","Ernedc (g/km)","Erwltp (g/km)","De","Vf"]`
    - imputation des Nan:  
        - par la moyenne pour les valeurs non problèmatiques
        - par 0 pour les variables correspondent à des critères seulement présent chez les voitures électriques. 
        (ex: consommation electrique `z (Wh/km)` n'a pas lieu d'être sur une voiture essence)

    Parameters:
    -----------
    df: pd.DataFrame

    Returns:
    --------
    pd.DataFrame
    
    Data frame avec les données correctement processed et 
    prêtes à l'emploi (ready to train)
    """
    col_categoricals = df.select_dtypes(include="object").columns.tolist()

    #  drop useless col
    colonnes_a_drop = ["MMS","r","Enedc (g/km)","Ernedc (g/km)","Erwltp (g/km)","De","Vf"]
    df_pp= df.drop(columns=colonnes_a_drop)

    #  keep only numerical
    col_numericals_2 = [col for col in df_pp.columns if col not in col_categoricals]
    df_pp= df_pp[col_numericals_2]

    #  drop dups
    if train:
        df_pp.drop_duplicates(inplace=True)

    #  Mean impute
    colonnes_mean_impute=['m (kg)','W (mm)','ep (KW)','At1 (mm)','At2 (mm)','Mt','ec (cm3)']
    df_pp[colonnes_mean_impute]= mean_imputer.fit_transform(df_pp[colonnes_mean_impute])

    #  Impute specific
    fill_values = {'Fuel consumption ':0, 'z (Wh/km)': 0,'Electric range (km)':0}
    df_pp.fillna(fill_values,inplace=True)

    return df_pp

######################################
#          Preprocessing 1           #
#                                    #
######################################


ohe_encoders = {}

def preprocess_and_encode(df: pd.DataFrame, column_name: str, is_train=True) -> pd.DataFrame:
    global ohe_encoders

    if is_train:
        mode_value = df[column_name].mode().iloc[0]
        df[column_name].fillna(mode_value, inplace=True)

        ohe_encoder = OneHotEncoder(sparse=False, drop='first')
        ohe_features = ohe_encoder.fit_transform(df[[column_name]])
        ohe_encoders[column_name] = ohe_encoder  # Stocker l'encodeur 
    else: # df= Test 

        mode_value = df[column_name].mode().iloc[0]
        df[column_name].fillna(mode_value, inplace=True)

        ohe_encoder = ohe_encoders[column_name]

        ohe_features = ohe_encoder.transform(df[[column_name]])

    ohe_features = pd.DataFrame(ohe_features, columns=ohe_encoder.get_feature_names_out([column_name]))

    df.drop(columns=column_name, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, ohe_features], axis=1)

    return df

def compute_surface(obs):
    max_largeur= max(obs['At1 (mm)'], obs['At2 (mm)'])
    return obs['W (mm)']*obs['At1 (mm)'] if max_largeur == obs['At1 (mm)'] else obs['W (mm)'] * obs['At2 (mm)']


def preprocessing_1(df:pd.DataFrame,is_train=True)->pd.DataFrame: 
    """
    Principle:
    ---------
    La fonction applique un préprocessing plus poussé avec encoding des valeurs 
    catégorielles, Feature Engineering  etc.

    On effectue: 
    - (1) Suppression des doublons impurs
    - (2) Suppression des colonnes inutiles (100% NaN ou 1 valeur unique)
    - (3) Traitement var TAN (création col `conforme`)
    - (4) OHE des var `Ct` et `Cr`
    - (5) Mean impute `W (mm)`, `At1 (mm)`, `At2 (mm)` + Création de la colonne `Surface`
    - (6) 
    - (7) 
    - (8) 
    - (9) 

    Parameters:
    -----------
    df: pd.DataFrame
    is_train: bool

    Returns:
    --------
    pd.DataFrame
    
    Data frame avec les données correctement processed et 
    prêtes à l'emploi (ready to train)
    """
    valeurs_uniques = {}
    nombre_val_unique={}
    for col in df.columns:
        valeurs_uniques[col]=df[col].unique().tolist()
        nombre_val_unique[col]=df[col].nunique()
#  (1)
    if is_train: # se débarasser des duplicates
        selected_columns = [col for col in df.columns if col not in ['ID','Ewltp (g/km)','Date of registration']]
        df.drop_duplicates(subset=selected_columns, inplace=True)
#  (2)
    for element in nombre_val_unique:
        if nombre_val_unique[element]<=1:
            df.drop(columns=element, inplace=True)
#  (3)
    df['conforme'] = df['Tan'].isna()
    df['conforme'] = df['conforme'].apply(lambda x: 1 if x==False else 0)
    df.drop(columns='Tan', inplace=True)
#  (4)
    if is_train:
        df = preprocess_and_encode(df, 'Ct')
        df = preprocess_and_encode(df, 'Cr')
    else: 
        df= preprocess_and_encode(df, 'Ct',False)
        df= preprocess_and_encode(df, 'Cr',False)
#  (5)
    mean_imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
    colonnes_mean_impute=['W (mm)','At1 (mm)','At2 (mm)']
    df[colonnes_mean_impute]= mean_imputer.fit_transform(df[colonnes_mean_impute])
    df['surface']= df.apply(compute_surface, axis=1)


    return df
