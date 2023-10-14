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

