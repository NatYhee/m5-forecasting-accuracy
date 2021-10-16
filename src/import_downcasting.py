import pandas as pd
import numpy as np


class import_downcasting:

    def __init__(self, path_filename:str) -> None:
        self._path_filename = path_filename
    

    def __call__(self) -> pd.DataFrame:
        df = pd.read_csv(self._path_filename)
        downcasted_df = self._downcasting(df)
        return downcasted_df 
    

    def _downcasting(df:pd.DataFrame) -> pd.DataFrame:
        """
        Downcasting data to reduce size when importing as dataframe
        """
        cols = df.dtypes.index.tolist()
        types = df.dtypes.values.tolist()
        for i,t in enumerate(types):
            if 'int' in str(t):
                if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                    df[cols[i]] = df[cols[i]].astype(np.int8)
                elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                    df[cols[i]] = df[cols[i]].astype(np.int16)
                elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                    df[cols[i]] = df[cols[i]].astype(np.int32)
                else:
                    df[cols[i]] = df[cols[i]].astype(np.int64)
            elif 'float' in str(t):
                if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                    df[cols[i]] = df[cols[i]].astype(np.float16)
                elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                    df[cols[i]] = df[cols[i]].astype(np.float32)
                else:
                    df[cols[i]] = df[cols[i]].astype(np.float64)
            elif t == np.object:
                if cols[i] == 'date':
                    df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
                else:
                    df[cols[i]] = df[cols[i]].astype('category')
        return df  

