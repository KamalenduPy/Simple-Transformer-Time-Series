
"""### Process and save the data in data folder as csv format"""

import pandas as pd

bejing=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv')
bejing=bejing.iloc[25:]

bejing.head()

{col:bejing[col].isnull().sum()/bejing.shape[0] for col in bejing.columns}

ch_var=[col for col in bejing.columns if bejing[col].dtype=='object']

bejing[ch_var].value_counts()

one_hot=pd.get_dummies(bejing)

one_hot.head()

## preprocessing

class preprocess():
    def __init__(self,data:pd.DataFrame):
        self.data=data

    def mis_drop_impute_scale(self,id_cols,miss_th=0.8):
        import pandas as pd
        miss_pct={col:self.data[col].isnull().sum()/self.data.shape[0] for col in self.data.columns}
        high_miss=[]
        for i in miss_pct:
            if miss_pct[i]>miss_th:
                high_miss.append(i)
        self.data=self.data.drop(high_miss,axis=1)

        ### encodeing and onehot
        self.data=pd.DataFrame(self.data.fillna(method='bfill'))

        ch_var =[col for col in self.data.columns if self.data[col].dtype=='object' ]
        num_var=[col for col in self.data.columns if col not in ch_var]

        self.data=pd.DataFrame(pd.get_dummies(self.data))

        # ## scaling
        import numpy as np
        mean=np.mean(self.data)
        mean[id_cols]=0
        std=np.std(self.data)
        std[id_cols]=1
        scaled=(self.data-mean)/std
        return(scaled)

scaler=preprocess(bejing)

processed_data=scaler.mis_drop_impute_scale(miss_th=0.7,id_cols=['No','year','day','hour','month'])

processed_data['timestamp']=pd.to_datetime(processed_data[['year','month','day','hour']])
processed_data=processed_data.drop(['year','month','day','hour','No'],axis=1)
processed_data.head()

[processed_data[col].isnull().sum() for col in processed_data.columns ]

processed_data['timestamp'].describe
