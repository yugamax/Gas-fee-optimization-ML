import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"main.eth.csv", on_bad_lines='skip')
print(df.dtypes)
print(df.shape)
df.drop(['name'], axis=1, inplace=True)
print(df.shape)

df.dropna(subset=["time", "hash"], inplace=True)
df.reset_index(drop=True, inplace=True)

string_cols = [col for col in df.select_dtypes(include=['object']).columns if col not in ['hash']]

le = LabelEncoder()
for col in string_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df.to_csv("cleaned_main.eth.csv", index=False)