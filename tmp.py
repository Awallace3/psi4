import pandas as pd
from pprint import pprint

df = pd.read_pickle("./adamantane_disp_xdm_errors.pkl")
print(df)
pprint(df.columns.tolist())
