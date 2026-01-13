import pandas as pd
df = pd.read_csv("pd.csv", index_col=0)
df = df.map(lambda x: [float(s) for s in x.split(" ") if s != ""])
print(df)