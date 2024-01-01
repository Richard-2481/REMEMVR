import pandas as pd

def descriptive_analysis(df):
    print(df.describe().round(2))
