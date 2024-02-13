import pandas as pd
from helpers import get_descriptives, pretty_print

def NART(working_df: pd.DataFrame):
    pass


def NART_desc(working_df: pd.DataFrame):
    """Print descriptive information for the NART."""
    nart_score = working_df["NART_score"].values
    nart_time = working_df["NART_time"].values

    desc = get_descriptives({"Score": nart_score, "Time": nart_time})
    
    pretty_print("NART", desc)