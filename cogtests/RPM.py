import pandas as pd
from helpers import get_descriptives, pretty_print

def RPM(working_df: pd.DataFrame):
    pass


def RPM_desc(working_df: pd.DataFrame):
    """Compare normative data for the RPM."""
    rpm_score = working_df["RPM_score"].values
    rpm_time = working_df["RPM_time"].values

    desc = get_descriptives({"Score": rpm_score, "Time": rpm_time})
    
    pretty_print("RPM", desc)