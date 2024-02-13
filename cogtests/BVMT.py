import pandas as pd
from cogtests.helpers import get_descriptives, pretty_print


def BVMT(working_df: pd.DataFrame, print_out: list|str|None = None):
    BVMT_Ts = BVMT_T_Scores(working_df)


def BVMT_T_Scores(working_df: pd.DataFrame):
    pass


def BVMT_desc(working_df: pd.DataFrame):
    """Compare normative data for the BVMT."""
    subtests = ['total_recall', 'learning', 'percent_recalled', 'recognition_hits',
    'recognition_false_alarms', 'recognition_discrimination_index', 'recognition_response_bias']
    subtest_dict = {}
    for name in subtests:
        subtest_dict[f"{name.replace('_', ' ').title()} Score"] = working_df[f"BVMT_{name}"].values
    
    desc = get_descriptives(subtest_dict)
    
    pretty_print("BVMT", desc)