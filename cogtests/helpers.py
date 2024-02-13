import pandas as pd
from scipy import stats as st

def Demographics(working_df: pd.DataFrame):
    """Get demographic descriptive data"""
    subtests = ['age', 'sex', 'education', 'vr_experience', 'typical_sleep', 'depression', 'anxiety', 'stress']

    subtest_dict = {}
    for name in subtests:
        subtest_dict[f"{name.replace('_', ' ').title()}"] = working_df[f"{name}"].values
    
    desc = get_descriptives(subtest_dict)
    
    pretty_print("Demographics", desc)


def pretty_print(name: str, descriptives: dict[dict]):
    """Taking an overall score name and a dictionary containing descriptives for each subscore
    in the form `{subscore1name: {descriptives}, subscore2name: {descriptives},...}`.\n
    Prints a nice string displaying the descriptive stats for each subscore."""
    print_str = f"{name}:\n"

    for (scorename, score_desc) in descriptives.items():
        min, max, mean, stdev = score_desc.values()
        print_str += f"   {scorename}:\n      Min: {min}, Max: {max}\n\
      Mean: {mean}, SD: {stdev}\n"
    
    print(print_str)


def get_descriptives(scores: dict):
    """Taking a dictionary of subscores in the form `{subscore1: [list,of,values], subscore2: [list,of,values],...}`
    returns a dictionary with the same keys but each value is a nested dict of relevant descriptive variables."""
    descriptives = {}
    for (scorename, score) in scores.items():
        descriptives[scorename] = {}
        score_desc = st.describe(score, nan_policy="omit")
        
        descriptives[scorename]["min"] = round(score_desc.minmax[0].__float__(),2)
        descriptives[scorename]["max"] = round(score_desc.minmax[1].__float__(),2)

        descriptives[scorename]["mean"] = round(score_desc.mean.__float__(), 2)
        descriptives[scorename]["stdev"] = round(score_desc.variance.__float__()**0.5, 2)

    return descriptives