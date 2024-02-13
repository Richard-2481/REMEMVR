import pandas as pd
from scipy import stats as st
import os, sys, inspect, importlib
import json


def __init__():
    """Load working dataframe or create it if it doesn't exist"""
    global working_df

    if os.path.exists("./cogtests/working_df.db"):
        print("Loaded working DataFrame!")
        working_df = pd.read_pickle("./cogtests/working_df.db")
    else:
        # Modify current path so that modules from parent folder can be imported
        current_dir = inspect.getfile( inspect.currentframe() )
        lib_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
        sys.path.append(lib_path)

        from main import startup
        import data

        # Get data
        master_df, working_df, _ = startup()

        with open('data/variable_list.json', 'r') as file:
            variables_json = json.load(file)

        # Add cognitive testing variables to the working dataframe
        for variable in variables_json:
            if variable["group"] == "cognitive":
                working_df = data.add(variable, working_df, master_df)
        
        # Save the working DataFrame to a file for quicker loading next time
        pd.to_pickle(working_df, "./cogtests/working_df.db")

        print("Saved working DataFrame!")

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

   
def NART():
    """Compare normative data for the NART."""
    nart_score = working_df["NART_score"].values
    nart_time = working_df["NART_time"].values

    desc = get_descriptives({"Score": nart_score, "Time": nart_time})
    
    pretty_print("Nart", desc)


def RAVLT():
    """Compare normative data for the RAVLT."""
    subtests = ['trial_1','trial_5','trial_distraction','trial_free_recall','trial_delayed_recall']
    # Not included: 'trial_2','trial_3','trial_4'
    subtest_dict = {}
    for name in subtests:
        subtest_dict[f"{name.replace('_', ' ').title()} Score"] = working_df[f"RAVLT_{name}_score"].values
        # Not included: subtest_dict[f"{name.replace('_', ' ').title()} Time"] = working_df[f"RAVLT_{name}_time"].values

    subtest_dict["Delay Duration Time"] = working_df[f"RAVLT_delayed_recall_duration"].values

    desc = get_descriptives(subtest_dict)
    
    pretty_print("RAVLT", desc)


def BVMT():
    """Compare normative data for the BVMT."""
    subtests = ['total_recall', 'learning', 'percent_recalled', 'recognition_hits',
    'recognition_false_alarms', 'recognition_discrimination_index', 'recognition_response_bias']
    subtest_dict = {}
    for name in subtests:
        subtest_dict[f"{name.replace('_', ' ').title()} Score"] = working_df[f"BVMT_{name}"].values
    
    desc = get_descriptives(subtest_dict)
    
    pretty_print("BVMT", desc)


def RPM():
    """Compare normative data for the RPM."""
    rpm_score = working_df["RPM_score"].values
    rpm_time = working_df["RPM_time"].values

    desc = get_descriptives({"Score": rpm_score, "Time": rpm_time})
    
    pretty_print("RPM", desc)


def Demographics():
    """Get demographic descriptive data"""
    subtests = ['age', 'sex', 'education', 'vr_experience', 'typical_sleep', 'depression', 'anxiety', 'stress']

    subtest_dict = {}
    for name in subtests:
        subtest_dict[f"{name.replace('_', ' ').title()}"] = working_df[f"{name}"].values
    
    desc = get_descriptives(subtest_dict)
    
    pretty_print("Demographics", desc)

# Run init on import
__init__()