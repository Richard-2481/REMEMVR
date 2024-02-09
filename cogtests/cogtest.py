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

def pretty_print(name: str, scores: dict):
    """Taking a score name and a dictionary of subscores in the form `{subscore1: [list,of,values], subscore2: [list,of,values],...}`
    prints a nice string displaying some descriptive stats for each score."""
    print_str = ""

    print_str += f"{name}:\n"

    for (scorename, score) in scores.items():
        score_desc = st.describe(score, nan_policy="omit")
        
        score_min = round(st.tmin(score, nan_policy="omit"))
        score_max = round(st.tmax(score, nan_policy="omit"))

        score_mean = round(score_desc.mean, 2)
        score_sd = round(score_desc.variance**0.5, 2)

        print_str += f"  {scorename}:\n    Min: {score_min}, Max: {score_max}\n    Mean: {score_mean}, SD: {score_sd}\n"
    
    print(print_str)

   
def NART():
    """Compare normative data for the NART."""
    nart_score = working_df["NART_score"].values
    nart_time = working_df["NART_time"].values
    
    pretty_print("Nart", {"Score": nart_score, "Time": nart_time})


def RAVLT():
    """Compare normative data for the RAVLT."""
    subtests = ['trial_1','trial_2','trial_3','trial_4','trial_5','trial_distraction','trial_free_recall','trial_delayed_recall']
    subtest_dict = {}
    for name in subtests:
        subtest_dict[f"{name.replace('_', ' ')} Score"] = working_df[f"RAVLT_{name}_score"].values
        subtest_dict[f"{name.replace('_', ' ')} Time"] = working_df[f"RAVLT_{name}_time"].values

    subtest_dict["Delay Duration"] = working_df[f"RAVLT_delayed_recall_duration"].values
    
    pretty_print("RAVLT", subtest_dict)


def BVMT():
    """Compare normative data for the BVMT."""
    pass


def RPM():
    """Compare normative data for the RPM."""
    rpm_score = working_df["RPM_score"].values
    rpm_time = working_df["RPM_time"].values
    
    pretty_print("RPM", {"Score": rpm_score, "Time": rpm_time})

# Run init on import
__init__()