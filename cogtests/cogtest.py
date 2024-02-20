import pandas as pd
import os, sys, inspect
import json

from cogtests.RAVLT import RAVLT
from cogtests.NART  import NART
from cogtests.BVMT  import BVMT
from cogtests.RPM   import RPM

from cogtests.helpers import Demographics, get_descriptives, pretty_print


def __init__():
    """Load working dataframe or create it if it doesn't exist"""
    global working_df

    if os.path.exists("./cogtests/data/working_df.db"):
        print("Loaded working DataFrame!")
        working_df = pd.read_pickle("./cogtests/data/working_df.db")
    else:
        # Modify current path so that modules from parent folder can be imported
        current_dir = inspect.getfile(inspect.currentframe())
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
        pd.to_pickle(working_df, "./cogtests/data/working_df.db")

        print("Saved working DataFrame!")
    

def cognitive_test_comparisons(print_out: list|str|None = "all"):
    """Run the comparison tests for each cognitive test.\n
    `print_out` may be None, "all", or any selection of: `"descriptives", "output", "unlikely"`."""
    if print_out == "all":
        print_out = ["descriptives", "output"]
        
    RAVLT(working_df, print_out, include_secondary=True)
    BVMT( working_df, print_out)
    NART( working_df, print_out)
    RPM(  working_df, print_out)


# Run init on import
__init__()

cognitive_test_comparisons("output")