import pandas as pd
import os, sys, inspect
import json

from RAVLT import RAVLT, RAVLT_desc
from NART  import NART,  NART_desc
from BVMT  import BVMT,  BVMT_desc
from RPM   import RPM,   RPM_desc

from helpers import Demographics


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


# Run init on import
__init__()


# Demographics(working_df)
        
# RAVLT_desc(working_df)
print(RAVLT(working_df))

# NART_desc(working_df)
print(NART(working_df))

# BVMT_desc(working_df)
print(BVMT(working_df))

# RPM_desc(working_df)
print(RPM(working_df))