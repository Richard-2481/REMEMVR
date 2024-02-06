import numpy as np

# This function adds columns to the working df
# It iterates through all the UID's processing each regex
# A function is then applied to the participant's results
# Then move to the next UID


def add(variable, working_df, master_df):

    regex_list = variable["regex"]

    for uid in working_df['UID']:
    
        results_list = []
        
        for regex in regex_list:

            regex_tags = master_df[uid].astype(str)
            matching_rows = regex_tags.str.contains(regex, na=False)
            results_temp = master_df.loc[matching_rows, uid + '-D'].tolist()

            for item in results_temp:
                if item == "x":
                    item = np.nan
                    # print(uid, variable["name"], regex, item)
                results_list.append(float(item))
        
        if variable["func"] == "sum":
            result = np.sum(results_list)
        if variable["func"] == "mean":
            result = np.mean(results_list)
        
        working_df.loc[working_df['UID'] == uid, variable["name"]] = result
    
    return working_df


