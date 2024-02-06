import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

master_df = pd.read_excel('data/master_data_copy.xlsx')

with open('data/variables.json', 'r') as file:
    variables_json = json.load(file)

uid_list = [
    'A010', 'A011', 'A012', 'A013', 'A014', 'A015', 'A016', 'A017', 'A018', 'A019',
    'A020', 'A021', 'A022', 'A023', 'A024', 'A025', 'A026', 'A027', 'A028', 'A029',
    'A030', 'A031', 'A032', 'A033', 'A034', 'A035', 'A036', 'A037', 'A038', 'A039',
    'A040', 'A041', 'A042', 'A043', 'A044', 'A045', 'A046', 'A047', 'A048', 'A049',
            'A051', 'A052', 'A053', 'A054', 'A055', 'A056', 'A057', 'A058', 'A059',
    'A060', 'A061', 'A062', 'A063', 'A064', 'A065', 'A066', 'A067', 'A068', 'A069',
    'A070', 'A071', 'A072', 'A073', 'A074', 'A075', 'A076', 'A077', 'A078', 'A079',
    'A080', 'A081', 'A082', 'A083', 'A084', 'A085', 'A086', 'A087', 'A088', 'A089',
            'A091', 'A092', 'A093', 'A094', 'A095', 'A096', 'A097', 'A098', 'A099',
    'A100', 'A101', 'A102', 'A103', 'A104', 'A105', 'A106', 'A107', 'A108', 'A109'
]

summary_df = pd.DataFrame(uid_list, columns=['UID'])

def lookup(regex_list=[], df=master_df, uid_list=uid_list):

    results_list = []
    
    for i in range(len(uid_list)):
    
        uid = uid_list[i]
        uid_results = [uid]     
        regex_tags = df[uid].astype(str)
    
        for regex in regex_list:
            matching_rows = regex_tags.str.contains(regex, na=False)
            data = df.loc[matching_rows, uid + '-D']       
        
            if len(data) == 0:
                continue
        
            else:
                for value in data:
                    print(uid)
                    print(regex)
                    print(value)
                    if value == "x":
                        value = np.nan
                    uid_results.append(float(value))
        
                results_list.append(uid_results)
    
    results = np.array(results_list)
    
    # Creates list as follows 
    # [UID 1, Result 1, Result 2...]
    # [UID 2, Result 1, Result 2...]

    return results

def main():
    for variable_name, variable_details in variables_json.items():
        regex = variable_details['regex']
        func = variable_details['func']
        results_df = lookup(regex)
        summary_df = pd.DataFrame(uid_list, columns=['UID'])

        for row in results_df:
            uid = row[0]
            row = row[1:]
            if func == 'sum':
                for item in row:
                    print(f"item = {item} type = {type(item)}")
                    
                    value += float(item)
            elif func == 'mean':
                value = np.mean(row)
            else:
                value = row[0]  # Assuming the default case is to take the first value

            # Update the summary DataFrame
            summary_df.loc[summary_df['UID'] == uid, variable_name] = value

    print(summary_df)
        
if __name__ == "__main__":
    main()