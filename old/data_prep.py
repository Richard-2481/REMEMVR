import pandas as pd
import numpy as np
import json

def main():

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

    for variable_name, variable_details in variables_json.items():
        summary_df[variable_name] = np.nan
        regex_list = variable_details['regex']
        func = variable_details['func']
        
        for uid in summary_df['UID']:
            regex_tags = master_df[uid].astype(str)

            data = []

            for regex in regex_list:
            
                matching_rows = regex_tags.str.contains(regex, na=False)
                data = master_df.loc[matching_rows, uid + '-D'].tolist()

                if len(data) == 0:
                    continue
                
                for value in data:
                    if value == "x":
                        value = np.nan
                    data.append(float(value))
            
            counter = 0
            result = 0

            if func == 'val':
                result = data[0]
            else:
                for i in len(range(data)):
                    result += data[i]
                    counter += 1

            if func == 'sum':
                result = result

            if func == 'mean':
                result = result / counter

            summary_df.loc[summary_df['UID'] == uid, variable_name] = result

    print(summary_df)

if __name__ == "__main__":
    main()