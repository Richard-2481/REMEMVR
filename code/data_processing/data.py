import pandas as pd
import os
import numpy as np

def addData(master_df, data_df, variable):

    print(".",end="",flush=True)
    
    variable_name = variable["Name"]
    func = variable["Function"]
    regex_list = variable["Regex"].split(", ")
    variable_type = variable["Type"]

    data_df = data_df.copy()
    # Add a new column to the working dataframe
    data_df[variable_name] = None
    data_df[variable_name] = data_df[variable_name].astype(object)

    # Now iterate through the working dataframe one participant at a time

    for i in range(len(data_df)):
        
        uid = data_df.loc[i, 'UID']
        test = data_df.loc[i, 'TEST']
        
        results_list = []
        skip = False

        for regex in regex_list:

            if variable_type == 'vr':

                regex = f"-T{test}-.*{regex}"

            if variable_type != 'vr' and test > 1:
                
                skip = True
                
                break

            regex_tags = master_df[uid].astype(str)
            matching_rows = regex_tags.str.contains(regex, na=False)
            results_temp = master_df.loc[matching_rows, uid + '-D'].tolist()

            
            for item in results_temp:

                if item == "x":

                    item = np.nan

                if func != "string":
                    try:
                        results_list.append(float(item))
                    except:
                        results_list.append(item)
                else:
                    results_list.append(item)
            
        if skip == True:

            result = data_df.loc[i-1, variable_name]

        else:

            if results_list == []:

                result = 0
                continue

            if func == "sum":
                
                result = round(np.sum(results_list),2)
                
            if func == "mean":
                
                result = round(np.mean(results_list),2)
                
            if func == "string":
                
                try:
                    result = ', '.join(results_list)
                except:
                    result = ""
                    for item in results_list:
                        result += str(item) + ", "

            if func == "multi":

                result = results_list[0] * results_list[1]

            if func == "list":

                result = results_list

        data_df.at[i, variable_name] = result

    return data_df

def createDataframe(dfMaster, dfVariables):

    dfData = pd.DataFrame(columns=['UID', 'TEST'])

    uidList = [
        'A010', 'A011', 'A012', 'A013', 'A014', 'A015', 'A016', 'A017', 'A018', 'A019',
        'A020', 'A021', 'A022', 'A023', 'A024', 'A025', 'A026', 'A027', 'A028', 'A029',
        'A030', 'A031', 'A032', 'A033', 'A034', 'A035', 'A036', 'A037', 'A038', 'A039',
        'A040', 'A041', 'A042', 'A043', 'A044', 'A045', 'A046', 'A047', 'A048', 'A049',
        'A050', 'A051', 'A052', 'A053', 'A054', 'A055', 'A056', 'A057', 'A058', 'A059',
        'A060', 'A061', 'A062', 'A063', 'A064', 'A065', 'A066', 'A067', 'A068', 'A069',
        'A070', 'A071', 'A072', 'A073', 'A074', 'A075', 'A076', 'A077', 'A078', 'A079',
        'A080', 'A081', 'A082', 'A083', 'A084', 'A085', 'A086', 'A087', 'A088', 'A089',
        'A090', 'A091', 'A092', 'A093', 'A094', 'A095', 'A096', 'A097', 'A098', 'A099',
        'A100', 'A101', 'A102', 'A103', 'A104', 'A105', 'A106', 'A107', 'A108', 'A109'
        ]

    # uidExclude = [
    #     'A026'
    # ]
    # for uid in uidExclude:
    #     uidList.remove(uid)

    test_list = [1,2,3,4]

    for px in uidList:

        for test in test_list:

            dfData.loc[len(dfData)] = [px, test]

    for index, variable in dfVariables.iterrows():

        dfData = addData(dfMaster, dfData, variable)

    return dfData

def read_csv_data(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file and returns it as a pandas DataFrame.

    Parameters:
    filepath (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(filepath)

def save_csv_data(filepath: str, dataframe: pd.DataFrame) -> None:
    """
    Saves a pandas DataFrame to a CSV file.

    Parameters:
    filepath (str): The path to save the CSV file.
    dataframe (pd.DataFrame): The DataFrame to be saved.
    """
    dataframe.to_csv(filepath, index=False)

def loadMaster():

    print("\n--- Master Data ---\n")

    # If master data is cached, load it. If not, load from Excel and create a new cache.

    if os.path.exists('cache/dfMaster.csv'):   

        print("Cached Version Found")

        # dfMaster = pd.read_pickle('cache/dfMaster.db')
        dfMaster = read_csv_data('cache/dfMaster.csv')

        print("Master Data Loaded")

    else:

        print("Cached Version Not Found\nLoading from Excel")

        dfMaster = pd.read_excel('master.xlsx')

        print("Creating New Cache")

        cache_files = os.listdir('cache')
        for file in cache_files:
            os.remove(f'cache/{file}')

        # pd.to_pickle(dfMaster, "cache/dfMaster.db")
        save_csv_data('cache/dfMaster.csv', dfMaster)

        print("Master Data Loaded")

    return dfMaster

def loadVariables():

    print("\n--- Variables ---\n")
    
    dfVariables = pd.read_csv('variables.csv')

    if os.path.exists('cache/dfVariables.csv'):   

        print("Cached Version Found")

        # dfVariablesCache = pd.read_pickle('cache/dfVariables.db')   
        dfVariablesCache = read_csv_data('cache/dfVariables.csv')

        if dfVariables.equals(dfVariablesCache):

            print("No Changes Detected")

        else:

            print("Changes Detected\nCreating New Cache")

            # pd.to_pickle(dfVariables, "cache/dfVariables.db")
            save_csv_data('cache/dfVariables.csv', dfVariables)

    else:

        print("Cache not found\nCreating New Cache")

        # pd.to_pickle(dfVariables, "cache/dfVariables.db")
        save_csv_data('cache/dfVariables.csv', dfVariables)

    print("Variables Loaded")

    return dfVariables

def loadData(dfMaster, dfVariables):

    print("\n--- Extracting Data ---\n")

    if os.path.exists('cache/dfData.csv'):

        print("Cached Version Found")

        # dfData = pd.read_pickle('cache/dfData.db')
        dfData = read_csv_data('cache/dfData.csv')

        print("\rData Loaded")

        return dfData
    
    else:

        print("Changes Detected\nRecomputing Data")

        dfData = createDataframe(dfMaster, dfVariables)

        # pd.to_pickle(dfData, "cache/dfData.db")
        save_csv_data('cache/dfData.csv', dfData)

        print("\rData Loaded")

        return dfData

def startup():

    """Loads dfMaster and creates dfWorking from variables.json"""

    print("""
██████  ███████ ███    ███ ███████ ███    ███ ██    ██ ██████  
██   ██ ██      ████  ████ ██      ████  ████ ██    ██ ██   ██ 
██████  █████   ██ ████ ██ █████   ██ ████ ██ ██    ██ ██████  
██   ██ ██      ██  ██  ██ ██      ██  ██  ██  ██  ██  ██   ██ 
██   ██ ███████ ██      ██ ███████ ██      ██   ████   ██   ██                                                              
""")

    dfMaster = loadMaster()                                # Load Master Data

    dfVariables = loadVariables()               # Load Variables

    dfData = loadData(dfMaster, dfVariables)  # Load Data
    
    return dfMaster, dfVariables, dfData

if __name__ == "__main__":

    startup()