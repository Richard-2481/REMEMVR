import data
import plot
import pandas as pd
import json

def startup():
    print("Loading Master Data...", end = "", flush=True)
    master_df = pd.read_excel('data/master_data_clean.xlsx')
    print(" Done")
    # master_df = []

    uid_list = [
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

    working_df = pd.DataFrame(uid_list, columns=['UID'])

    with open('data/variable_list.json', 'r') as file:
        variables_json = json.load(file)

    # Add the variables from plot list to the working dataframe
    print("Creating working dataframe", end = "", flush=True)

    with open('data/plots.json', 'r') as file:
        plots_json = json.load(file)

    variable_list = []
    for figure in plots_json:
        if figure["draw"] == 1:
            for variable in figure["xyz"]:
                variable_list.append(variable)

    # Remove duplicates from variable_list
    variable_list = list(set(variable_list))
    
    for variable in variables_json:
        if variable["name"] in variable_list:
            working_df = data.add(variable, working_df, master_df)
            print(".", end = "", flush=True)

    print(" Done")

    return master_df, working_df, plots_json

def main():

    # Start the program by loading the master dataset and preparing our working dataframe and loading variables in JSON format
    master_df, working_df, plots_json = startup()

    # Now we plot our variables
    
    for figure in plots_json:
        if figure["draw"] == 1:
            plot.plot(working_df, figure)
                                
if __name__ == "__main__":
    main()

    # Add demographic variables to the working dataframe
    # for variable in variables_json:
    #     if variable["group"] == "demographics":
    #         working_df = data.add(variable, working_df, master_df)
    #         # plot.box(working_df[[variable["name"]]])
    #         plot.histogram(working_df[[variable["name"]]])
    
    # Add cognitive testing variables to the working dataframe
    # for variable in variables_json:
    #     if variable["group"] == "cognitive":
    #         working_df = data.add(variable, working_df, master_df)
    #         plot.box(working_df[[variable["name"]]])


    # Example of calling a specific variable
    # variable_list = ["age", "sex"]
    # for variable in variables_json:
    #     if variable["name"] in variable_list:
    #         print(variable["regex"])