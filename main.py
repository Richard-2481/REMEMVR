import data_loader
import pandas as pd

def get_data(query, data_df):
    
    variables_list = query.split(" ")[1:]
    search_results_df = data_loader.load(variables_list)

    if query.startswith("/load"):
        return search_results_df        
        
    if query.startswith("/add"):
        merged_df = pd.merge(data_df, search_results_df, on='UID', how='left')
        return merged_df

def main():
    
    data_df = pd.DataFrame()
    
    query = "/load age sex education RAVLT_sum_of_trials"
    data_df = get_data(query, data_df)
    data_df.to_csv("data.csv", index=False)  # Save dataframe to CSV
    from stats import descriptive_analysis, plot_histogram, plot_scatterplot
    descriptive_analysis(data_df)
    plot_histogram(data_df)
    plot_scatterplot(data_df, 'age', 'REMEMVR')

    # query = "/add education vr_experience"
    # data_df = get_data(query, data_df)
    # data_df.to_csv("data.csv", index=False)  # Save dataframe to CSV
    # print(data_df)

    # query = "/add RAVLT_sum_of_trials"
    # data_df = get_data(query, data_df)
    # data_df.to_csv("data.csv", index=False)  # Save dataframe to CSV
    # print(data_df)

if __name__ == "__main__":
    main()
