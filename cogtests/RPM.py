import pandas as pd
from cogtests.helpers import get_descriptives, pretty_print, str_to_range, calc_group_t

## Normative data acquired from: https://doi.org/10.1111/jnp.12308

def RPM(working_df: pd.DataFrame, print_out: list|str|None = None):
    """Compare RPM scores from this sample against population scores."""
    print("\nCalculating RPM T-Scores...")
    
    if print_out and ("descriptives" in print_out):
        RPM_desc(working_df)

    # Get population norm data for each age range
    age_norms = pd.read_excel("./cogtests/data/RPM age norms.xlsx").fillna('')
    age_norms["age"] = age_norms["age"].map(str_to_range)
    bin_nums = age_norms.index

    # Create a bin for each age range
    age_bins = {num:[] for num in bin_nums}

    # This function classifies a given age into its relevant bin
    get_bin = lambda age: int(age_norms.loc[[age in row for row in age_norms["age"]]].index[0])

    # Iterate through each participant, adding their ID to the relevant bin
    for PID, participant in working_df.iterrows():
        age_bin = get_bin(participant["age"])
        age_bins[age_bin].append(PID)
        
    # Create list of title strings for each age range
    group_indices = [f"Score (ages {age_range[0]}-{age_range[-1]})" for age_range in age_norms["age"]]

    # Split the participants into seperate binned dataframes based on their age range
    participant_bins = {group_indices[num]:working_df.loc[age_bins[num]]["RPM_score"] for num in bin_nums}

    RPM_group_Ts = pd.DataFrame(index = group_indices, columns = ["t-value", "DF", "p-value"])

    for bin_num in bin_nums:
        age_range = age_norms.loc[bin_num]["age"]

        index = f"Score (ages {age_range[0]}-{age_range[-1]})"

        sample_RPM_scores = participant_bins[index]

        pop_norms = age_norms.loc[bin_num]

        RPM_group_Ts.loc[index] = calc_group_t(sample_RPM_scores,
                                                num_comparisons=len(group_indices), 
                                                pop_mean=pop_norms["mean"],
                                                pop_sd=pop_norms["sd"],
                                                pop_n=pop_norms["n"])
        

    if print_out and ("descriptives" in print_out):
        participant__binned_descriptives = get_descriptives({k:list(v) for k,v in participant_bins.items()})
        pretty_print("RPM", participant__binned_descriptives)

    if print_out and ("output" in print_out):
        print(RPM_group_Ts)

    if print_out and ("unlikely" in print_out):
        unlikely_vals = RPM_group_Ts[RPM_group_Ts["p-value"]<0.05]
        if not unlikely_vals.empty:
            print("Variables with p-values less than 0.05:")
            print(unlikely_vals)
        else:
            print("No p-values less than 0.05 found for the NART tests.")

    return RPM_group_Ts
        




def RPM_desc(working_df: pd.DataFrame):
    """Compare normative data for the RPM."""
    rpm_score = working_df["RPM_score"].values
    rpm_time = working_df["RPM_time"].values

    desc = get_descriptives({"Score": rpm_score, "Time": rpm_time})
    
    pretty_print("RPM", desc)