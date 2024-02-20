import pandas as pd
from scipy import stats as st
from cogtests.helpers import get_descriptives, pretty_print, education_years, calc_group_t, str_to_range

## Normative data acquired from: https://doi.org/10.1017%2FS1355617720000752


def RAVLT(working_df: pd.DataFrame, print_out: list|str|None = None, include_secondary = True):
    """Convert the sample of RAVLT scores into demographic-adjusted T-Scores then compare the
    distribution of T-Scores against the population distribution."""
    print("\nCalculating RAVLT T-Scores...")

    if print_out and ("descriptives" in print_out):
        RAVLT_desc(working_df)

    RAVLT_Participant_Ts = RAVLT_Participant_T_Scores(working_df, include_secondary)

    RAVLT_group_Ts = pd.DataFrame(index = RAVLT_Participant_Ts.columns, columns = ["t-value", "DF", "p-value"])

    for name, t_scores in RAVLT_Participant_Ts.items():
        # Calculate group t-value for this scoring variable
        welch_t, welch_df, adjusted_p_value = calc_group_t(t_scores,
                                                           num_comparisons=len(RAVLT_group_Ts),
                                                           pop_mean=50,
                                                           pop_sd=10,
                                                           pop_n=4428)

        # Add to group-level dataframe
        RAVLT_group_Ts.loc[name] = [welch_t, welch_df, adjusted_p_value]

    if print_out and ("descriptives" in print_out):
        pretty_print("RAVLT T-Scores", get_descriptives(dict(RAVLT_Participant_Ts)))

    if print_out and ("output" in print_out):
        print(RAVLT_group_Ts)

    if print_out and ("unlikely" in print_out):
        unlikely_vals = RAVLT_group_Ts[RAVLT_group_Ts["p-value"]<0.05]
        if not unlikely_vals.empty:
            print("Variables with p-values less than 0.05:")
            print(unlikely_vals)
        else:
            print("No p-values less than 0.05 found for the RAVLT tests.")
    
    return RAVLT_group_Ts


def RAVLT_Participant_T_Scores(working_df: pd.DataFrame, include_secondary = False):
    """Get the demographic-adjusted T-Scores for each participant in each scoring variable."""
    # Select which scores to include (primary only by default)
    if include_secondary:
        score_names = ["Trials 1-5 Total", "Trials 1-3 Total", "Sum of Trials",
                       "Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5",
                       "Distractor", "Delayed Recall", "Free Recall",
                       "Short-Term Retention PC", "Long-Term Retention PC",
                       "Memory Efficiency", "Recognition PC"]
    else:
        score_names = ["Trials 1-5 Total", "Delayed Recall", "Sum of Trials", "Recognition PC"]

    t_scores = pd.DataFrame(columns=score_names)

    for _, participant in working_df.iterrows():
        participant["education"] = education_years(participant["education"])

        scaled_scores = RAVLT_scaled_scores(participant, score_names, include_secondary)

        t_scores.loc[participant["UID"]] = RAVLT_calc_t_scores(scaled_scores, participant, score_names)
    
    return t_scores


def RAVLT_desc(working_df: pd.DataFrame):
    """Print descriptive information for each subscore of the RAVLT."""
    subtests = ['trial_1','trial_2','trial_3','trial_4','trial_5','trial_distraction','trial_free_recall','trial_delayed_recall']

    subtest_dict = {}
    for name in subtests:
        title_name = name.replace('_', ' ').title()
        subtest_dict[f"{title_name} Score"] = working_df[f"RAVLT_{name}_score"].values
        subtest_dict[f"{title_name} Time"]  = working_df[f"RAVLT_{name}_time"].values

    subtest_dict["Delay Duration Time"] = working_df[f"RAVLT_delayed_recall_duration"].values

    desc = get_descriptives(subtest_dict)
    
    pretty_print("RAVLT", desc)
    

def RAVLT_scaled_scores(participant, score_names, include_secondary = False):
    """Given a participant, converts the individual trial scores into several scaled scores, as specified by the RAVLT formula.\n
    Uses the table saved in `./normdata/RAVLT scaled scores.xlsx`."""
    # Get relevant scores
    scores = {}
    scores["Trials 1-5 Total"] = participant["RAVLT_trial_12345_total"]
    scores["Sum of Trials"] = participant["RAVLT_sum_of_trials"]
    scores["Delayed Recall"] = participant["RAVLT_trial_delayed_recall_score"]

    # Calculate recognition percentage correct
    recognition_hits = participant["RAVLT_recognition_hits"]
    recognition_misses = participant["RAVLT_recognition_misses"]
    scores["Recognition PC"] = round(((recognition_hits + (15 - recognition_misses)) / 30) * 100)

    # Calculate secondary scores if selected
    if include_secondary:
        for tnum in range(1,6):
            scores[f"Trial {tnum}"] = participant[f"RAVLT_trial_{tnum}_score"]
        
        scores["Trials 1-3 Total"] = sum([participant[f"RAVLT_trial_{num}_score"] for num in range(1,4)])
        scores["Distractor"] = participant["RAVLT_trial_distraction_score"]
        scores["Free Recall"] = participant["RAVLT_trial_free_recall_score"]
        scores["Short-Term Retention PC"] = min(round(100*(scores["Free Recall"]   /participant["RAVLT_trial_5_score"])), 100)\
                                            if participant["RAVLT_trial_5_score"]>0 else 0
        scores["Long-Term Retention PC"]  = min(round(100*(scores["Delayed Recall"]/participant["RAVLT_trial_5_score"])), 100)\
                                            if participant["RAVLT_trial_5_score"]>0 else 0
        scores["Memory Efficiency"] = round((((scores["Delayed Recall"]/15)/(scores["Trials 1-5 Total"]/ 75))+\
                                      ((recognition_hits/15)-(recognition_misses/15)))*100)

    # Get relevant table for conversion to scaled scores
    # Also converts everything into an integer range for use with get_SS below
    score_table = pd.read_excel("./cogtests/data/RAVLT scaled scores.xlsx", index_col="SS").fillna('').map(str_to_range)

    # Create lambda function to find the SS for a given value in a given column
    get_SS = lambda name: int(score_table.loc[[scores[name] in row for row in score_table[name]]].index[0])

    # Using get_SS, find the scaled scores for this participant in each of the 4 categories
    scaled_scores = {name: get_SS(name) for name in score_names}
    
    return scaled_scores


def RAVLT_calc_t_scores(scaled_scores, participant, score_names):
    """The raw calculations to find the demographic-adjusted T-Scores for a given participant.\n
    If `include_secondary` is True, then will also calculate T-Scores for secondary variables"""
    # Get a DataFrame of magic numbers required for calculating each T-Score, as per RAVLT formula
    mnums = pd.read_pickle("./cogtests/data/RAVLT magic numbers.db")

    age = participant["age"]
    sex = participant["sex"]
    education = participant["education"]

    # The actual calculation. Don't even worry about it.
    t_score = lambda name: round(50+((((scaled_scores[name]-(mnums[name][0]\
                            +(age*mnums[name][1])+(sex*mnums[name][2])\
                            +(education*mnums[name][3])\
                            +(age**2*mnums[name][4])))/(mnums[name][5]\
                            +(age**2*mnums[name][6])))+mnums[name][7])/mnums[name][8]))
    
    # Make a dict of the selected T-Scores for this participant
    t_scores = pd.Series({name: t_score(name) for name in score_names}, dtype=int)

    return t_scores