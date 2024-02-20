import pandas as pd
from cogtests.helpers import get_descriptives, pretty_print, education_years, calc_group_t, str_to_range

# Normative data acquired from: https://www.tandfonline.com/doi/full/10.1080/13803395.2021.1917523
#                           and https://doi.org/10.1080/13803395.2011.559157


def BVMT(working_df: pd.DataFrame, print_out: list|str|None = None):
    """Compare BVMT scores from this sample against population scores."""
    print("\nCalculating BVMT T-Scores...")
    
    if print_out and ("descriptives" in print_out):
        BVMT_desc(working_df)

    score_vars = ["Learning Rate", "Scaled Total Recall", "Scaled Delayed Recall",
                  "Raw Total Recall", "Raw Delayed Recall"]
    
    LR_scores = pd.DataFrame(columns=score_vars)
    for PID, participant in working_df.iterrows():
        # Read table for scaling total and delayed recall scores
        score_table = pd.read_excel("./cogtests/data/BVMT scaled scores.xlsx", index_col="SS").fillna('').map(str_to_range)

        total_recall, delayed_recall, total_recall_raw, delayed_recall_raw = BVMT_calc_recall_scores(participant, score_table)

        learning = BVMT_calc_LR_T_score(participant)

        participant_scores = {"Learning Rate": learning,
                              "Scaled Total Recall": total_recall,
                              "Scaled Delayed Recall": delayed_recall,
                              "Raw Total Recall": total_recall_raw,
                              "Raw Delayed Recall": delayed_recall_raw}
        
        LR_scores.loc[PID] = participant_scores

    BVMT_group_Ts = pd.DataFrame(index = score_vars, columns = ["t-value", "DF", "p-value"])

    pop_data = lambda m,sd,n: {"pop_mean":m, "pop_sd":sd, "pop_n":n}

    pop_comparisons = {"Learning Rate":         pop_data(50,10,200),
                       "Scaled Total Recall":   pop_data(50,10,143),
                       "Scaled Delayed Recall": pop_data(50,10,143),
                       "Raw Total Recall":      pop_data(26.5,5.9,143),
                       "Raw Delayed Recall":    pop_data(10.2,1.7,143)}
    
    for name, pop in pop_comparisons.items():
        BVMT_group_Ts.loc[name] = calc_group_t(LR_scores[name],
                                    num_comparisons = len(score_vars),
                                    **pop)
    
    if print_out and ("descriptives" in print_out):
        pretty_print("BVMT", get_descriptives(dict(LR_scores)))

    if print_out and ("output" in print_out):
        print(BVMT_group_Ts)

    if print_out and ("unlikely" in print_out):
        unlikely_vals = BVMT_group_Ts[BVMT_group_Ts["p-value"]<0.05]
        if not unlikely_vals.empty:
            print("Variables with p-values less than 0.05:")
            print(unlikely_vals)
        else:
            print("No p-values less than 0.05 found for the BVMT tests.")


def BVMT_calc_LR_T_score(participant):
    age = participant["age"]
    education = education_years(participant["education"])
    # No correlation was found for sex, so it was not included in the regression
    sex = participant["sex"]

    t1_score = participant["BVMT_trial_1_score"]
    t3_score = participant["BVMT_trial_3_score"]

    # Calculate learning rate score from trial scores
    LR_score = (t3_score-t1_score)/(12-t1_score) if t1_score != 12 else 0

    # Correct learning rate score based on demographic variables
    LR_T_score = ((LR_score - (0.97 - (0.009*age) + (0.016*education)))/0.22) * 10 + 50

    # print(t1_score, t3_score, LR_score, LR_corrected_score)

    return LR_T_score


def BVMT_calc_recall_scores(participant, score_table: pd.DataFrame):
    age = participant["age"]
    education = education_years(participant["education"])
    # No correlation was found for sex, so it was not included in the regression
    sex = participant["sex"]

    # Get unscaled raw scores
    total_recall_unscaled = round(sum([participant[f"BVMT_trial_{n}_score"] for n in range(1,4)]))
    delayed_recall_unscaled = round(participant["BVMT_delayed_recall"])

    scores = {"Total Recall": total_recall_unscaled,
              "Delayed Recall": delayed_recall_unscaled}

    # Create lambda function to find the SS for a given value in a given column
    get_SS = lambda name: int(score_table.loc[[scores[name] in row for row in score_table[name]]].index[0])

    # Get scaled scores for each variable
    total_recall_scaled = get_SS("Total Recall")
    delayed_recall_scaled = get_SS("Delayed Recall")

    # Apply demographic corrections to scaled scores
    total_recall = ((total_recall_scaled-(0.2589*(education-14.11)+(-0.0515)*(age-37.62)+0.9276*sex+10.0712))/2.8912)*10+50
    delayed_recall = ((delayed_recall_scaled-(0.2084*(education-14.11)+(-0.0286)*(age-37.62)+10.3007))/2.6989)*10+50

    return (total_recall, delayed_recall, total_recall_unscaled, delayed_recall_unscaled)


def BVMT_desc(working_df: pd.DataFrame):
    """Compare normative data for the BVMT."""
    subtests = ['total_recall', 'learning', 'percent_recalled', 'recognition_hits',
    'recognition_false_alarms', 'recognition_discrimination_index', 'recognition_response_bias',
    'trial_1_score', 'trial_2_score', 'trial_3_score']
    subtest_dict = {}
    for name in subtests:
        subtest_dict[f"{name.replace('_', ' ').title()} Score"] = working_df[f"BVMT_{name}"].values
    
    desc = get_descriptives(subtest_dict)
    
    pretty_print("BVMT", desc)