import pandas as pd
from helpers import get_descriptives, pretty_print

def RAVLT(working_df: pd.DataFrame, include_secondary = False):
    """Compare normative data for the RAVLT."""
    t_scores = {}
    for _, participant in working_df.iterrows():
        participant["education"] = RAVLT_education(participant["education"])

        scaled_scores = RAVLT_scaled_scores(participant, include_secondary)

        t_scores[participant["UID"]] = RAVLT_t_scores(scaled_scores, participant, include_secondary)
    
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


def str_to_range(string: str):
    """Convert a string (e.g. `'1-5'`) or int into a range (e.g. `range(1,6)`)."""
    string = str(string)
    if not string:
        # String is empty
        return range(0,0)
    if '-' in string:
        # String contains range
        a, b = string.split('-')
        # Swap m character with minus sign
        a = a.replace("m","-")
        b = b.replace("m","-")
        a, b = int(float(a)), int(float(b))
        return range(a, b + 1)
    else:
        # String is a single integer
        val = int(float(string))
        return range(val, val + 1)
    

def RAVLT_education(education: int) -> int:
    """Convert education score (1-9) into the 'number of years' format specified by the RAVLT formula (9-20)"""
    education_convert = {
        # High school (Year 9 or lower):
        0: 9,
        # High school (Year 10):
        1: 10,
        # High school (Year 12):
        2: 12,
        # Certificate 1 & 2:
        3: 12,
        # Certificate 3 & 4:
        4: 12,
        # Diploma or Advanced Diploma:
        5: 14,
        # Bachelor's Degree:
        6: 16,
        # Graduate Certificate or Graduate Diploma:
        7: 17,
        # Master's Degree:
        8: 18,
        # Doctoral Degree:
        9: 20}
    
    return education_convert[education]
    

def RAVLT_scaled_scores(participant, include_secondary = False):
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
        scores["Short-Term Retention PC"] = 100*(scores["Free Recall"]   /participant["RAVLT_trial_5_score"])
        scores["Long-Term Retention PC"]  = 100*(scores["Delayed Recall"]/participant["RAVLT_trial_5_score"])
        scores["Memory Efficiency"] = ((scores["Delayed Recall"]/15)/(scores["Trials 1-5 Total"]/ 75))+\
                                      ((recognition_hits/15)-(recognition_misses/15))

    # Get relevant table for conversion to scaled scores
    # Also converts everything into an integer range for use with get_SS below
    score_table = pd.read_excel("./cogtests/data/RAVLT scaled scores.xlsx", index_col="SS").fillna('').map(str_to_range)

    # Crete lambda function to find the SS for a given value in a given column
    get_SS = lambda name: int(score_table.loc[[scores[name] in row for row in score_table[name]]].index[0])

    score_names = ["Trials 1-5 Total", "Sum of Trials", "Delayed Recall", "Recognition PC"]
    if include_secondary:
        # Use all scores
        score_names = list(score_table.columns)

    # Using get_SS, find the scaled scores for this participant in each of the 4 categories
    scaled_scores = {name: get_SS(name) for name in score_names}
    
    return scaled_scores


def RAVLT_t_scores(scaled_scores, participant, include_secondary = False):
    """The raw calculations to find the demographic-adjusted T-Scores for a given participant.\n
    If `include_secondary` is True, then will also calculate T-Scores for secondary variables"""
    # Get a DataFrame of magic numbers required for calculating each T-Score, as per RAVLT formula
    mnums = pd.read_pickle("./data/RAVLT magic numbers.db")

    age = participant["age"]
    sex = participant["sex"]
    education = participant["education"]

    # The actual calculation. Don't even worry about it.
    t_score = lambda name: round(50+((((scaled_scores[name]-(mnums[name][0]\
                            +(age*mnums[name][1])+(sex*mnums[name][2])\
                            +(education*mnums[name][3])\
                            +(age**2*mnums[name][4])))/(mnums[name][5]\
                            +(age**2*mnums[name][6])))+mnums[name][7])/mnums[name][8]))

    # Make a list of primary scores
    score_names = ["Trials 1-5 Total", "Delayed Recall", "Sum of Trials", "Recognition PC"]
    if include_secondary:
        # Use all scores
        score_names = list(mnums.columns)
    
    # Make a dict of the selected T-Scores for this participant
    t_scores = {name: t_score(name) for name in score_names}

    return t_scores