import pandas as pd
from scipy import stats as st
import os, sys, inspect
import json


def __init__():
    """Load working dataframe or create it if it doesn't exist"""
    global working_df

    if os.path.exists("./cogtests/working_df.db"):
        print("Loaded working DataFrame!")
        working_df = pd.read_pickle("./cogtests/working_df.db")
    else:
        # Modify current path so that modules from parent folder can be imported
        current_dir = inspect.getfile(inspect.currentframe())
        lib_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
        sys.path.append(lib_path)

        from main import startup
        import data

        # Get data
        master_df, working_df, _ = startup()

        with open('data/variable_list.json', 'r') as file:
            variables_json = json.load(file)

        # Add cognitive testing variables to the working dataframe
        for variable in variables_json:
            if variable["group"] == "cognitive":
                working_df = data.add(variable, working_df, master_df)
        
        # Save the working DataFrame to a file for quicker loading next time
        pd.to_pickle(working_df, "./cogtests/working_df.db")

        print("Saved working DataFrame!")


## Cognitive tests
   
def NART():
    """Compare normative data for the NART."""
    nart_score = working_df["NART_score"].values
    nart_time = working_df["NART_time"].values

    desc = get_descriptives({"Score": nart_score, "Time": nart_time})
    
    pretty_print("NART", desc)


def RAVLT():
    """Compare normative data for the RAVLT."""
    subtests = ['trial_1','trial_5','trial_distraction','trial_free_recall','trial_delayed_recall']
    # Not included: 'trial_2','trial_3','trial_4'
    subtest_dict = {}
    for name in subtests:
        subtest_dict[f"{name.replace('_', ' ').title()} Score"] = working_df[f"RAVLT_{name}_score"].values
        # Not included: subtest_dict[f"{name.replace('_', ' ').title()} Time"] = working_df[f"RAVLT_{name}_time"].values

    subtest_dict["Delay Duration Time"] = working_df[f"RAVLT_delayed_recall_duration"].values

    desc = get_descriptives(subtest_dict)
    
    pretty_print("RAVLT", desc)

    for index, participant in working_df.iterrows():
        participant["education"] = RAVLT_education(participant["education"])
        scaled_scores = RAVLT_scaled_scores(participant)




def BVMT():
    """Compare normative data for the BVMT."""
    subtests = ['total_recall', 'learning', 'percent_recalled', 'recognition_hits',
    'recognition_false_alarms', 'recognition_discrimination_index', 'recognition_response_bias']
    subtest_dict = {}
    for name in subtests:
        subtest_dict[f"{name.replace('_', ' ').title()} Score"] = working_df[f"BVMT_{name}"].values
    
    desc = get_descriptives(subtest_dict)
    
    pretty_print("BVMT", desc)


def RPM():
    """Compare normative data for the RPM."""
    rpm_score = working_df["RPM_score"].values
    rpm_time = working_df["RPM_time"].values

    desc = get_descriptives({"Score": rpm_score, "Time": rpm_time})
    
    pretty_print("RPM", desc)


# Helper functions

def Demographics():
    """Get demographic descriptive data"""
    subtests = ['age', 'sex', 'education', 'vr_experience', 'typical_sleep', 'depression', 'anxiety', 'stress']

    subtest_dict = {}
    for name in subtests:
        subtest_dict[f"{name.replace('_', ' ').title()}"] = working_df[f"{name}"].values
    
    desc = get_descriptives(subtest_dict)
    
    pretty_print("Demographics", desc)


def pretty_print(name: str, descriptives: dict[dict]):
    """Taking an overall score name and a dictionary containing descriptives for each subscore
    in the form `{subscore1name: {descriptives}, subscore2name: {descriptives},...}`.\n
    Prints a nice string displaying the descriptive stats for each subscore."""
    print_str = f"{name}:\n"

    for (scorename, score_desc) in descriptives.items():
        min, max, mean, stdev = score_desc.values()
        print_str += f"   {scorename}:\n      Min: {min}, Max: {max}\n\
      Mean: {mean}, SD: {stdev}\n"
    
    print(print_str)


def get_descriptives(scores: dict):
    """Taking a dictionary of subscores in the form `{subscore1: [list,of,values], subscore2: [list,of,values],...}`
    returns a dictionary with the same keys but each value is a nested dict of relevant descriptive variables."""
    descriptives = {}
    for (scorename, score) in scores.items():
        descriptives[scorename] = {}
        score_desc = st.describe(score, nan_policy="omit")
        
        descriptives[scorename]["min"] = round(score_desc.minmax[0].__float__(),2)
        descriptives[scorename]["max"] = round(score_desc.minmax[1].__float__(),2)

        descriptives[scorename]["mean"] = round(score_desc.mean.__float__(), 2)
        descriptives[scorename]["stdev"] = round(score_desc.variance.__float__()**0.5, 2)

    return descriptives


def str_to_range(string: str) -> list[int]:
    """Convert a string (e.g. `'1-5'`) or int into a list of integers (e.g. `[1,2,3,4,5]`)."""
    string = str(string)
    if not string:
        # String is empty
        return []
    if '-' in string:
        # String contains range
        a, b = string.split('-')
        a, b = int(a), int(b)
        return list(range(a, b + 1))
    else:
        # String is a single integer
        return [int(string),]
    

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
    

def RAVLT_scaled_scores(participant) -> int:
    """Given a participant, converts the individual trial scores into several scaled scores, as specified by the RAVLT formula.\n
    Uses the table saved in `./normdata/RAVLT scaled scores.xlsx`."""
    # Get relevant scores
    trial_12345_total = participant["RAVLT_trial_12345_total"]
    sum_of_trials = participant["RAVLT_sum_of_trials"]
    delayed_recall = participant["RAVLT_trial_delayed_recall_score"]

    # Calculate recognition percentage correct
    recognition_hits = participant["RAVLT_recognition_hits"]
    recognition_misses = participant["RAVLT_recognition_misses"]
    recognition_score_pc = round(((recognition_hits + (15 - recognition_misses)) / 30) * 100)

    # Get relevant totals for conversion to scaled scores
    # Also converts everything into lists of integers for use with get_SS below, including number ranges (such that '1-5' becomes '[1,2,3,4,5]')
    score_table = pd.read_excel("./cogtests/normdata/RAVLT scaled scores.xlsx", index_col="SS").fillna('').map(str_to_range)

    # Crete lambda function to find the SS for a given value in a given column
    get_SS = lambda val,col: score_table.loc[[val in row for row in score_table[col]]].index[0]

    # Using get_SS, find the scaled scores for this participant in each of the 4 categories
    scaled_scores = {"Trials 1-5 Total": get_SS(trial_12345_total, "Trials 1-5 Total"),
                     "Delayed Recall":   get_SS(delayed_recall, "30-Min Recall"),
                     "Sum of Trials":    get_SS(sum_of_trials, "Sum of Trials"),
                     "Recognition PC":   get_SS(recognition_score_pc, "Recognition PC")}
    
    return scaled_scores


def RAVLT_t_scores(scaled_scores, age, sex, education, include_secondary = False):
    """The raw calculations to find the demographic-adjusted T-Scores for this participant.\n
    If `include_secondary` is True, then will also calculate T-Scores for secondary variables"""
    # Dict of magic numbers required for calculating each T-Score, as per RAVLT formula
    mnums = {"Trials 1-5 Total":        [10.2048820335,  0.0696731708, -2.0691847063,  0.2076286782, -0.0014410120,  0.0000000637336, 0.23569807],
             "Delayed Recall":          [12.4118437425, -0.0016432817, -1.8612455591,  0.1380628944, -0.0007027918,  0.0000001024411, 0.25299505],
             "Sum of Trials":           [10.8349191766,  0.0514562686, -2.0670904968,  0.1915793153, -0.0012694294, -0.0000001038205, 0.23673872],
             "Recognition PC":          [10.7915054797,  0.0163995950, -1.8832719513,  0.1180746912, -0.0005488200,  0.0000001925238, 0.29155771],
             "Trial 1":                 [10.5554207904,  0.0361599800, -0.0009181852, -1.2432854518,  0.1518778446, -0.0000001305326, 0.26867547],
             "Trial 2":                 [10.2054872384,  0.0513655747, -0.0011932848, -1.7651639080,  0.1919280336, -0.0000000448036, 0.25613384],
             "Trial 3":                 [10.6083066798,  0.0436932539, -0.0011424483, -1.7605746863,  0.1870822095, -0.0000001720072, 0.24470110],
             "Trial 4":                 [10.3271703981,  0.0637213583, -0.0013266346, -1.9598250972,  0.1859304791, -0.0000002083300, 0.24650622],
             "Trial 5":                 [ 9.9952872306,  0.0622550674, -0.0012837374, -1.9754640301,  0.1870905944, -0.0000000482041, 0.24516225],
             "Trials 1-3 Total":        [10.4083623066,  0.0537576200, -0.0012667176, -1.8439550539,  0.1987705165, -0.0000000560410, 0.23872421],
             "Distractor":              [ 8.9167820377,  0.0780069203, -0.0013677187, -1.1375184278,  0.1914262059,  2.3996448266, -0.0000533322, 0.00000967380900, 0.12287159], #???
             "Free Recall":             [],
             "Short-Term Retention PC": [],
             "Long-Term Retention PC":  [],
             "Memory Efficiency":       []}

    # The actual calculation. Don't even worry about it.
    t_score = lambda name: round(50+((((scaled_scores[name]-(mnums[name][0]+(age*mnums[name][1])+(sex*mnums[name][2])+(education*mnums[name][3])+(age**2*mnums[name][4])))/1)+mnums[name][5])/mnums[name][6]))

    # Make a dict of the primary T-Scores for this participant
    t_scores = {
        "trials_12345_total": t_score("Trials 1-5 Total"),
        "delayed_recall":     t_score("Delayed Recall"),
        "sum_of_trials":      t_score("Sum of Trials"),
        "recognition_PC":     t_score("Recognition PC")
        }
    
    return t_scores



# Run init on import
__init__()