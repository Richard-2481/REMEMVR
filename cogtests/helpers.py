import pandas as pd
from math import sqrt
from scipy import stats as st

def Demographics(working_df: pd.DataFrame):
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


def get_descriptives(scores: dict) -> dict[str,dict]:
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


def education_years(education: int) -> int:
    """Convert education score (1-9) into the 'number of years' format (9-20)."""
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


def calc_group_t(sample_scores, num_comparisons, pop_mean, pop_sd, pop_n) -> tuple[float, float, float]:
    desc = get_descriptives({"score": sample_scores})["score"]

    # Get values for Welch's T-Test:
    # Mean
    mean1 = desc["mean"]
    mean2 = pop_mean

    # Sample size
    n1 = len(sample_scores)
    n2 = pop_n

    # Variance
    var1 = desc["stdev"]**2
    var2 = pop_sd**2

    # Scaled variance
    p1 = (var1**2)/n1
    p2 = (var2**2)/n2
    
    # Perform Welch's T-Test comparing the demographic-adjusted
    # T-Score distributions found in this study against the expected
    # population demographic-adjusted T-Score distributions
    welch_t = abs((mean1-mean2)/sqrt((var1/n1)+(var2/n2)))

    # Calculate degrees of freedom
    welch_df = ((p1+p2)**2)/(((p1**2)/(n1-1))+((p2**2)/(n2-1)))

    # Calculate probability of this T-Score assuming null hypothesis
    p_value = st.t.sf(welch_t, welch_df)

    # Adjust p-value for multiple comparisons using simple Bonferroni correction
    adjusted_p_value = min(0.999999,p_value*num_comparisons)

    return welch_t, welch_df, adjusted_p_value


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