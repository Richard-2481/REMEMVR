import pandas as pd
from cogtests.helpers import get_descriptives, pretty_print, education_years, calc_group_t

## Normative data acquired from: https://doi.org/10.1111/ajpy.12188 and https://doi.org/10.1080/09602011.2016.1231121


def NART(working_df: pd.DataFrame, print_out: list|str|None = None):
    """Compare  NART scores from this sample against population scores."""
    print("\nCalculating NART T-Scores...")
    
    if print_out and ("descriptives" in print_out):
        NART_desc(working_df)

    IQ_scores = NART_IQ_Scores(working_df)

    NART_group_Ts = pd.DataFrame(index = ["Error Score", "Estimated IQ"], columns = ["t-value", "DF", "p-value"])

    NART_group_Ts.loc["Error Score"] = calc_group_t(IQ_scores["Errors"],
                                                        num_comparisons = 2, 
                                                        pop_mean=18.30,
                                                        pop_sd=8.98,
                                                        pop_n=92)

    NART_group_Ts.loc["Estimated IQ"] = calc_group_t(IQ_scores["Estimated IQ"],
                                                        num_comparisons = 2, 
                                                        pop_mean=110.78,
                                                        pop_sd=4.57,
                                                        pop_n=111)

    if print_out and ("descriptives" in print_out):
        pretty_print("NART", get_descriptives(dict(IQ_scores)))

    if print_out and ("output" in print_out):
        print(NART_group_Ts)

    if print_out and ("unlikely" in print_out):
        unlikely_vals = NART_group_Ts[NART_group_Ts["p-value"]<0.05]
        if not unlikely_vals.empty:
            print("Variables with p-values less than 0.05:")
            print(unlikely_vals)
        else:
            print("No p-values less than 0.05 found for the NART tests.")

    return NART_group_Ts


def NART_IQ_Scores(working_df: pd.DataFrame):
    """Calculate WAIS-IV Full-Scale demographic-adjusted IQ scores for each participant
    then compare the disribution of IQ scores against the population."""
    IQ_Scores = pd.DataFrame(columns = ["Errors", "Estimated IQ"])

    for _, participant in working_df.iterrows():
        education = participant["education"] = education_years(participant["education"])

        age = participant["age"]
        sex = 1-participant["sex"]

        # Get NART error score
        NART_correct = participant["NART_score"]
        NART_error = 50-participant["NART_score"]

        # Calculate participant estimated IQ score
        IQ = 123.20-1.172*(NART_error)-0.118*age-1.524*sex+0.944*education

        IQ_Scores.loc[participant["UID"]] = [NART_error, IQ]

    return IQ_Scores


def NART_desc(working_df: pd.DataFrame):
    """Print descriptive information for the NART."""
    nart_score = working_df["NART_score"].values
    nart_time = working_df["NART_time"].values

    desc = get_descriptives({"Score": nart_score, "Time": nart_time})
    
    pretty_print("NART", desc)