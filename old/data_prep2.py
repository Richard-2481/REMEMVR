import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

master = pd.read_excel('data/master_data_copy.xlsx')

uid_all = [
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

age_groups = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10']
           
def lookup(regex_pattern='', df=master, uid_list=uid_all):

    results_list = []

    for i in range(len(uid_list)):

        uid = uid_list[i]
        uid_results = [uid]     
        regex_tags = df[uid].astype(str)
        matching_rows = regex_tags.str.contains(regex_pattern, na=False)
        data = df.loc[matching_rows, uid + '-D']
        
        if len(data) == 0:
            continue
        
        else:
            for value in data:
                uid_results.append(value)
            results_list.append(uid_results)
        
    results = np.array(results_list)
       
    return results

def rememvr_scores(regex_pattern='',df=master, uid_list=uid_all):
    
    data = {}

    for i in range(4):
        tsvr_results = lookup(f"T{i+1}.*TSVR", master, uid_all)
        for entry in tsvr_results:
            uid = entry[0]
            tsvr_val = float(entry[1])
            if uid not in data:
                data[uid] = {'tsvr': [None] * 4, 'score': [None] * 4}
            data[uid]['tsvr'][i] = tsvr_val
        score_results = lookup(f"T{i+1}.*{regex_pattern}", master, uid_all)
        for entry in score_results:
            uid = entry[0]
            score_list = entry[1:]
            numeric_values = [float(num) for num in score_list]
            mean_value = np.mean(numeric_values)
            mean_value_rounded = round(mean_value,2)
            data[uid]['score'][i] = mean_value_rounded

    return data

def group(data, age_groups):
    # Initialize a dictionary to hold positional data
    positional_data = {group: {'tsvr': [], 'score': []} for group in age_groups}

    # Iterate over the data and organize by age group and position
    for user_id, values in data.items():
        group_prefix = user_id[:3]  # Get the first three characters as the age group
        if group_prefix in positional_data:
            for i, (tsvr_value, score_value) in enumerate(zip(values['tsvr'], values['score'])):
                # Extend the list if it's not long enough
                while len(positional_data[group_prefix]['tsvr']) <= i:
                    positional_data[group_prefix]['tsvr'].append([])
                    positional_data[group_prefix]['score'].append([])

                positional_data[group_prefix]['tsvr'][i].append(tsvr_value)
                positional_data[group_prefix]['score'][i].append(score_value)

    # Calculate the averages for each position in each age group
    averaged_data = {}
    for group, values in positional_data.items():
        averaged_data[group] = {
            'tsvr': [np.mean(position) if position else None for position in values['tsvr']],
            'score': [np.mean(position) if position else None for position in values['score']]
        }

    return averaged_data

def create_color_gradient(start_color, end_color, steps):
    start = np.array(mcolors.to_rgb(start_color))
    end = np.array(mcolors.to_rgb(end_color))
    return [mcolors.to_hex((1 - ratio) * start + ratio * end) for ratio in np.linspace(0, 1, steps)]

def plot(data, title):

    custom_colors = create_color_gradient('#ff0000', '#0000ff', 10)

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)
    plt.rcParams['lines.linewidth'] = 2

    plt.figure(figsize=(10, 6))

    for uid, info in data.items():
        tsvr = info['tsvr']
        score = info['score']
        # plt.scatter(tsvr, score, label=None)
        plt.plot(tsvr, score, label=uid)
        
    plt.xlabel("Hours Since VR")
    plt.ylabel("Mean Score")
    plt.title(title)
    plt.ylim(0, 1)

    plt.legend()
    plt.show()

def main():
    search_results = rememvr_scores('-N-.*ANS')
    new_data = group(search_results, age_groups)
    plot(new_data,'by_group')
    print(new_data)

if __name__ == "__main__":
    main()

# def mean(df, regex, uid, sf = 2): 
#     val = regex_lookup(df, regex, uid, "num")
#     try:
#         av = round(sum(val)/len(val),sf)
#     except:
#         av = np.nan
#         # print("!", end = "", flush = True)
#     return av

# regex = [

#     # DEMOGRAPHICS    
#     'Age', 'age',
#     'Sex', 'sex',
#     'Education', 'education',
#     'VR_Exp', 'vr-experience',
#     'DASS_Dep', 'depression',
#     'DASS_Anx', 'anxiety',
#     'DASS_Str', 'stress',

#     # COGNITIVE TESTS
#     'RPM-Scor', 'rpm-s',
#     'NAR-Scor', 'nart-s',
#     'BVM-T1Sc', 'bvmt-t1-s',
#     'BVM-T2Sc', 'bvmt-t2-s',
#     'BVM-T3Sc', 'bvmt-t3-s',
#     'BVM-TDSc', 'bvmt-dr-s',
#     'RAV-T1Sc', 'ravlt-t1-s',
#     'RAV-T2Sc', 'ravlt-t2-s',
#     'RAV-T3Sc', 'ravlt-t3-s',
#     'RAV-T4Sc', 'ravlt-t4-s',
#     'RAV-T5Sc', 'ravlt-t5-s',
#     'RAV-TDSc', 'ravlt-dist-s',
#     'RAV-FRSc', 'ravlt-fr-s',
#     'RAV-DRSc', 'ravlt-dr-s',
    
#     # REMEMVR
#     'TSVR', 'tsvr',

#     # REMEMVR Scores
#     'RFR-N-STRA-ANS', 'fr-strange-what-s',
#     'RFR-L-STRA-ANS', 'fr-strange-where-s',
#     'RRE-N-STRA-ANS', 're-strange-what-s',
#     'RRE-L-STRA-ANS', 're-strange-where-s',

#     'RFR-N-PORT-ANS', 'fr-port-what-s',
#     'RFR-L-PORT-ANS', 'fr-port-where-s',
#     'RRE-N-PORT-ANS', 're-port-what-s',
#     'RRE-L-PORT-ANS', 're-port-where-s',

#     'RFR-N-LAND-ANS', 'fr-land-what-s',
#     'RFR-L-LAND-ANS', 'fr-land-where-s',
#     'RRE-N-LAND-ANS', 're-land-what-s',
#     'RRE-L-LAND-ANS', 're-land-where-s',

#     'IFR-N.*ANS', 'fr-item-what-s',
#     'IFR-O.*ANS', 'fr-item-when-s',
#     'IFR-U.*ANS', 'fr-item-up-s',
#     'IFR-D.*ANS', 'fr-item-down-s',

#     'ICR-N.*ANS', 'cr-item-what-s',
#     'ICR-O.*ANS', 'cr-item-when-s',
#     'ICR-U.*ANS', 'cr-item-up-s',
#     'ICR-D.*ANS', 'cr-item-down-s',

#     'IRE-N.*ANS', 're-item-what-s',
#     'IRE-O.*ANS', 're-item-when-s',
#     'IRE-U.*ANS', 're-item-up-s',
#     'IRE-D.*ANS', 're-item-down-s',

#     'IFR.*n1.*ANS', 'fr-item-n1-s',
#     'IFR.*n2.*ANS', 'fr-item-n2-s',
#     'IFR.*n3.*ANS', 'fr-item-n3-s',
#     'IFR.*n4.*ANS', 'fr-item-n4-s',
#     'IFR.*n5.*ANS', 'fr-item-n5-s',
#     'IFR.*n6.*ANS', 'fr-item-n6-s',

#     'ICR.*n1.*ANS', 'cr-item-n1-s',
#     'ICR.*n2.*ANS', 'cr-item-n2-s',
#     'ICR.*n3.*ANS', 'cr-item-n3-s',
#     'ICR.*n4.*ANS', 'cr-item-n4-s',
#     'ICR.*n5.*ANS', 'cr-item-n5-s',
#     'ICR.*n6.*ANS', 'cr-item-n6-s',

#     'IRE.*n1.*ANS', 're-item-n1-s',
#     'IRE.*n2.*ANS', 're-item-n2-s',
#     'IRE.*n3.*ANS', 're-item-n3-s',
#     'IRE.*n4.*ANS', 're-item-n4-s',
#     'IRE.*n5.*ANS', 're-item-n5-s',
#     'IRE.*n6.*ANS', 're-item-n6-s',

#     'IFR.*i.CM.*ANS', 'fr-item-common-s',
#     'IFR.*i.CG.*ANS', 'fr-item-congruent-s',
#     'IFR.*i.IN.*ANS', 'fr-item-incongruent-s',

#     'ICR.*i.CM.*ANS', 'cr-item-common-s',
#     'ICR.*i.CG.*ANS', 'cr-item-congruent-s',
#     'ICR.*i.IN.*ANS', 'cr-item-incongruent-s',

#     'IRE.*i.CM.*ANS', 're-item-common-s',
#     'IRE.*i.CG.*ANS', 're-item-congruent-s',
#     'IRE.*i.IN.*ANS', 're-item-incongruent-s',

#     # REMEMVR Confidence
#     'RFR-N-STRA-CON', 'fr-strange-what-c',
#     'RFR-L-STRA-CON', 'fr-strange-where-c',
#     'RRE-N-STRA-CON', 're-strange-what-c',
#     'RRE-L-STRA-CON', 're-strange-where-c',

#     'RFR-N-PORT-CON', 'fr-port-what-c',
#     'RFR-L-PORT-CON', 'fr-port-where-c',
#     'RRE-N-PORT-CON', 're-port-what-c',
#     'RRE-L-PORT-CON', 're-port-where-c',

#     'RFR-N-LAND-CON', 'fr-land-what-c',
#     'RFR-L-LAND-CON', 'fr-land-where-c',
#     'RRE-N-LAND-CON', 're-land-what-c',
#     'RRE-L-LAND-CON', 're-land-where-c',

#     'IFR-N.*CON', 'fr-item-what-c',
#     'IFR-O.*CON', 'fr-item-when-c',
#     'IFR-U.*CON', 'fr-item-up-c',
#     'IFR-D.*CON', 'fr-item-down-c',

#     'ICR-N.*CON', 'cr-item-what-c',
#     'ICR-O.*CON', 'cr-item-when-c',
#     'ICR-U.*CON', 'cr-item-up-c',
#     'ICR-D.*CON', 'cr-item-down-c',

#     'IRE-N.*CON', 're-item-what-c',
#     'IRE-O.*CON', 're-item-when-c',
#     'IRE-U.*CON', 're-item-up-c',
#     'IRE-D.*CON', 're-item-down-c',

#     'IFR.*n1.*CON', 'fr-item-n1-c',
#     'IFR.*n2.*CON', 'fr-item-n2-c',
#     'IFR.*n3.*CON', 'fr-item-n3-c',
#     'IFR.*n4.*CON', 'fr-item-n4-c',
#     'IFR.*n5.*CON', 'fr-item-n5-c',
#     'IFR.*n6.*CON', 'fr-item-n6-c',

#     'ICR.*n1.*CON', 'cr-item-n1-c',
#     'ICR.*n2.*CON', 'cr-item-n2-c',
#     'ICR.*n3.*CON', 'cr-item-n3-c',
#     'ICR.*n4.*CON', 'cr-item-n4-c',
#     'ICR.*n5.*CON', 'cr-item-n5-c',
#     'ICR.*n6.*CON', 'cr-item-n6-c',

#     'IRE.*n1.*CON', 're-item-n1-c',
#     'IRE.*n2.*CON', 're-item-n2-c',
#     'IRE.*n3.*CON', 're-item-n3-c',
#     'IRE.*n4.*CON', 're-item-n4-c',
#     'IRE.*n5.*CON', 're-item-n5-c',
#     'IRE.*n6.*CON', 're-item-n6-c',

#     'IFR.*i.CM.*CON', 'fr-item-common-c',
#     'IFR.*i.CG.*CON', 'fr-item-congruent-c',
#     'IFR.*i.IN.*CON', 'fr-item-incongruent-c',

#     'ICR.*i.CM.*CON', 'cr-item-common-c',
#     'ICR.*i.CG.*CON', 'cr-item-congruent-c',
#     'ICR.*i.IN.*CON', 'cr-item-incongruent-c',

#     'IRE.*i.CM.*CON', 're-item-common-c',
#     'IRE.*i.CG.*CON', 're-item-congruent-c',
#     'IRE.*i.IN.*CON', 're-item-incongruent-c',

#     ]

# # for i in range(len(summary)):
# # # for i in range(8):

# #     uid = summary.loc[i,'user-id']
# #     test = summary.loc[i,'test']
# #     print(".",end = "", flush = True)
# #     # print(f"{uid}", flush=True)
# #     # print(f"T{test}", flush=True)
# #     for j in range(0, len(regex), 2):
# #         val = mean(master, f"{uid}.*T{test}.*{regex[j]}", uid, 2)
# #         if np.isnan(val):
# #             try:
# #                 val = mean(master, f"{uid}.*X.*{regex[j]}", uid, 2)
# #             except:
# #                 val = np.nan
# #         # print(f"{regex[j]} = {val}", flush=True)
# #         summary.loc[i,regex[j + 1]] = val
# # summary = summary.dropna(subset=['tsvr'])
# # print("\n")
# # print(summary)

# # summary.to_excel('output.xlsx')
