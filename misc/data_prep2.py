import pandas as pd
import numpy as np

master = pd.read_excel('Master_Data_Raw.xlsx')
summary = pd.read_excel('Summary.xlsx')

def txt(df, regex_pattern):
    matching_values = []
    for identifier_col in df.columns[::2]:
        data_col = identifier_col + '-D'
        identifier_series = df[identifier_col].astype(str)
        matches = identifier_series.str.contains(regex_pattern, na=False)
        matching_data = df.loc[matches, data_col]
        matching_values.extend(matching_data.tolist())
    array = np.array(matching_values)
    return array

def num(df, regex_pattern, uid):
    data_col = uid + '-D'
    if data_col in df.columns:
        identifier_series = df[uid].astype(str)
        matches = identifier_series.str.contains(regex_pattern, na=False)
        matching_data = df.loc[matches, data_col].replace('', np.nan).dropna().astype(float)
        array = np.array(matching_data)
        return array
    else:
        # Return an empty array if the column doesn't exist
        return np.array([])

def mean(df, regex, uid, sf = 2): 
    val = num(df, regex, uid)
    try:
        av = round(sum(val)/len(val),sf)
    except:
        av = np.nan
        # print("!", end = "", flush = True)
    return av

regex = [

    # DEMOGRAPHICS    
    'Age', 'age',
    'Sex', 'sex',
    'Education', 'education',
    'VR_Exp', 'vr-experience',
    'DASS_Dep', 'depression',
    'DASS_Anx', 'anxiety',
    'DASS_Str', 'stress',

    # COGNITIVE TESTS
    'RPM-Scor', 'rpm-s',
    'NAR-Scor', 'nart-s',
    'BVM-T1Sc', 'bvmt-t1-s',
    'BVM-T2Sc', 'bvmt-t2-s',
    'BVM-T3Sc', 'bvmt-t3-s',
    'BVM-TDSc', 'bvmt-dr-s',
    'RAV-T1Sc', 'ravlt-t1-s',
    'RAV-T2Sc', 'ravlt-t2-s',
    'RAV-T3Sc', 'ravlt-t3-s',
    'RAV-T4Sc', 'ravlt-t4-s',
    'RAV-T5Sc', 'ravlt-t5-s',
    'RAV-TDSc', 'ravlt-dist-s',
    'RAV-FRSc', 'ravlt-fr-s',
    'RAV-DRSc', 'ravlt-dr-s',
    
    # REMEMVR
    'TSVR', 'tsvr',

    # REMEMVR Scores
    'RFR-N-STRA-ANS', 'fr-strange-what-s',
    'RFR-L-STRA-ANS', 'fr-strange-where-s',
    'RRE-N-STRA-ANS', 're-strange-what-s',
    'RRE-L-STRA-ANS', 're-strange-where-s',

    'RFR-N-PORT-ANS', 'fr-port-what-s',
    'RFR-L-PORT-ANS', 'fr-port-where-s',
    'RRE-N-PORT-ANS', 're-port-what-s',
    'RRE-L-PORT-ANS', 're-port-where-s',

    'RFR-N-LAND-ANS', 'fr-land-what-s',
    'RFR-L-LAND-ANS', 'fr-land-where-s',
    'RRE-N-LAND-ANS', 're-land-what-s',
    'RRE-L-LAND-ANS', 're-land-where-s',

    'IFR-N.*ANS', 'fr-item-what-s',
    'IFR-O.*ANS', 'fr-item-when-s',
    'IFR-U.*ANS', 'fr-item-up-s',
    'IFR-D.*ANS', 'fr-item-down-s',

    'ICR-N.*ANS', 'cr-item-what-s',
    'ICR-O.*ANS', 'cr-item-when-s',
    'ICR-U.*ANS', 'cr-item-up-s',
    'ICR-D.*ANS', 'cr-item-down-s',

    'IRE-N.*ANS', 're-item-what-s',
    'IRE-O.*ANS', 're-item-when-s',
    'IRE-U.*ANS', 're-item-up-s',
    'IRE-D.*ANS', 're-item-down-s',

    'IFR.*n1.*ANS', 'fr-item-n1-s',
    'IFR.*n2.*ANS', 'fr-item-n2-s',
    'IFR.*n3.*ANS', 'fr-item-n3-s',
    'IFR.*n4.*ANS', 'fr-item-n4-s',
    'IFR.*n5.*ANS', 'fr-item-n5-s',
    'IFR.*n6.*ANS', 'fr-item-n6-s',

    'ICR.*n1.*ANS', 'cr-item-n1-s',
    'ICR.*n2.*ANS', 'cr-item-n2-s',
    'ICR.*n3.*ANS', 'cr-item-n3-s',
    'ICR.*n4.*ANS', 'cr-item-n4-s',
    'ICR.*n5.*ANS', 'cr-item-n5-s',
    'ICR.*n6.*ANS', 'cr-item-n6-s',

    'IRE.*n1.*ANS', 're-item-n1-s',
    'IRE.*n2.*ANS', 're-item-n2-s',
    'IRE.*n3.*ANS', 're-item-n3-s',
    'IRE.*n4.*ANS', 're-item-n4-s',
    'IRE.*n5.*ANS', 're-item-n5-s',
    'IRE.*n6.*ANS', 're-item-n6-s',

    'IFR.*i.CM.*ANS', 'fr-item-common-s',
    'IFR.*i.CG.*ANS', 'fr-item-congruent-s',
    'IFR.*i.IN.*ANS', 'fr-item-incongruent-s',

    'ICR.*i.CM.*ANS', 'cr-item-common-s',
    'ICR.*i.CG.*ANS', 'cr-item-congruent-s',
    'ICR.*i.IN.*ANS', 'cr-item-incongruent-s',

    'IRE.*i.CM.*ANS', 're-item-common-s',
    'IRE.*i.CG.*ANS', 're-item-congruent-s',
    'IRE.*i.IN.*ANS', 're-item-incongruent-s',

    # REMEMVR Confidence
    'RFR-N-STRA-CON', 'fr-strange-what-c',
    'RFR-L-STRA-CON', 'fr-strange-where-c',
    'RRE-N-STRA-CON', 're-strange-what-c',
    'RRE-L-STRA-CON', 're-strange-where-c',

    'RFR-N-PORT-CON', 'fr-port-what-c',
    'RFR-L-PORT-CON', 'fr-port-where-c',
    'RRE-N-PORT-CON', 're-port-what-c',
    'RRE-L-PORT-CON', 're-port-where-c',

    'RFR-N-LAND-CON', 'fr-land-what-c',
    'RFR-L-LAND-CON', 'fr-land-where-c',
    'RRE-N-LAND-CON', 're-land-what-c',
    'RRE-L-LAND-CON', 're-land-where-c',

    'IFR-N.*CON', 'fr-item-what-c',
    'IFR-O.*CON', 'fr-item-when-c',
    'IFR-U.*CON', 'fr-item-up-c',
    'IFR-D.*CON', 'fr-item-down-c',

    'ICR-N.*CON', 'cr-item-what-c',
    'ICR-O.*CON', 'cr-item-when-c',
    'ICR-U.*CON', 'cr-item-up-c',
    'ICR-D.*CON', 'cr-item-down-c',

    'IRE-N.*CON', 're-item-what-c',
    'IRE-O.*CON', 're-item-when-c',
    'IRE-U.*CON', 're-item-up-c',
    'IRE-D.*CON', 're-item-down-c',

    'IFR.*n1.*CON', 'fr-item-n1-c',
    'IFR.*n2.*CON', 'fr-item-n2-c',
    'IFR.*n3.*CON', 'fr-item-n3-c',
    'IFR.*n4.*CON', 'fr-item-n4-c',
    'IFR.*n5.*CON', 'fr-item-n5-c',
    'IFR.*n6.*CON', 'fr-item-n6-c',

    'ICR.*n1.*CON', 'cr-item-n1-c',
    'ICR.*n2.*CON', 'cr-item-n2-c',
    'ICR.*n3.*CON', 'cr-item-n3-c',
    'ICR.*n4.*CON', 'cr-item-n4-c',
    'ICR.*n5.*CON', 'cr-item-n5-c',
    'ICR.*n6.*CON', 'cr-item-n6-c',

    'IRE.*n1.*CON', 're-item-n1-c',
    'IRE.*n2.*CON', 're-item-n2-c',
    'IRE.*n3.*CON', 're-item-n3-c',
    'IRE.*n4.*CON', 're-item-n4-c',
    'IRE.*n5.*CON', 're-item-n5-c',
    'IRE.*n6.*CON', 're-item-n6-c',

    'IFR.*i.CM.*CON', 'fr-item-common-c',
    'IFR.*i.CG.*CON', 'fr-item-congruent-c',
    'IFR.*i.IN.*CON', 'fr-item-incongruent-c',

    'ICR.*i.CM.*CON', 'cr-item-common-c',
    'ICR.*i.CG.*CON', 'cr-item-congruent-c',
    'ICR.*i.IN.*CON', 'cr-item-incongruent-c',

    'IRE.*i.CM.*CON', 're-item-common-c',
    'IRE.*i.CG.*CON', 're-item-congruent-c',
    'IRE.*i.IN.*CON', 're-item-incongruent-c',

    ]

for i in range(len(summary)):
# for i in range(8):

    uid = summary.loc[i,'user-id']
    test = summary.loc[i,'test']
    print(".",end = "", flush = True)
    # print(f"{uid}", flush=True)
    # print(f"T{test}", flush=True)
    for j in range(0, len(regex), 2):
        val = mean(master, f"{uid}.*T{test}.*{regex[j]}", uid, 2)
        if np.isnan(val):
            try:
                val = mean(master, f"{uid}.*X.*{regex[j]}", uid, 2)
            except:
                val = np.nan
        # print(f"{regex[j]} = {val}", flush=True)
        summary.loc[i,regex[j + 1]] = val
summary = summary.dropna(subset=['tsvr'])
print("\n")
print(summary)

summary.to_excel('output.xlsx')
