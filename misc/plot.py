import pandas as pd
import numpy as np

df = pd.read_excel('output.xlsx')

tsvr1d = 0
tsvr2d = 0
tsvr3d = 0
tsvr4d = 0
tsvr1c = 0
tsvr2c = 0
tsvr3c = 0
tsvr4c = 0

# var = [
#     fr-strange-what-s
# fr-strange-where-s
# re-strange-what-s
# re-strange-where-s
# fr-port-what-s
# fr-port-where-s
# re-port-what-s
# re-port-where-s
# fr-land-what-s
# fr-land-where-s
# re-land-what-s
# re-land-where-s
# fr-item-what-s
# fr-item-when-s
# fr-item-up-s
# fr-item-down-s
# cr-item-what-s
# cr-item-when-s
# cr-item-up-s
# cr-item-down-s
# re-item-what-s
# re-item-when-s
# re-item-up-s
# re-item-down-s
# fr-item-n1-s
# fr-item-n2-s
# fr-item-n3-s
# fr-item-n4-s
# fr-item-n5-s
# fr-item-n6-s
# cr-item-n1-s
# cr-item-n2-s
# cr-item-n3-s
# cr-item-n4-s
# cr-item-n5-s
# cr-item-n6-s
# re-item-n1-s
# re-item-n2-s
# re-item-n3-s
# re-item-n4-s
# re-item-n5-s
# re-item-n6-s
# fr-item-common-s
# fr-item-congruent-s
# fr-item-incongruent-s
# cr-item-common-s
# cr-item-congruent-s
# cr-item-incongruent-s
# re-item-common-s
# re-item-congruent-s
# re-item-incongruent-s
# }

for i in range(len(df)):
    test = df.loc[i,'test']
    if ~np.isnan(df.loc[i,var]):
        if test == 1:
            tsvr1d += df.loc[i,var]
            tsvr1c += 1
        if test == 2:
            tsvr2d += df.loc[i,var]
            tsvr2c += 1
        if test == 3:
            tsvr3d += df.loc[i,var]
            tsvr3c += 1
        if test == 4:
            tsvr4d += df.loc[i,var]
            tsvr4c += 1

print(var)
print(f"1 = {round(tsvr1d/tsvr1c,2)}")
print(f"2 = {round(tsvr2d/tsvr2c,2)}")
print(f"3 = {round(tsvr3d/tsvr3c,2)}")
print(f"4 = {round(tsvr4d/tsvr4c,2)}")