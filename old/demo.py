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

def demo_scores(regex_pattern='',df=master, uid_list=uid_all):
    
    data = {}

    age_results = lookup(f"Age", master, uid_all)
        
    for entry in age_results:
        uid = entry[0]
        age = float(entry[1])
        
        if uid not in data:
            data[uid] = {'age': [None] * 4, 'score': [None] * 4}
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

def create_color_gradient(steps):
    # Create a list of hues ranging from 0 to 360 degrees
    hues = np.linspace(0, 360, steps, endpoint=False)

    # Convert HSV colors (with full saturation and value) to RGB and then to hex
    gradient = [mcolors.to_hex(mcolors.hsv_to_rgb([hue/360, 1, 1])) for hue in hues]

    return gradient

def plot(data, title):
    # Define the colors and line styles
    colors = ['#ff0000', '#ff0000', '#ff7f00', '#ff7f00', 
              '#00ff00', '#00ff00', '#00ffff', '#00ffff', 
              '#0000ff', '#0000ff']
    line_styles = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--']

    plt.figure(figsize=(10, 6))

    for index, (uid, info) in enumerate(data.items()):
        tsvr = info['tsvr']
        score = info['score']
        # Cycle through colors and line styles
        color = colors[index % len(colors)]
        line_style = line_styles[index % len(line_styles)]
        plt.plot(tsvr, score, label=uid, color=color, linestyle=line_style)

    plt.xlabel("Hours Since VR")
    plt.ylabel("Mean Score")
    plt.title(title)
    plt.ylim(0, 1)

    plt.legend()
    plt.show()

def main():
    regex_pattern = '-ANS'
    search_results = rememvr_scores(regex_pattern)
    new_data = group(search_results, age_groups)
    plot(new_data, regex_pattern)
    print(new_data)

if __name__ == "__main__":
    main()
