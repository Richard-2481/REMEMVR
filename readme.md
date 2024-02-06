

What am I trying to do?
    Run all my data analyses in Python

Why?
    Because it's easier and faster

What format is my data?
    In my jankey REMEMVR pseudoSQL
    I have build a function to collect data together (data_loader.py)

        get_data(query, dataframe)
            query = "/load age sex education RAVLT_sum_of_trials"

    # Need to add all this a new file called data.py

    variables_json
    {
        "name": 
        "group":
        "regex":
        "func": 
        "text":
    },


main.py
    This is our main program
    Create the main dataframe with all the UID's in uid column

data.py
    This is where the data is imported from the master database and exported as a pandas dataframe

plot.py
    This does all the data visualisations

stats.py
    This does all the statistical analysis



What analyses do I need to run?
    Do the cog values match normative data?


Future ideas:
    A FastAPI webpage where I can select variables from a dropdown to create plots
        x axis = variable(s)
        x function = sum/mean/sd of the variables selected
        y axis = variable(s)
        y function = sum/mean/sd of the variables selected
        type = histogram/scatter etc

    A fast regression/correlation calculator
        Check assumptions
        input variables
        etc

    Use code interpreter to run analyses?
        Pros: Awesome
        Cons: Are the answers correct?
