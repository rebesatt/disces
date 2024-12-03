import os
import sys
import pandas as pd
import glob
import numpy as np
import logging

#Logger Configuration:
LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

CURRENT_WD = os.getcwd()

EXPERIMENT_DIR = os.path.abspath(__file__).replace("scalability.py", "")
os.chdir(EXPERIMENT_DIR)
os.chdir("../src")
SRC_DIR = os.getcwd()

os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)

# Include files used for discovery here
from testbench_helper_functions import generate_plots, match_algos, _read_sample



def min_trace_length(iterations:int, repetitions:int, overwrite:bool = False):
    file_name = f'trace_length'
    file_path = f'experiment_results/{file_name}.csv'
    results = []
    sample_path = "../datasets/synt/trace_length"
    if not os.path.isdir(sample_path):
        os.makedirs(sample_path)
    all_files = os.listdir(sample_path)
    if not all_files:
        os.system("python ../datasets/synt/synt_sample_generator.py")
        all_files = os.listdir(sample_path)
    file_list = [file.split('.')[0] for file in all_files]
    for file in file_list:
        j = int(file.split('_')[-1])
        new_sample_list = _read_sample(file, sample_path)
        for _ in range(repetitions):
            results, columns = match_algos(sample_list= new_sample_list, results=results, iterations=iterations, j=j+1, mod='trace length', file_path=file_path)
        # columns = ['sample size', 'trace length','iteration', 'algorithm', 'time', 'iterations', 'queryset size', 'queryset', 'mode', 'searchspace', 'max type occourence', 'trace max type occourence', 'min pattern length', 'sample set', 'max query length']
        dataframe = pd.DataFrame(results, columns=columns)
        dataframe.to_csv(file_path)
    
    return dataframe

def streams(iterations:int, repetitions:int, overwrite:bool = False):
    file_name = f'streams'
    file_path = f'experiment_results/{file_name}.csv'
    results = []
    sample_path = f'../datasets/synt/{file_name}'
    if not os.path.isdir(sample_path):
        os.makedirs(sample_path)
    all_files = os.listdir(sample_path)
    if not all_files:
        os.system("python ../datasets/synt/synt_sample_generator.py")
        all_files = os.listdir(sample_path)
    file_list = [file.split('.')[0] for file in all_files]
    for file in file_list:
        j = int(file.split('_')[-1])
        print(j)
        if os.path.isfile(file_path):
            dataframe = pd.read_csv(file_path, header=0, index_col=0)
            columns = dataframe.columns
            if dataframe.loc[dataframe['iterations']==j].shape[0] == 0:
                new_sample_list = _read_sample(file, sample_path)
                for i in range(repetitions):
                    print(i)
                    results, columns = match_algos(sample_list= new_sample_list, results=results, iterations=iterations, j=j, mod='trace length', file_path=file_path)
            else:
                results = dataframe.values.tolist()
        else:
            new_sample_list = _read_sample(file, sample_path)

            for i in range(repetitions):
                    print(i)
                    results, columns = match_algos(sample_list= new_sample_list, results=results, iterations=iterations, j=j, mod='trace length', file_path=file_path)
        # columns = ['sample size', 'trace length','iteration', 'algorithm', 'time', 'iterations', 'queryset size', 'queryset', 'mode', 'searchspace', 'max type occourence', 'trace max type occourence', 'min pattern length', 'sample set', 'max query length']
        dataframe = pd.DataFrame(results, columns=columns)
        dataframe.to_csv(file_path)


    return dataframe



if __name__ == "__main__":
    LOGGER.info("Starting experiments for scalability analysis")
    
    iterations = 10000
    file_name = f'trace_length'
    file_path = f'experiment_results/{file_name}.csv'
    if os.path.isfile(file_path):
        df1 = pd.read_csv(file_path)
    else:
        df1 = min_trace_length(iterations=iterations, repetitions= 5, overwrite=False)
    df1['mode'] = 'stream length'

    LOGGER.info("Finished 2/6")
    
    
    iterations = 100000
    file_name = f'streams'
    file_path = f'experiment_results/{file_name}.csv'
    # if os.path.isfile(file_path):
    #     df4 = pd.read_csv(file_path)
    # else:
    df4 = streams(iterations=iterations, repetitions= 5, overwrite=False)
    df4['mode'] = 'streams'

    frames = [df4, df1]
    result = pd.concat(frames)
    result.to_csv('experiment_results/scalability_plots.csv')
    file_name = 'scalability_plots'
    generate_plots(dataframe=result, file_name=file_name, x='iterations', y='time[s]',
                    hue='algorithm', col = 'mode', col_wrap=2, kind='scale',
                    facet_kws={'sharex': False, 'sharey': True})
    LOGGER.info("Finished generating plot")
    # save plot of {experiment_name} to 'experiment_results/{experiment_name}.pdf'
    # Example: plot of 'sota_4_broken' to 'experiment_results/synt_plots.pdf'

    # Return to the origin
    os.chdir(CURRENT_WD)
