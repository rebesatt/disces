import os
import sys
import pandas as pd
import glob
import numpy as np
import logging

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

CURRENT_WD = os.getcwd()

EXPERIMENT_DIR = os.path.abspath(__file__).replace("synt_plot.py", "")
os.chdir(EXPERIMENT_DIR)
os.chdir("../src")
SRC_DIR = os.getcwd()

os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)

# Include files used for discovery here
from testbench_helper_functions import generate_plots, match_algos, _read_sample

def max_pattern_ratio(iterations:int, repetitions:int, overwrite: bool = False):
    file_name = 'max_pattern'
    file_path = f'experiment_results/{file_name}.csv'
    # dir_path = f'../datasets/{file_name}.'
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
        new_sample_list = _read_sample(file, sample_path)
        for _ in range(repetitions):
            results, columns = match_algos(sample_list= new_sample_list, results=results, iterations=iterations, j=j+1, mod='pattern sum', file_path=file_path)
    # columns = ['sample size', 'trace length','iteration', 'algorithm', 'time', 'iterations', 'queryset size', 'queryset', 'mode', 'searchspace', 'max type occourence', 'trace max type occourence', 'min pattern length', 'sample set', 'max query length']
    dataframe = pd.DataFrame(results, columns=columns)
    dataframe.to_csv(file_path)
    return dataframe

def type_ratio(iterations:int, repetitions:int, overwrite:bool = False):
    file_name = f'types'
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
        new_sample_list = _read_sample(file,sample_path)
        for _ in range(repetitions):
            results, columns = match_algos(sample_list= new_sample_list, 
                                           results=results, iterations=iterations,
                                            j=j, mod='types', file_path=file_path)
        dataframe = pd.DataFrame(results, columns=columns)
        dataframe.to_csv(file_path)

    return dataframe

def max_type(iterations:int, repetitions:int, overwrite:bool = False):
    file_name = f'max_type'
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
        new_sample_list = _read_sample(file,sample_path)
        for _ in range(repetitions):
            results, columns = match_algos(sample_list= new_sample_list, results=results, iterations=iterations,
                                            j=j, mod='max type', file_path=file_path)
        dataframe = pd.DataFrame(results, columns=columns)
        dataframe.to_csv(file_path)
    

    return dataframe

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
        new_sample_list = _read_sample(file, sample_path)
        for _ in range(repetitions):
                results, columns = match_algos(sample_list= new_sample_list, results=results, iterations=iterations, j=j, mod='trace length', file_path=file_path)
        # columns = ['sample size', 'trace length','iteration', 'algorithm', 'time', 'iterations', 'queryset size', 'queryset', 'mode', 'searchspace', 'max type occourence', 'trace max type occourence', 'min pattern length', 'sample set', 'max query length']
        dataframe = pd.DataFrame(results, columns=columns)
        dataframe.to_csv(file_path)


    return dataframe

def domain_size(iterations:int, repetitions:int, overwrite:bool = False):
    file_name = f'domain_size'
    file_path = f'experiment_results/{file_name}.csv'
    results = []
    sample_path = f'../datasets/synt/{file_name}'
    if not os.path.isdir(sample_path):
        os.makedirs(sample_path)
    all_files = os.listdir(sample_path)
    if not all_files:
        os.system("python ../datasets/synt/synt_sample_generator.py")
        all_files = os.listdir(sample_path)
    file_list = [file.split('.')[-2] for file in all_files]
    for file in file_list:
        j = int(file.split('_')[-1])
        new_sample_list = _read_sample(file, sample_path)
        for _ in range(repetitions):
            results, columns = match_algos(sample_list= new_sample_list, results=results,
                                            iterations=iterations, j=j+1, mod="domain size", file_path=file_path)
        dataframe = pd.DataFrame(results, columns=columns)
        dataframe.to_csv(file_path)

    return dataframe



if __name__ == "__main__":
    LOGGER.info("Starting experiments with synthetic data")
    iterations = 101
    file_name = f'types'
    file_path = f'experiment_results/{file_name}.csv'
    result_path = 'experiment_results'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    if os.path.isfile(file_path):
        df3 = pd.read_csv(file_path)
    else:
        df3 = type_ratio(iterations=iterations, repetitions= 5, overwrite=False)
        df3['mode'] = 'types'
    
    LOGGER.info("Finished 1/6")

    
    iterations = 100
    file_name = f'trace_length'
    file_path = f'experiment_results/{file_name}.csv'
    if os.path.isfile(file_path):
        df1 = pd.read_csv(file_path)
    else:
        df1 = min_trace_length(iterations=iterations, repetitions= 5, overwrite=False)
    df1['mode'] = 'stream length'

    LOGGER.info("Finished 2/6")
    
    iterations = 101
    file_name = f'domain_size'
    file_path = f'experiment_results/{file_name}.csv'
    if os.path.isfile(file_path):
        df6 = pd.read_csv(file_path)
    else:
        df6 = domain_size(iterations=iterations, repetitions= 5, overwrite=False)
        df6['mode'] = 'attributes'

    LOGGER.info("Finished 3/6")
    iterations = 1000
    file_name = f'streams'
    file_path = f'experiment_results/{file_name}.csv'
    if os.path.isfile(file_path):
        df4 = pd.read_csv(file_path)
    else:
        df4 = streams(iterations=iterations, repetitions= 5, overwrite=False)
    df4['mode'] = 'streams'

    LOGGER.info("Finished 4/6")
    iterations = 10
    file_name = f'max_pattern'
    file_path = f'experiment_results/{file_name}.csv'
    if os.path.isfile(file_path):
        df5 = pd.read_csv(file_path)
    else:
        df5 = max_pattern_ratio(iterations=iterations, repetitions= 5, overwrite=False)
        df5['mode'] = 'pattern sum'
    LOGGER.info("Finished 5/6")
    iterations = 100
    file_name = f'max_type'
    file_path = f'experiment_results/{file_name}.csv'
    if os.path.isfile(file_path):
        df2 = pd.read_csv(file_path)
    else:
        df2 = max_type(iterations=iterations, repetitions= 5, overwrite=False)
        df2['mode'] = 'type sum'
    LOGGER.info("Finished 6/6")

    frames = [df4, df6, df2, df1, df3, df5]
    result = pd.concat(frames)
    result.to_csv('experiment_results/synt_plots.csv')
    file_name = 'synt_plots'
    generate_plots(dataframe=result, file_name=file_name, x='iterations', y='time[s]',
                    hue='algorithm', col = 'mode', col_wrap=3, kind='synt',
                    facet_kws={'sharex': False, 'sharey': False})
    LOGGER.info("Finished generating plot")
    # save plot of {experiment_name} to 'experiment_results/{experiment_name}.pdf'
    # Example: plot of 'sota_4_broken' to 'experiment_results/synt_plots.pdf'

    # Return to the origin
    os.chdir(CURRENT_WD)
