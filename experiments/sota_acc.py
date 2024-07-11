import os
import sys
import gzip
import pandas as pd
import logging
import argparse
from argparse import RawTextHelpFormatter
from statistics import mean

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

CURRENT_WD = os.getcwd()

EXPERIMENT_DIR = os.path.abspath(__file__).replace("sota_acc.py", "")
os.chdir(EXPERIMENT_DIR)
os.chdir("../src")
SRC_DIR = os.getcwd()

os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)

# Include files used for discovery here
from testbench_helper_functions import generate_plots, match_algos

def sota_acc(overwrite: bool = False, max_query_length = -1) -> pd.DataFrame:
    """Experiment for the State of the Art comparison
    Args:
        sample_size (int): number of traces in the sample
        trace_length (int): number of events per trace in the sample
        overwrite (bool, optional): Option to overwrite existing results and rerun experiment. Defaults to False.

    Returns:
        Dataframe: Collected data for each iteration
    """
    LOGGER.info(f'Running experiment for State of the Art comparison')
    file_path = f'../experiments/experiment_results/acc.csv'
    results = []
    if not os.path.isfile(file_path) or overwrite:
        repetition = 5
        run = 0
        
        discoveries = ['uni', 'sep', 'ups', 'sps', 'ilm_lossy']
        
        for abstraction in ['F1', 'F2', 'F3', 'G1', 'G2', 'G3']:
            if abstraction[0] == 'F':
                mod = 'finance'
            else:
                mod = 'google'
            
            current_path = f'../experiments/experiment_results/{abstraction}.csv'
            if not os.path.isfile(current_path):

                if abstraction == 'F1':
                    sample_path = '../datasets/finance/finance_query1.txt.gz'
                    trace_length = 51
                    sample_size = 79


                elif abstraction == 'F2':
                    trace_length = 41
                    sample_size = 1000
                    sample_path = '../datasets/finance/finance_query2.txt.gz'

                elif abstraction == 'F3':
                    trace_length = 37
                    sample_size = 250
                    max_query_length = 5
                    sample_path = '../datasets/finance/finance_query3.txt.gz'


                elif abstraction == 'G1':
                    trace_length = 8
                    sample_size = 1000
                    sample_path = '../datasets/google/google_query1.txt.gz'


                elif abstraction == 'G2':
                    trace_length = 8
                    sample_size = 1000
                    sample_path = '../datasets/google/google_query2.txt.gz'

                elif abstraction == 'G3':
                    trace_length = 5
                    sample_size = 10
                    max_query_length = -1
                    sample_path = '../datasets/google/google_query3.txt.gz'

                sample_list = []
                
                counter = 0
                file = gzip.open(sample_path, 'rb')
                for trace1 in file:
                    if counter == sample_size:
                        break

                    if trace_length == -1:
                        sample_list.append(' '.join(trace1.decode().split()))
                    else:
                        trace = ' '.join(trace1.decode().split()[-trace_length:])
                        sample_list.append(trace)
                        # length = len(trace.split())
                        # print(length)

                    counter += 1
                file.close()
                for rep in range(repetition):
                    results, columns = match_algos(sample_list=sample_list, results=results, iterations=abstraction, mod=mod,
                                        j=abstraction, discovery=discoveries, file_path=file_path, max_query_length=max_query_length)
                    run += 1
                    LOGGER.info(f'Finished {run} out of 30: {abstraction} for {mod}, Repetition {rep+1}/{repetition}')
                dataframe = pd.DataFrame(results, columns=columns)
                current_df = dataframe.loc[(dataframe['mode'] == mod) & (dataframe['iterations'] == abstraction)]
                current_df.to_csv(f'../experiments/experiment_results/{abstraction}.csv')
            else:
                current_df = pd.read_csv(current_path)
                current_list = [c_list[1:] for c_list in current_df.values.tolist()]
                results.extend(current_list)
                run += 5
                LOGGER.info(f'Finished {run} out of 30: {abstraction} for {mod}')
        if os.path.isfile(file_path) and not overwrite:
            dataframe = pd.read_csv(file_path)
        else:
            columns = current_df.columns.values[1:]
            dataframe = pd.DataFrame(results, columns=columns)
            dataframe.to_csv(file_path)
    else:
        dataframe = pd.read_csv(file_path)

    
    # df_tex.to_latex(tex_path, index=False)
    return dataframe

if __name__ == "__main__":

    file_path = '../experiments/experiment_results/acc.csv'
    file_name = 'sota_acc'
    acc_file_path = f'../experiments/experiment_results/{file_name}.csv'
    result_path = '../experiments/experiment_results'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    max_query_length = 4
    if not os.path.isfile(file_path):
        dataframe = sota_acc(overwrite=False, max_query_length=max_query_length)
    else:
        dataframe = pd.read_csv(file_path)
    
    results = []
    tex_result = []
    for abstraction in ['F1', 'F2', 'F3', 'G1', 'G2', 'G3']:
        if abstraction[0] == 'F':
            mod = 'finance'
        else:
            mod = 'google'

        descriptive_queryset1 = set(dataframe.loc[(dataframe['iterations'] == abstraction)& (dataframe['algorithm'] == 'uni')].queryset.values[0].replace("{", "").replace("}","").replace("'","").split(','))
        descriptive_queryset = set(querystring.strip() for querystring in descriptive_queryset1)
        runtimes_disces = []
        for discovery in ['uni', 'sep', 'ups', 'sps', 'ilm_lossy']:
            current_df = dataframe.loc[(dataframe['iterations'] == abstraction) & (dataframe['algorithm'] == discovery)]
            if discovery == 'ilm_lossy':
                current_df_queryset = current_df.queryset
                ilm_queryset1 = set(current_df_queryset.values[0].replace("{", "").replace("}", "").replace("'","").split(','))
                ilm_queryset = set(querystring.strip() for querystring in ilm_queryset1)
                true_positives = len(descriptive_queryset & ilm_queryset)
                false_negatives = len(descriptive_queryset - ilm_queryset)
                false_positives = len(ilm_queryset - descriptive_queryset)
                f_score = 2* true_positives / (2*true_positives + false_negatives + false_positives)
                avg_runtime_ilm = mean(current_df.time.values)
            else:
                f_score = 1
                runtimes_disces.extend(current_df.time.values)
            for current_time in current_df.time.values:
                results.append([mod, abstraction, discovery, current_time, f_score])
        
        avg_runtime_disces = mean(runtimes_disces)
        rel_time_change = str(round(avg_runtime_ilm / avg_runtime_disces, 2))
        
        tex_result.append([abstraction, len(descriptive_queryset), true_positives, false_positives, rel_time_change])


    columns = ['mode', 'iterations', 'algorithm', 'time', 'f-score']
    acc_dataframe = pd.DataFrame(results, columns=columns)
    acc_dataframe.to_csv(acc_file_path)

    tex_path = f'../experiments/experiment_results/{file_name}.tex'
    columns = ["Abstraction", "$D$(escriptive)", "Found", "$\\neg D$", "Rel Time Change"]
    df_tex = pd.DataFrame(tex_result, columns=columns)
    df_tex[columns[1:]] = df_tex[columns[1:]].apply(pd.to_numeric)

    df_finance = df_tex.loc[df_tex['Abstraction'].str.contains('F')]
    df_google = df_tex.loc[df_tex['Abstraction'].str.contains('G')]

    summarized_results = []
    summarized_columns = ["Datasets", "Avg ${}^{\\text{Time Lossy ILM}}/_{\\text{Time \\sys{}}}$", 
                          "{Desc}(riptive)", "Found", "$\\neg$ {Desc}"]
    finance_list = ['F1b-F3b', df_finance['Rel Time Change'].mean(),  
                    df_finance['$D$(escriptive)'].sum(),  df_finance['Found'].sum(), df_finance['$\\neg D$'].sum()]
    google_list = ['G1b-G3b', df_google['Rel Time Change'].mean(),
                    df_google['$D$(escriptive)'].sum(), df_google['Found'].sum(), df_google['$\\neg D$'].sum()]
    summarized_results.append(finance_list)
    summarized_results.append(google_list)
    df_tex_summarized = pd.DataFrame(summarized_results, columns=summarized_columns)
    s = df_tex_summarized.style.format(precision=2).set_table_styles([
    {'selector': 'toprule', 'props': ':hline;'},
    {'selector': 'midrule', 'props': ':hline;'},
    {'selector': 'bottomrule', 'props': ':hline;'},], overwrite=True).hide(axis='index')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(s.to_latex(column_format='c c c c c'))
    f.close()
    

    # Return to the origin
    os.chdir(CURRENT_WD)
