import os
import sys
import gzip
from statistics import mean
import pandas as pd
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

EXPERIMENT_DIR = os.path.abspath(__file__).replace("cluster.py", "")
os.chdir(EXPERIMENT_DIR)
os.chdir("../src")
SRC_DIR = os.getcwd()

os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)

# Include files used for discovery here
from testbench_helper_functions import generate_plots, match_algos

def clustering(sample_size:int, trace_length:int, overwrite: bool = False, max_query_length = -1) -> pd.DataFrame:
    """Experiment for the State of the Art comparison
    Args:
        sample_size (int): number of traces in the sample
        trace_length (int): number of events per trace in the sample
        overwrite (bool, optional): Option to overwrite existing results and rerun experiment. Defaults to False.

    Returns:
        Dataframe: Collected data for each iteration
    """
    LOGGER.info(f'Running experiment for clustering')
    file_name = f'clustering'
    file_path = f'../experiments/experiment_results/{file_name}.csv'
    result_path = '../experiments/experiment_results'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    results = []
    if not os.path.isfile(file_path) or overwrite:
        repetition = 5
        run = 0
        discoveries = ['uni', 'sep', 'ups', 'sps']
        for mod in ['finance', 'google']:
            if mod == 'finance':
                max_query_length = -1
            else:
                max_query_length = -1
            current_run = f'clustering_{mod}'
            current_path = f'../experiments/experiment_results/{current_run}.csv'
            if not os.path.isfile(current_path):
                for cluster in ['C1', 'C2', 'C3']:
                    if mod == 'finance':
                        if cluster == 'C1':
                            sample_path = '../datasets/finance/finance_query3.txt.gz'

                        elif cluster == 'C2':
                            sample_path = '../datasets/finance/finance_query3_close100.txt.gz'

                        else:
                            sample_path = '../datasets/finance/finance_query3_close50.txt.gz'

                    elif mod == 'google':
                        if cluster == 'C1':
                            sample_path = '../datasets/google/google_query3_priority1.txt.gz'

                        elif cluster == 'C2':
                            sample_path = '../datasets/google/google_query3_priority2.txt.gz'

                        elif cluster == 'C3':
                            sample_path = '../datasets/google/google_query3_priority3.txt.gz'

                    file = gzip.open(sample_path, 'rb')
                    
                    sample_list = []
                    counter = 0
                    for trace1 in file:
                        # sample_set2.append(' '.join(trace1.decode().split()))
                        if mod == 'google':

                            sample_list.append(' '.join(trace1.decode().split()[-9:]))
                        else:
                            sample_list.append(' '.join(trace1.decode().replace(', ', ',').split()[-trace_length:]))
                        if counter == sample_size:
                            break
                        counter += 1

                    file.close()
                    for _ in range(repetition):
                        results, columns = match_algos(sample_list=sample_list, results=results, iterations=repetition, mod=mod,
                                            j=cluster, discovery=discoveries, file_path=file_path, max_query_length=max_query_length)
                    run += 1
                    LOGGER.info(f'Finished {cluster} for {mod}')
                    LOGGER.info(f'{run} out of 6')
                dataframe = pd.DataFrame(results, columns=columns)
                current_df = dataframe.loc[(dataframe['mode'] == mod)]
                current_df.to_csv(f'../experiments/experiment_results/{current_run}.csv')
            else:
                current_df = pd.read_csv(current_path)
                columns = current_df.columns.values[1:]
                current_list = [c_list[1:] for c_list in current_df.values.tolist()]
                results.extend(current_list)

        dataframe = pd.DataFrame(results, columns=columns)
        dataframe.to_csv(file_path)
    else:
        dataframe = pd.read_csv(file_path)
    return dataframe


if __name__ == "__main__":
    sample_size = 100
    trace_length = 40
    max_query_length = -1
    file_name = f'clustering'
    file_path = f'../experiments/experiment_results/{file_name}.csv'
    results = []
    if not os.path.isfile(file_path):
        df_cluster = clustering(sample_size=sample_size, trace_length=trace_length, overwrite=False, max_query_length=max_query_length)
    else:
        df_cluster = pd.read_csv(file_path)
    df_cluster['rel time change'] = df_cluster['time']
    for mod in ['finance', 'google']:
        df_mod = df_cluster.loc[df_cluster['mode'] == mod]
        for algo in ['uni', 'sep', 'ups', 'sps']:
            times = df_mod.loc[(df_mod['algorithm'] == algo) & (df_mod['iterations'] == 'C1')].time.values
            baseline = mean([float(time) for time in times])
            df_algo = df_mod.loc[df_mod['algorithm'] == algo]
            for idx in df_algo.index:
                df_cluster.loc[idx, 'rel time change'] = float(df_cluster.loc[idx, 'time']) / baseline
    file_name = f'cluster'
    generate_plots(dataframe=df_cluster, file_name=file_name, x='iterations', y='rel time change',
                    hue='algorithm', col='mode', kind='cluster', facet_kws={'sharex': False, 'sharey': True})

    # Return to the origin
    os.chdir(CURRENT_WD)
