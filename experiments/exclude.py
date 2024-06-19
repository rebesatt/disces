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

EXPERIMENT_DIR = os.path.abspath(__file__).replace("exclude.py", "")
os.chdir(EXPERIMENT_DIR)
os.chdir("../src")
SRC_DIR = os.getcwd()

os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)

# Include files used for discovery here
from testbench_helper_functions import generate_plots, match_algos

def exclude_types(overwrite: bool = False, max_query_length = -1) -> pd.DataFrame:
    """Experiment for the State of the Art comparison
    Args:
        sample_size (int): number of traces in the sample
        trace_length (int): number of events per trace in the sample
        overwrite (bool, optional): Option to overwrite existing results and rerun experiment. Defaults to False.

    Returns:
        Dataframe: Collected data for each iteration
    """
    LOGGER.info(f'Running experiment for domain exclusion')
    file_name = f'exclude_types'
    file_path = f'../experiments/experiment_results/{file_name}.csv'
    result_path = '../experiments/experiment_results'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    results = []
    if not os.path.isfile(file_path) or overwrite:
        repetition = 5
        run = 0
        discoveries = ['uni', 'sep', 'ups', 'sps']
        sample_path = ''
        for mod in ['finance', 'google']:
            for cluster in ['E1', 'E2', 'E3', 'E4']:
                if mod == 'finance':
                    if cluster == 'E1':
                        sample_path = '../datasets/finance/finance_query1.txt.gz'

                    elif cluster == 'E2':
                        sample_path = '../datasets/finance/finance_query1_vol100.txt.gz'

                    elif cluster == 'E3':
                        sample_path = '../datasets/finance/finance_query1_vol200.txt.gz'

                    elif cluster == 'E4':
                        sample_path = '../datasets/finance/finance_query1_vol400.txt.gz'
                    

                elif mod == 'google':
                    if cluster == 'E1':
                        sample_path = '../datasets/google/google_query1.txt.gz'

                    elif cluster == 'E2':
                        sample_path = '../datasets/google/google_query1_status1.txt.gz'

                    elif cluster == 'E3':
                        sample_path = '../datasets/google/google_query1_status0.txt.gz'

                    elif cluster == 'E4':
                        sample_path = '../datasets/google/google_query1_status2.txt.gz'

                file = gzip.open(sample_path, 'rb')

                sample_list = []
                counter = 0
                sample_size = 200
                trace_length_finance = 80
                trace_length_google = 15
                for trace1 in file:
                    # sample_set2.append(' '.join(trace1.decode().split()))
                    if mod == 'google':
                        sample_list.append(' '.join(trace1.decode().split()[-trace_length_google:]))


                    else:
                        sample_list.append(' '.join(trace1.decode().replace(', ', ',').split()[-trace_length_finance:]))
                    if counter == sample_size:
                        break
                    counter += 1

                file.close()
                for _ in range(repetition):
                    results, columns = match_algos(sample_list=sample_list, results=results, iterations=repetition, mod=mod,
                                          j=cluster, discovery=discoveries, file_path=file_path, max_query_length=max_query_length)
                run += 1
                LOGGER.info(f'Finished {cluster} for {mod}')
                LOGGER.info(f'{run} out of 8')
        dataframe = pd.DataFrame(results, columns=columns)
        dataframe.to_csv(file_path)
    else:
        dataframe = pd.read_csv(file_path)
    return dataframe

if __name__ == "__main__":
    max_query_length = 4
    df_cluster = exclude_types(overwrite=False, max_query_length=max_query_length)
    df_cluster['rel time change'] = df_cluster['time']
    for mod in ['finance', 'google']:
        df_mod = df_cluster.loc[df_cluster['mode'] == mod]
        for algo in ['uni', 'sep', 'ups', 'sps']:
            times = df_mod.loc[(df_mod['algorithm'] == algo) & (df_mod['iterations'] == 'E1')].time.values
            baseline = mean([float(time) for time in times])
            df_algo = df_mod.loc[df_mod['algorithm'] == algo]
            for idx in df_algo.index:
                df_cluster.loc[idx, 'rel time change'] = float(df_cluster.loc[idx, 'time']) / baseline
    file_name = f'exclude'
    generate_plots(dataframe=df_cluster, file_name=file_name, x='iterations', y='rel time change',
                    hue='algorithm', col='mode', kind='exclude', facet_kws={'sharex': False, 'sharey': False})

    # Return to the origin
    os.chdir(CURRENT_WD)
