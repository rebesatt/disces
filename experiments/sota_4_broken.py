import os
import sys
import gzip
import pandas as pd
import logging
import argparse
from argparse import RawTextHelpFormatter

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

CURRENT_WD = os.getcwd()

EXPERIMENT_DIR = os.path.abspath(__file__).replace("sota_4_broken.py", "")
os.chdir(EXPERIMENT_DIR)
os.chdir("../src")
SRC_DIR = os.getcwd()

os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)

# Include files used for discovery here
from testbench_helper_functions import generate_plots, match_algos

def sota(overwrite: bool = False, max_query_length = -1, b_run_ilm : bool = False) -> pd.DataFrame:
    """Experiment for the State of the Art comparison
    Args:
        sample_size (int): number of traces in the sample
        trace_length (int): number of events per trace in the sample
        overwrite (bool, optional): Option to overwrite existing results and rerun experiment. Defaults to False.

    Returns:
        Dataframe: Collected data for each iteration
    """
    LOGGER.info(f'Running experiment for State of the Art comparison')
    file_name = f'sota'
    file_path = f'../experiments/experiment_results/{file_name}.csv'
    results = []
    if not os.path.isfile(file_path) or overwrite:
        repetition = 5
        run = 0
        if b_run_ilm:
            discoveries = ['uni', 'sep', 'ups', 'sps', 'ilm']
        else:
            discoveries = ['uni', 'sep', 'ups', 'sps']
        for abstraction in ['F1', 'F2', 'F3', 'G1', 'G2', 'G3']:
            if abstraction[0] == 'F':
                mod = 'finance'
            else:
                mod = 'google'
            for version in [1, 2]:
                current_run = f'{mod}_{abstraction}_{version}'
                current_path = f'../experiments/experiment_results/{current_run}.csv'
                if not os.path.isfile(current_path):

                    if abstraction == 'F1':
                        sample_path = '../datasets/finance/finance_query1.txt.gz'
                        if version == 2:
                            trace_length = 51
                        else:
                            trace_length = 50
                        sample_size = 79


                    elif abstraction == 'F2':
                        if version == 2:
                            trace_length = 41
                        else:
                            trace_length = 40
                        sample_size = 1000
                        sample_path = '../datasets/finance/finance_query2.txt.gz'

                    elif abstraction == 'F3':
                        if version == 2:
                            trace_length = 26
                        else:
                            trace_length = 25
                        sample_size = 250
                        max_query_length = 5
                        sample_path = '../datasets/finance/finance_query3.txt.gz'


                    elif abstraction == 'G1':
                        if version == 2:
                            trace_length = 8
                        else:
                            trace_length = 7
                        sample_size = 1000
                        sample_path = '../datasets/google/google_query1.txt.gz'


                    elif abstraction == 'G2':
                        if version == 2:
                            trace_length = 8
                        else:
                            trace_length = 7
                        sample_size = 1000
                        sample_path = '../datasets/google/google_query2.txt.gz'

                    elif abstraction == 'G3':
                        if version == 2:
                            trace_length = 8
                        else:
                            trace_length = 7
                        sample_size = 1000
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
                        if version == 2:
                            j=abstraction + 'b'
                        else:
                            j=abstraction + 'a'
                        results, columns = match_algos(sample_list=sample_list, results=results, iterations=version, mod=mod,
                                            j=j, discovery=discoveries, file_path=file_path, max_query_length=max_query_length)
                        run += 1
                        LOGGER.info(f'Finished {run} out of 60: {abstraction} for {mod}, Version {version}, Repetition {rep+1}/{repetition}')
                    dataframe = pd.DataFrame(results, columns=columns)
                    current_df = dataframe.loc[(dataframe['mode'] == mod) & (dataframe['iterations'] == j)]
                    current_df.to_csv(f'../experiments/experiment_results/{current_run}.csv')
                else:
                    current_df = pd.read_csv(current_path)
                    current_list = [c_list[1:] for c_list in current_df.values.tolist()]
                    results.extend(current_list)
                    run += 5
                    LOGGER.info(f'Finished {run} out of 60: {abstraction} for {mod}, Version {version}')
        if os.path.isfile(file_path) and not overwrite:
            dataframe = pd.read_csv(file_path)
        else:
            columns = current_df.columns.values[1:]
            dataframe = pd.DataFrame(results, columns=columns)
            dataframe.to_csv(file_path)
    else:
        dataframe = pd.read_csv(file_path)
    return dataframe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description =  "...\n")
    parser.add_argument("--ilm", dest="b_run_extended_il_miner",
            help="",
            action="store_true")
    args = parser.parse_args()

    file_path = f'../experiments/experiment_results/sota.csv'
    max_query_length = 4
    if not os.path.isfile(file_path):
        df_sota = sota(overwrite=False, max_query_length=max_query_length, b_run_ilm=args.b_run_extended_il_miner)
    else:
        df_sota = pd.read_csv(file_path)
    file_name = f'sota_{max_query_length}'
    generate_plots(dataframe=df_sota, file_name=file_name, x='iterations', y='time', hue='algorithm', kind='sota')
    # save plot of {experiment_name} to 'experiment_results/{experiment_name}.pdf'
    # Example: plot of 'sota_4_broken' to 'experiment_results/sota_4_broken.pdf'

    # Return to the origin
    os.chdir(CURRENT_WD)
