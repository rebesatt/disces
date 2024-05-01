import os
import sys
import gzip
import pandas as pd
import time
import logging
import collections

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

CURRENT_WD = os.getcwd()

EXPERIMENT_DIR = os.path.abspath(__file__).replace("correlation.py", "")
os.chdir(EXPERIMENT_DIR)
os.chdir("../src")
SRC_DIR = os.getcwd()

os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)

# Include files used for discovery here
from testbench_helper_functions import generate_plots, match_algos
from sample_multidim import MultidimSample

def char_correlation(overwrite: bool = False) -> pd.DataFrame:
    """_summary_

    Args:
        overwrite (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    file_path = '../experiments/experiment_results/corr_data.csv'
    result_path = '../experiments/experiment_results/corr_results.csv'
    path = '../experiments/experiment_results'
    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isfile(result_path) or overwrite:
        discoveries = ['uni', 'sep', 'ups', 'sps']
        result_list = []
        # for mod in ['finance', 'google']:
        for abstraction in ['F1', 'F2', 'F3', 'G1', 'G2', 'G3']:
            if abstraction[0] == 'F':
                mod = 'finance'
            else:
                mod = 'google'
            # for version in [1, 2]:
            LOGGER.info('%s %s', mod, abstraction)
            
            # if mod == 'finance':
            if abstraction == 'F1':
                sample_path = '../datasets/finance/finance_query1.txt.gz'
                trace_length = 70
                sample_size = 79

            elif abstraction == 'F2':
                trace_length = 70
                sample_size = 100
                sample_path = '../datasets/finance/finance_query2.txt.gz'
            elif abstraction == 'F3':
                trace_length = 40
                sample_size = 10
                sample_path = '../datasets/finance/finance_query3.txt.gz'
                
            elif abstraction == 'G1':
                trace_length = 12
                sample_size = 300
                sample_path = '../datasets/google/google_query1.txt.gz'


            elif abstraction == 'G2':
                trace_length = 10
                sample_size = 1000
                sample_path = '../datasets/google/google_query2.txt.gz'

            elif abstraction == 'G3':
                trace_length = 10
                sample_size = 1000
                sample_path = '../datasets/google/google_query3.txt.gz'

            
            file = gzip.open(sample_path, 'rb')
            sample_list = []

            counter = 0
            for trace1 in file:
                if counter == sample_size:
                    break
                
                if trace_length == -1:
                    sample_list.append(' '.join(trace1.decode().split()))
                else:
                    sample_list.append(' '.join(trace1.decode().split()[-trace_length:]))

                counter += 1
            file.close()
            current_run = f'corr_{mod}_{abstraction}_{sample_size}_{trace_length}'
            current_path = f'../experiments/experiment_results/{current_run}.csv'
            if not os.path.isfile(current_path):
                max_query_length = -1
                results = []
                results, columns = match_algos(sample_list=sample_list, results=results, iterations=1, mod=mod,
                                        j=abstraction, discovery=discoveries, file_path=file_path, max_query_length=max_query_length)
                dataframe = pd.DataFrame(results, columns=columns)
                dataframe.to_csv(current_path)
                
            else:
                current_df = pd.read_csv(current_path)
                current_list = [c_list[1:] for c_list in current_df.values.tolist()]
                results = current_list

            result_list.append(get_features(sample_list, results))
            result_df = pd.DataFrame(result_list, columns=['uni', 'sep', 'ups', 'sps', 'sum_pattern', 'sum_types', 'domainsize', 'number_traces', 'trace_length', 'number_types'])
            result_df.to_csv(result_path)
        
        return result_df

def get_features(sample_list: list, results: list):
    result_dict = []
    sample = MultidimSample()
    sample.set_sample(sample_list)
    number_traces = len(sample_list)
    alphabet = sample.get_sample_supported_typeset()
    domain_sample_dict = sample.get_dim_sample_dict()
    max_pattern = 0
    max_sum_pattern = 0
    max_types = 0
    max_sum_types = 0
    domainsize = 0
    trace_length = 0
    number_types = 0

    max_pattern_list = []
    sum_pattern_list = []
    max_type_list = []
    sum_type_list = []

    for trace_id in range(number_traces):
        max_pattern_dom_list = []
        sum_pattern_dom_list = []
        max_type_dom_list = []
        sum_type_dom_list = []
        for dom_sample in domain_sample_dict.values():
            supported_alphabet = dom_sample.get_sample_supported_typeset()
            trace_list = dom_sample._sample[trace_id].split()
            event_counter = collections.Counter(trace_list)
            freq_list = event_counter.most_common()
            if freq_list[0][1] > 1:
                max_pattern_dom_list.append(freq_list[0][1])
            pattern_sum = sum(tup[1] for tup in freq_list if tup[1] > 1)
            sum_pattern_dom_list.append(pattern_sum)
            type_sum = sum(tup[1] for tup in freq_list if tup[0].replace(';','') in supported_alphabet)
            sum_type_dom_list.append(type_sum)
            for tup in freq_list:
                if tup[0].replace(';','') in supported_alphabet:
                    max_type_dom_list.append(tup[1])
                    break
        max_pattern_list.append(max(max_pattern_dom_list, default=0))
        sum_pattern_list.append(max(sum_pattern_dom_list, default=0))
        max_type_list.append(max(max_type_dom_list, default=0))
        sum_type_list.append(max(sum_type_dom_list, default=0))

    max_pattern = min(max_pattern_list, default=0)
    max_sum_pattern = min(sum_pattern_list, default=0)
    max_types = max(max_type_list, default=0)
    max_sum_types = max(sum_type_list, default=0)
    domainsize = len(domain_sample_dict)
    trace_length = len(sample_list[0].split())
    number_types = len(alphabet)

    result_list = [results[0][4],results[1][4],results[2][4],results[3][4], max_sum_pattern, max_sum_types, domainsize, number_traces, trace_length, number_types]

    return result_list


if __name__ == "__main__":

    file_path = '../experiments/experiment_results/corr_results.csv'
    if not os.path.isfile(file_path):
        df_corr = char_correlation(overwrite=False)
    else:
        df_corr = pd.read_csv(file_path)

    # df_corr = df_corr.loc[df_corr['iterations'] == 'F1']
    tex_path = '../experiments/experiment_results/char_corr.tex'
    df_corr.rename(columns={'sum_pattern': '$V$','sum_types': '$Y$','domainsize': '$\\Delta$','number_types': '$T$'}, inplace=True)
    slice_ = ['uni', 'sep', 'ups', 'sps', '$V$', '$Y$','$\\Delta$','$T$']
    df_corr_adapted = df_corr[slice_]
    df_corr_adapted.columns = pd.MultiIndex.from_tuples([
    ("Algorithm runtime", "uni"),
    ("Algorithm runtime", "sep"),
    ("Algorithm runtime", "ups"),
    ("Algorithm runtime", "ups"),
    ("Database properties", '$V$'),
    ("Database properties", '$Y$'),
    ("Database properties", '$\\Delta$'),
    ("Database properties", '$T$'),])
    # s = df_corr.style.background_gradient(subset=slice_).hide(axis='index')
    
    s = df_corr_adapted.style.set_table_styles([
    {'selector': 'toprule', 'props': ':hline;'},
    {'selector': 'midrule', 'props': ':hline;'},
    {'selector': 'bottomrule', 'props': ':hline;'},], overwrite=True).format(precision=2).hide(axis='index')
    
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(s.to_latex(convert_css=True, 
                           column_format="|r|r|r|r|r|r|r|r|", multicol_align='c|'))
    # Return to the origin
    os.chdir(CURRENT_WD)
