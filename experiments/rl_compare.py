import os
import sys
import gzip
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

EXPERIMENT_DIR = os.path.abspath(__file__).replace("rl_compare.py", "")
os.chdir(EXPERIMENT_DIR)
os.chdir("../src")
SRC_DIR = os.getcwd()

os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)

# Include files used for discovery here
from datasets.DISCES.DISCES.src.query_multidim import MultidimQuery
from sample_multidim import MultidimSample
from testbench_helper_functions import generate_plots, match_algos

def rl_compare():
    LOGGER.info(f'Running experiment to compare an RL algorithm to our approach')
    file_name = 'rl_compare'
    file_path = f'../experiments/experiment_results/{file_name}.csv'
    results = []

    sample_path = '../datasets/google/google_query2_rl.txt.gz'
    file = gzip.open(sample_path, 'rb')
    repetition = 5
    sample_list_google = []
    counter = 0
    for trace1 in file:
        sample_list_google.append(' '.join(trace1.decode().split()))
        counter += 1
        if counter == 100:
            break

    file.close()

    sample_path = '../datasets/finance/finance_query2_rl.txt.gz'
    file = gzip.open(sample_path, 'rb')
    sample_list_finance = []
    for trace1 in file:
        sample_list_finance.append(' '.join(trace1.decode().split()))

    file.close()
    if not os.path.isfile(file_path):
        for _ in range(repetition):
            results, columns = match_algos(sample_list=sample_list_google, results=results, iterations='google', mod='mod', j='google',
                                discovery=['uni'], file_path=file_path, only_types=True)
            results, columns = match_algos(sample_list=sample_list_finance, results=results, iterations='finance', mod='mod', j='finance',
                                discovery=['uni'], file_path=file_path, only_types=True)


        dataframe = pd.DataFrame(results, columns=columns)
        queryset_bu = dataframe.loc[dataframe.iteration == 'google']['queryset'].values[0]
        rl_google_results = pd.read_csv('../datasets/google/rl_google_cpu_results.txt', sep=', ', engine='python')
        df = pd.concat([dataframe, rl_google_results]) #.to_csv(file_path)
        df.fillna({'algorithm':'rl', 'iterations':'google', 'iteration':'google'}, inplace=True)
        df.to_csv(file_path)
        queryset_bu = dataframe.loc[dataframe.iteration == 'finance']['queryset'].values[0]
        rl_finance_results = pd.read_csv('../datasets/finance/rl_finance_cpu_results.txt', sep=', ', engine='python')
        df = pd.concat([df, rl_finance_results]) #.to_csv(file_path)
        df.fillna({'algorithm':'rl', 'iterations':'finance', 'iteration':'finance'}, inplace=True)
        df.to_csv(file_path)
    else:
        df = pd.read_csv(file_path)

    rl_result = []
    for mod in ['google', 'finance']:
        parent_dict = {}
        dict_iter = {}
        sample = MultidimSample()
        if mod == 'finance':
            sample.set_sample(sample_list_finance)
            dataset = 'F2b'
        else:
            sample.set_sample(sample_list_google)
            dataset = 'G2b'
        alphabet = sample.get_sample_typeset()
        queryset = set()
        descritptive = 0
        non_descritptive = 0
        not_matching = 0
        queryset_object = df.loc[(df.iteration == mod)&(df.algorithm=='rl')].queryset.values[0]
        queryset_bu = df.loc[(df.iteration == mod)&(df.algorithm=='uni')]['queryset'].values[0]
        desc_queryset = set()
        for querystring in queryset_bu:
            if ';' in querystring:
                desc_queryset.add(querystring)
        for querystring in queryset_object.split("'"):
            if ';' in querystring:
                queryset.add(querystring)
                query = MultidimQuery()
                query.set_query_string(querystring)
                query.set_query_matchtest('smarter')
                query.set_pos_last_type_and_variable()
                if querystring in desc_queryset:
                    descritptive += 1
                else:
                    matching = query.match_sample(sample=sample, supp=1.0, dict_iter=dict_iter,
                                                patternset=alphabet, parent_dict=parent_dict)
                    if matching:
                        non_descritptive += 1
                    else:
                        not_matching += 1
        rl_result.append([dataset, len(desc_queryset),descritptive, non_descritptive, not_matching])
    tex_path = f'../experiments/experiment_results/{file_name}.tex'
    columns = ["Dataset", "{Desc}(riptive)", "Found", "$\\neg$ {Desc}", "$\\neg$ Matching"]
    df_tex = pd.DataFrame(rl_result, columns=columns).sort_values(by='Dataset')
    s = df_tex.style.set_table_styles([
    {'selector': 'toprule', 'props': ':hline;'},
    {'selector': 'midrule', 'props': ':hline;'},
    {'selector': 'bottomrule', 'props': ':hline;'},], overwrite=True).hide(axis='index')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(s.to_latex(column_format='c c c c c'))
    return df

if __name__ == "__main__":
    df_rl = rl_compare()
    file_name = 'rl_compare'
    result_path = '../experiments/experiment_results'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    df_rl.to_csv(f'../experiments/experiment_results/{file_name}_results.csv')
    generate_plots(dataframe=df_rl, file_name=file_name, x='iterations', y='time', hue='algorithm', kind='rl_compare')

    # Return to the origin
    os.chdir(CURRENT_WD)
