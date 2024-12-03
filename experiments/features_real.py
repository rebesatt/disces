import os
import sys
import gzip
import pandas as pd
import time
import logging
import collections
from copy import deepcopy


#Logger Configuration:
LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

CURRENT_WD = os.getcwd()

EXPERIMENT_DIR = os.path.abspath(__file__).replace("features_real.py", "")
os.chdir(EXPERIMENT_DIR)
os.chdir("../src")
SRC_DIR = os.getcwd()

os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)

# Include files used for discovery here
from testbench_helper_functions import generate_plots, match_algos
from sample_multidim import MultidimSample
import seaborn as sns


def attributes(overwrite:bool = False) -> pd.DataFrame:
    file_path = '../experiments/experiment_results/rw_features/feature_ext.csv'
    # result_path = '../experiments/experiment_results/rw_features/feat_ext_results.csv'
    path = '../experiments/experiment_results/rw_features'
    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isfile(file_path) or overwrite:
        discoveries = ['uni', 'sep', 'ups', 'sps']
        results = []
        for abstraction in ['G1', 'G2', 'G3']:
            mod = 'google'


            if abstraction == 'G1':
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

            att_typeset, dim_sets, dom_trace_dict = samplelist2dict(sample_list)
            priorities = sorted([int(prio) for prio in att_typeset[3].keys()])
            status = sorted([int(stat) for stat in att_typeset[2].keys()])
            priority_dict = {prio:count for count,prio in enumerate(priorities)}
            # for count, prio in enumerate(priorities):
            new_sample_stat_list = []
            new_sample_stat_prio_list = []
            sample_split = [trace.split() for trace in sample_list]
            for trace_split in sample_split:
                new_trace_status_split = []
                new_trace_stat_prio_split = []
                for event in trace_split:
                    event_split = event.split(';')

                    new_event_dim_status = ';'*len(status)
                    new_event_dim_status_split = new_event_dim_status.split(';')
                    new_event_dim_status_split[int(event_split[2])] = str(event_split[2])

                    new_event_dim_prio = ';'*len(priorities)
                    new_event_dim_prio_split = new_event_dim_prio.split(';')
                    new_event_dim_prio_split[priority_dict[int(event_split[3])]] = str(event_split[3])

                    new_event_status= ';'.join(event_split[:2] + new_event_dim_status_split + event_split[3:])
                    new_event_status_prio= ';'.join(event_split[:2] + new_event_dim_status_split + new_event_dim_prio_split + event_split[4:])
                    new_trace_status_split.append(new_event_status)
                    new_trace_stat_prio_split.append(new_event_status_prio)
                new_sample_stat_list.append(' '.join(new_trace_status_split))
                new_sample_stat_prio_list.append(' '.join(new_trace_stat_prio_split))

            for samp_list, name_ext in zip([sample_list, new_sample_stat_list, new_sample_stat_prio_list], ['gen', 'status', 'statprio']):
                att_count = samp_list[0].split()[0].count(';')
                current_run = f'att_real_{abstraction}_{sample_size}_{trace_length}_{name_ext}'
                current_path = f'../experiments/experiment_results/rw_features/{current_run}.csv'
                if not os.path.isfile(current_path):
                    max_query_length = -1

                    results, columns = match_algos(sample_list=samp_list, results=results, iterations=att_count, mod=mod,
                                            j=abstraction, discovery=discoveries, file_path=file_path, max_query_length=max_query_length)
                    dataframe = pd.DataFrame(results, columns=columns)
                    # current_dataframe = dataframe.loc[(dataframe['iterations']== abstraction)&(dataframe['iteration']==att_count)]
                    dataframe.to_csv(current_path)

                else:
                    current_df = pd.read_csv(current_path, header =0, index_col=0)
                    columns = current_df.columns
                    # current_list = [c_list for c_list in current_df.values.tolist()]
                    results.extend(current_df.values.tolist())

                # result_list.append(get_features(sample_list, results))
                #result_df = pd.DataFrame(result_list, columns=['uni', 'sep', 'ups', 'sps', 'sum_pattern', 'sum_types', 'domainsize', 'number_traces', 'trace_length', 'number_types'])
                result_df = pd.DataFrame(results, columns=columns)
                result_df.to_csv(file_path)


    else:
        result_df = pd.read_csv(file_path, header=0, index_col=0)

    return result_df


def alphabet(overwrite:bool = False) -> pd.DataFrame:
    file_path = '../experiments/experiment_results/rw_features/alphabet.csv'
    # result_path = '../experiments/experiment_results/rw_features/feat_ext_results.csv'
    path = '../experiments/experiment_results/rw_features'
    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isfile(file_path) or overwrite:
        discoveries = ['uni', 'sep', 'ups', 'sps']
        results = []
        for abstraction in ['G1', 'G2', 'G3']:
            mod = 'google'


            if abstraction == 'G1':
                max_query_length = 3
                trace_length = 5
                sample_size = 30
                sample_path = '../datasets/google/google_query1.txt.gz'
                trace_id = 15
                machine_mod = 10
                job_mod = 5


            elif abstraction == 'G2':
                max_query_length = 3
                trace_length = 5
                sample_size = 1000
                sample_path = '../datasets/google/google_query2.txt.gz'
                trace_id = 9
                machine_mod = 5
                job_mod = 100

            elif abstraction == 'G3':
                max_query_length = 3
                trace_length = 6
                sample_size = 1000
                sample_path = '../datasets/google/google_query3.txt.gz'
                trace_id = 13
                machine_mod = 5
                job_mod = 100


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

            new_list = []
            new_sample_machine_job_list = []
            sample_split = [trace.split() for trace in sample_list]

            # for trace in sample_list[:10]:
            #     trace_split = trace.split()
            #     new_trace_list = []
            #     new_trace_list2 = []
            #     for event in trace_split:
            #         event_split = event.split(';')
            #         new_trace_list.append(';'.join([str(int(event_split[0])%2)]+ event_split[1:]))
            #         if event_split[1]:
            #             new_trace_list2.append(';'.join([str(int(event_split[0])%2)]+ [str(int(event_split[1])%2)] + event_split[2:]))
            #         else:
            #             new_trace_list2.append(';'.join([str(int(event_split[0])%2)]+ event_split[1:]))
            #     new_sample_machine_list.append(' '.join(new_trace_list))
            #     new_sample_machine_job_list.append(' '.join(new_trace_list2))

            for i in range(5):
            #     sample_list[0]
            
            # for samp_list, name_ext in zip([sample_list, new_sample_machine_list, sample_list[:10]],
            #                                ['1', '2', '3' ]):
                new_list.append(' '.join(sample_split[trace_id][i:]))
                att_count = new_list[0].split()[0].count(';')
                current_run = f'alphabet_{abstraction}_{i}'
                current_path = f'../experiments/experiment_results/rw_features/{current_run}.csv'
                if not os.path.isfile(current_path):

                    results, columns = match_algos(sample_list=new_list, results=results, iterations=att_count, mod=mod,
                                            j=abstraction, discovery=discoveries, file_path=file_path, max_query_length=max_query_length)
                    dataframe = pd.DataFrame(results, columns=columns)
                    current_dataframe = dataframe.loc[(dataframe['iterations']== abstraction) & (dataframe['sample size']== str(i+1))]
                    current_dataframe.to_csv(current_path)

                else:
                    current_df = pd.read_csv(current_path, header =0, index_col=0)
                    columns = current_df.columns
                    # current_list = [c_list for c_list in current_df.values.tolist()]
                    results.extend(current_df.values.tolist())

                # result_list.append(get_features(sample_list, results))
                #result_df = pd.DataFrame(result_list, columns=['uni', 'sep', 'ups', 'sps', 'sum_pattern', 'sum_types', 'domainsize', 'number_traces', 'trace_length', 'number_types'])
                result_df = pd.DataFrame(results, columns=columns)
                result_df.to_csv(file_path)


    else:
        result_df = pd.read_csv(file_path, header=0, index_col=0)

    return result_df

def max_type(overwrite:bool = False) -> pd.DataFrame:
    file_path = '../experiments/experiment_results/rw_features/types.csv'
    # result_path = '../experiments/experiment_results/rw_features/feat_ext_results.csv'
    path = '../experiments/experiment_results/rw_features'
    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isfile(file_path) or overwrite:
        discoveries = ['uni', 'sep', 'ups', 'sps']
        results = []
        for abstraction in ['G1', 'G2', 'G3']:
            mod = 'google'


            if abstraction == 'G1':
                trace_length = 8
                sample_size = 30
                att_change = '1'
                sample_path = '../datasets/google/google_query1.txt.gz'


            elif abstraction == 'G2':
                trace_length = 8
                sample_size = 10
                att_change = '4'
                sample_path = '../datasets/google/google_query2.txt.gz'


            elif abstraction == 'G3':
                trace_length = 6
                sample_size = 10
                att_change = '2'
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


            sample_split = [trace.split() for trace in sample_list]
            att_count = sample_list[0].split()[0].count(';')
            gen_event = att_count * ';'
            gen_event_list = [i for i in gen_event]
            # new_sample_machine_list.append(' '.join(new_trace_machine_split))
            # new_sample_machine_job_list.append(' '.join(new_trace_machine_job_split))
            new_sample_2_list = []
            new_sample_3_list = []
            for trace_id, trace_split in enumerate(sample_split):
                new_trace_2_list = []
                new_trace_3_list = []
                if trace_id <= len(sample_split)*0.1:
                    for event in trace_split:
                        event_split = event.split(';')
                        new_event = ';'.join([att if att== att_change 
                                              else '' for att in event_split])

                        new_trace_2_list.append(event)
                        new_trace_3_list.append(event)
                        if new_event != gen_event:
                            for _ in range(5):
                                new_trace_2_list.append(new_event)
                                new_trace_3_list.append(new_event)
                                new_trace_3_list.append(new_event)
                    new_sample_2_list.append(' '.join(new_trace_2_list))
                    new_sample_3_list.append(' '.join(new_trace_3_list))

                else:
                    new_sample_2_list.append(' '.join(trace_split))
                    new_sample_3_list.append(' '.join(trace_split))

            for samp_list, name_ext in zip([sample_list, new_sample_2_list, new_sample_3_list],
                                           ['1', '2', '3' ]):
                trace_length = samp_list[0].count(' ') + 1
                current_run = f'types_{abstraction}_{name_ext}'
                current_path = f'../experiments/experiment_results/rw_features/{current_run}.csv'
                if not os.path.isfile(current_path):
                    max_query_length = -1
                    results, columns = match_algos(sample_list=samp_list, results=results, iterations=name_ext, mod=mod,
                                            j=abstraction, discovery=discoveries, file_path=file_path, max_query_length=max_query_length)
                    dataframe = pd.DataFrame(results, columns=columns)
                    current_dataframe = dataframe.loc[(dataframe['iterations']== abstraction) &
                                                      (dataframe['iteration']== name_ext)]
                    current_dataframe.to_csv(current_path)

                else:
                    current_df = pd.read_csv(current_path, header =0, index_col=0)
                    columns = current_df.columns
                    # current_list = [c_list for c_list in current_df.values.tolist()]
                    results.extend(current_df.values.tolist())

                # result_list.append(get_features(sample_list, results))
                #result_df = pd.DataFrame(result_list, columns=['uni', 'sep', 'ups', 'sps', 'sum_pattern', 'sum_types', 'domainsize', 'number_traces', 'trace_length', 'number_types'])
                result_df = pd.DataFrame(results, columns=columns)
                result_df.to_csv(file_path)


    else:
        result_df = pd.read_csv(file_path, header=0, index_col=0)

    return result_df

def max_pattern(overwrite:bool = False) -> pd.DataFrame:
    mode = 'pattern'
    file_path = f'../experiments/experiment_results/rw_features/{mode}.csv'
    # result_path = '../experiments/experiment_results/rw_features/feat_ext_results.csv'
    path = '../experiments/experiment_results/rw_features'
    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isfile(file_path) or overwrite:
        discoveries = ['uni', 'sep', 'ups', 'sps']
        results = []
        for abstraction in ['G1', 'G2', 'G3']:
            mod = 'google'


            if abstraction == 'G1':
                trace_length = 7
                sample_size = 3000
                att_change = '1'
                sample_path = '../datasets/google/google_query1.txt.gz'


            elif abstraction == 'G2':
                trace_length = 7
                sample_size = 1000
                att_change = '4'
                sample_path = '../datasets/google/google_query2.txt.gz'


            elif abstraction == 'G3':
                trace_length = 6
                sample_size = 1000
                att_change = '2'
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


            sample_split = [trace.split() for trace in sample_list]
            att_count = sample_list[0].split()[0].count(';')
            gen_event = att_count * ';'
            gen_event_list = [i for i in gen_event]
            # new_sample_machine_list.append(' '.join(new_trace_machine_split))
            # new_sample_machine_job_list.append(' '.join(new_trace_machine_job_split))
            new_sample_2_list = []
            new_sample_3_list = []
            for trace_id, trace_split in enumerate(sample_split):
                new_trace_2_list = [event for event in trace_split]
                new_trace_3_list = [event for event in trace_split]
                event_split = trace_split[0].split(';')
                new_event2 = ';'.join([att if att== event_split[0] 
                                            else '' for att in event_split])

                new_event3 = ';'.join([att if att== event_split[3] 
                                            else '' for att in event_split])
                new_trace_2_list.append(new_event2)
                new_trace_3_list.append(new_event2)
                new_trace_3_list.append(new_event3)

                new_trace_3_list.append(new_event2)
                new_trace_2_list.append(new_event3)
                new_trace_3_list.append(new_event3)

                new_sample_2_list.append(' '.join(new_trace_2_list))
                new_sample_3_list.append(' '.join(new_trace_3_list))

            for samp_list, name_ext in zip([sample_list, new_sample_2_list, new_sample_3_list],
                                           ['1', '2', '3' ]):
                trace_length = samp_list[0].count(' ') + 1
                current_run = f'{mode}_{abstraction}_{name_ext}'
                current_path = f'../experiments/experiment_results/rw_features/{current_run}.csv'
                if not os.path.isfile(current_path):
                    max_query_length = -1
                    results, columns = match_algos(sample_list=samp_list, results=results, iterations=name_ext, mod=mod,
                                            j=abstraction, discovery=discoveries, file_path=file_path, max_query_length=max_query_length)
                    dataframe = pd.DataFrame(results, columns=columns)
                    current_dataframe = dataframe.loc[(dataframe['iterations']== abstraction) &
                                                      (dataframe['iteration']== name_ext)]
                    current_dataframe.to_csv(current_path)

                else:
                    current_df = pd.read_csv(current_path, header =0, index_col=0)
                    columns = current_df.columns
                    # current_list = [c_list for c_list in current_df.values.tolist()]
                    results.extend(current_df.values.tolist())

                # result_list.append(get_features(sample_list, results))
                #result_df = pd.DataFrame(result_list, columns=['uni', 'sep', 'ups', 'sps', 'sum_pattern', 'sum_types', 'domainsize', 'number_traces', 'trace_length', 'number_types'])
                result_df = pd.DataFrame(results, columns=columns)
                result_df.to_csv(file_path)


    else:
        result_df = pd.read_csv(file_path, header=0, index_col=0)

    return result_df

def attributes_v2(overwrite:bool = False) -> pd.DataFrame:
    mode = 'attributes'
    file_path = f'../experiments/experiment_results/rw_features/{mode}.csv'
    # result_path = '../experiments/experiment_results/rw_features/feat_ext_results.csv'
    path = '../experiments/experiment_results/rw_features'
    if not os.path.isdir(path):
        os.mkdir(path)

    mod = 'google'
    short_mod = 'G'
    if not os.path.isfile(file_path) or overwrite:
        discoveries = ['uni', 'sep', 'ups', 'sps']
        results = []
        for abstraction in [f'{short_mod}1', f'{short_mod}2', f'{short_mod}3']:
            if abstraction == f'{short_mod}1':
                trace_length = 6
                sample_size = 300
                max_query_length = -1
                att1 = 2
                att2 = 3
                sample_path = f'../datasets/{mod}/{mod}_query1.txt.gz'


            elif abstraction == f'{short_mod}2':
                trace_length = 5
                sample_size = 10000
                max_query_length = 3
                att1 = 0
                att2 = 0
                sample_path = f'../datasets/{mod}/{mod}_query2.txt.gz'


            elif abstraction == f'{short_mod}3':
                trace_length = 4
                sample_size = 10000
                max_query_length = -1
                att1 = 1
                att2 = 0
                sample_path = f'../datasets/{mod}/{mod}_query3.txt.gz'



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


            sample_split = [trace.split() for trace in sample_list]
            new_sample_list = []
            new_sample_2_list = []
            new_sample_3_list = []
            for trace_split in sample_split:
                new_trace_1_list = []
                new_trace_2_list = []
                new_trace_3_list = []
                for event in trace_split:
                    if abstraction == f'{short_mod}1':
                        event_split = ['' if cnt in [3] and att=='1' else att for cnt,att in
                                    enumerate(event.split(';'))]
                        event_split1 = ['' if cnt in [3] and att=='1' else att for cnt,att in
                                    enumerate(event.split(';'))][:-1]
                        event_split2 = ['' if cnt in [3] and att=='1' else att for cnt,att in
                                    enumerate(event.split(';'))][:-1]
                    elif abstraction == f'{short_mod}2':
                        event_split = ['' if cnt in [2,3] and att=='1' else att for cnt,att in
                                    enumerate(event.split(';'))]
                        event_split1 = ['' if cnt in [2,3] and att=='1' else att for cnt,att in
                                    enumerate(event.split(';'))][:-1]
                        event_split2 = ['' if cnt in [2,3] and att=='1' else att for cnt,att in
                                    enumerate(event.split(';'))][:-1]
                    else:
                        event_split = event.split(';')
                        event_split1 = event.split(';')[:-1]
                        event_split2 = event.split(';')[:-1]

                    for _ in range(3):
                        event_split1.append(event_split1[att1])
                        event_split2.append(event_split2[att1])
                        event_split2.append(event_split2[att1])
                    for _ in range(2):

                        event_split1.append(event_split1[att2])
                        event_split2.append(event_split2[att2])
                        event_split2.append(event_split2[att2])

                    new_event1 = ';'.join(event_split1)+';'
                    new_event2 = ';'.join(event_split2)+';'
                    new_trace_1_list.append(';'.join(event_split))
                    new_trace_2_list.append(new_event1)
                    new_trace_3_list.append(new_event2)
                new_sample_list.append(' '.join(new_trace_1_list))
                new_sample_2_list.append(' '.join(new_trace_2_list))
                new_sample_3_list.append(' '.join(new_trace_3_list))


            for samp_list, att_change in zip([new_sample_list, new_sample_2_list, new_sample_3_list], [0,5,10]):
                trace_length = samp_list[0].count(' ') + 1
                # att_count = samp_list[0].split()[0].count(';')
                current_run = f'{mode}_{abstraction}_{att_change}'
                current_path = f'../experiments/experiment_results/rw_features/{current_run}.csv'
                if not os.path.isfile(current_path):
                    
                    results, columns = match_algos(sample_list=samp_list, results=results, iterations=att_change, mod=mod,
                                            j=abstraction, discovery=discoveries, file_path=file_path, max_query_length=max_query_length)
                    dataframe = pd.DataFrame(results, columns=columns)
                    current_dataframe = dataframe.loc[(dataframe['iterations']== abstraction) &
                                                      (dataframe['iteration']== str(att_change))]
                    current_dataframe.to_csv(current_path)

                else:
                    current_df = pd.read_csv(current_path, header =0, index_col=0)
                    columns = current_df.columns
                    # current_list = [c_list for c_list in current_df.values.tolist()]
                    results.extend(current_df.values.tolist())

                # result_list.append(get_features(sample_list, results))
                #result_df = pd.DataFrame(result_list, columns=['uni', 'sep', 'ups', 'sps', 'sum_pattern', 'sum_types', 'domainsize', 'number_traces', 'trace_length', 'number_types'])
                result_df = pd.DataFrame(results, columns=columns)
                result_df.to_csv(file_path)


    else:
        result_df = pd.read_csv(file_path, header=0, index_col=0)

    return result_df

def samplelist2dict(sample_list:list):
    dim_count = sample_list[0].split()[0].count(';')
    att_typeset = {key: {} for key in range(dim_count)}
    dim_sets = {key: set() for key in range(dim_count)}
    dom_trace_dict = {key: {} for key in range(dim_count)}
    for trace_id, trace in enumerate(sample_list):
        # if trace_id in df1.index:
        # print(trace_id)
        dim_sets_new = {key: set() for key in range(dim_count)}
        for dom in dom_trace_dict.keys():
            dom_trace_dict[dom][trace_id]={} 
        for event in trace.split():
            for dom, letter in enumerate(event.split(';')[:-1]):
                if letter in att_typeset[dom]:
                    att_typeset[dom][letter].add(trace_id)
                else:
                    att_typeset[dom][letter] = set()
                    att_typeset[dom][letter].add(trace_id)
                    
                if letter in dom_trace_dict[dom][trace_id]:
                    dom_trace_dict[dom][trace_id][letter] +=1
                else:
                    dom_trace_dict[dom][trace_id][letter] =1
                dim_sets_new[dom].add(letter)
                    
        if trace_id != 0:
            for i in  dim_sets.keys():
                dim_sets[i] = dim_sets[i] & dim_sets_new[i]
                
        else:
            for i in  dim_sets.keys():
                dim_sets[i] = dim_sets_new[i]
    return att_typeset, dim_sets, dom_trace_dict

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


def gen_plot(dataframe, x, file_name, facet_kws):
    sns.set_context('poster', font_scale=1.2)
    dataframe.replace({'uni': 'D-U-C', 'sep': 'D-U-S',
                       'ups': 'B-S-C', 'sps': 'B-S-S'}, inplace=True)
    replace_dict ={'attributes': r'$|\mathcal{A}|$',
                       'supported types': r'$|\Gamma_D|$', 'type sum': r'$\rho_S$', 'pattern sum':r'$\rho_R$'} 
    if x in replace_dict:
        x = replace_dict[x]
    dataframe.rename(columns = replace_dict, inplace=True)
    palett= sns.color_palette().as_hex()[:4]
    new_palett = {'D-U-C': palett[0],
                  'D-U-S': palett[3], 
                  'B-S-C': palett[1], 
                  'B-S-S': palett[2]}
    grid = sns.relplot(data=dataframe, y='time', x=x, hue='algorithm', err_style='bars',
                           kind='line', col='iterations', style='algorithm', markers=True,
                           facet_kws=facet_kws, palette=new_palett)
    grid.set(yscale='log', yticks=[1, 10, 100])
    grid.set_titles(col_template= "{col_name}")
    grid.savefig(f'../experiments/experiment_results/{file_name}.pdf')

    dataframe['rel timechange'] = 0
    for abstraction in dataframe['iterations'].unique():
        for algorithm in dataframe['algorithm'].unique():
            df = dataframe.loc[(dataframe['iterations']== abstraction)&(dataframe['algorithm']==algorithm)]
            first_value = min(df['time'].to_list())
            first_value = df.loc[df['iteration'] == df['iteration'].min()]['time'].to_list()[0]

            for idx in df.index:
                dataframe.loc[idx, 'rel timechange'] = float(df.loc[df.index == idx]['time'].to_list()[0])/float(first_value)

    grid2 = sns.relplot(data=dataframe, y='rel timechange', x=x, hue='algorithm', err_style='bars',
                           kind='line', col='iterations', style='algorithm', markers=True,
                           facet_kws=facet_kws, palette=new_palett)
    grid2.set(yscale='log', yticks=[1, 10, 100])
    grid2.savefig(f'../experiments/experiment_results/{file_name}_rel.pdf')


if __name__ == "__main__":

    df = attributes()
    df['attributes'] = df['iteration']
    gen_plot(df, x = 'attributes', file_name='rw_attributes',
             facet_kws={'sharey': True, 'sharex': False})

    df = attributes_v2()
    df['attributes'] = df['iteration']
    gen_plot(df, x = 'attributes', file_name='rw_attributes_v2',
             facet_kws={'sharey': False, 'sharex': False})

    df = alphabet()
    df['iteration'] = df['trace length']
    gen_plot(df, x = 'supported types', file_name='rw_alphabet',
             facet_kws={'sharey': False, 'sharex': False})

    df = max_type()
    gen_plot(df, x = 'type sum', file_name='rw_types',
             facet_kws={'sharey': False, 'sharex': False})

    df = max_pattern()
    gen_plot(df, x = 'pattern sum', file_name='rw_pattern',
             facet_kws={'sharey': False, 'sharex': False})
    os.chdir(CURRENT_WD)
