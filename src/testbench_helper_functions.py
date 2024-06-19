#!/usr/bin/python3
"""Contains functions to read and write samples to / from files and generate experiments"""
import os
import re
import logging
import time
import gzip
import msgpack
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import collections
from itertools import product
from statistics import mean
from math import isnan
from copy import deepcopy
import func_timeout

from generator import SampleGenerator
from generator_multidim import MultidimSampleGenerator
from discovery_bu_multidim import bu_discovery_multidim
from sample_multidim import MultidimSample
from discovery_il_miner import il_miner
from discovery_bu_pts_multidim import discovery_bu_pts_multidim

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

FILENAME_REGEXP = '^(?P<sample_size>[^\\.]*)\\.(?P<min_trace_length>[0-9]*)\\.(?P<max_trace_length>[0-9]*)\\.(?P<type_length>[-]*[0-9]*)\\.(?P<event_dimension>[-]*[0-9]*)\\.(?P<dataset>[^\\.]*)\\.seq'

def _write_sample(sample:list, filename:str, dirname:str='samples') -> None:
    """
        Packs a _sample, described in class (Multidim)Sample, in a binary file.

        Note that _sample is only an attribut of a (Multidim)Sample instance,
        i.e. a list of strings, where each string represents a trace.

        Args:
            sample: The sample, which has to be stored, as a list.

            filename: A string which stores the name of the binary file the
            user wants to create.

            dirname: A string which stores the name of the directory where the
            file should be stored. Default is 'samples'.

    """
    LOGGER.debug('_write_sample - Starting')
    if len(sample)==0:
        raise RuntimeError("No sample given")

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    with open(dirname+'/'+filename+'.msgpack', 'wb') as outfile:
        msgpack.pack(sample, outfile, use_bin_type=True)
    LOGGER.debug('_write_sample - Finished')

def _read_sample(filename:str, dirname:str="sample") -> list:
    """
        Loads and unpacks a binary file which contains a list of traces.

        Args:
            filename: A string which stores the name of the binary file the
            user wants to read.

        Returns:
            The list of traces which is stored in file.
    """
    LOGGER.debug('_read_sample - Starting')
    if filename is None:
        raise RuntimeError("No name of file given")
    with open(dirname+'/'+filename+'.msgpack', 'rb') as read_file:
        sample = msgpack.unpack(read_file, raw=False)

    LOGGER.debug('_read_sample - Finished')
    return sample

def _dataset_to_sample(path_to_file:str, cutoff=-1, trace_len=-1, multidim:bool=True, types:list|None=None) -> None:
    """
        Loads a txt.gz file and writes a binary file.

        Args:
            path_to_file: String containing the path to a specific dataset
                which should be converted to a sample. Dataset should be stored
                as '*txt.gz'. We assume the txt.gz file to be in the correct
                format, i.e. each line consists of types, separated by a
                whitespace.

            cutoff [=-1]: If set to an integer >0, only the last cutoff events
                of each trace in file build the trace which is added to the
                sample.

            multidim [=True]: If True calculates event dimension for filename.

            types [=None]: List of types which have to be contained in a trace.
                A trace is only added to the sample, if it contains all types
                in this list.
    """
    LOGGER.info('_dataset_to_sample - Starting')
    LOGGER.info('_dataset_to_sample - Filename: %s', path_to_file)
    #path_to_filename = 'datasets/google/GoogleTraces_BTW23_using_ILMiner_Queries/'+filename
    path_to_filename = path_to_file
    if not os.path.exists(path_to_filename):
        LOGGER.info('_dataset_to_sample - Filename %s does not exist!', path_to_file)
        return
    file = gzip.open(path_to_filename, 'rb')
    sample = []
    line_counter = 0
    sample_min_trace = ""
    sample_max_trace = ""
    start = time.time()
    for trace in file:
        line_counter+=1
        # Only use the last cutoff events for the trace, if cutoff > -1
        if cutoff > -1:
            next_trace = ''.join(trace.decode().split()[-cutoff:])
        else:
            next_trace = ''.join(trace.decode())
        # Delete whitespace character at the end of next_trace if necessary
        alphabet = {'0','1','2','3','4','5','6','7','8','9','.',';'}
        while next_trace[len(next_trace)-1] not in alphabet:
            next_trace = next_trace[:-1]
        # Ignore trace if its length exceeds the given threshold
        if trace_len>-1 and next_trace.count(" ")+1 > trace_len:
            continue
        # Update min_trace and max_trace
        if next_trace.count(" ") < sample_min_trace.count(" ") or len(sample_min_trace) == 0:
            sample_min_trace = next_trace
        if next_trace.count(" ") > sample_max_trace.count(" ") or len(sample_max_trace) == 0:
            sample_max_trace = next_trace
        # Only append the trace if it contains all wanted types
        append = True
        if types is not None:
            assert isinstance(types,list)
            for wanted_type in types:
                if wanted_type not in next_trace:
                    append = False
                    break
        if append is True:
            sample.append(next_trace)
    file.close()
    end = time.time() - start
    LOGGER.info('_dataset_to_sample - Finished reading file %s: %i lines in %f seconds', path_to_file, line_counter, end)

    # Build the filename
    LOGGER.info('_dataset_to_sample - Build sample_filename')
    sample_size = len(sample)
    sample_min_trace_len = sample_min_trace.count(" ") + 1
    sample_max_trace_len = sample_max_trace.count(" ") + 1
    sample_type_len = -1
    sample_filename = str(sample_size) + "." + str(sample_min_trace_len) + "." + str(sample_max_trace_len) + "." + str(sample_type_len) + "."
    if multidim is True:
        sample_event_dimension = (sample[0].split(" "))[0].count(";")
        sample_filename = sample_filename + str(sample_event_dimension) + "."
    filename = path_to_file.split("/")[-1]
    sample_filename = sample_filename + filename.split(".")[0] + ".seq"

    # Write the sample to a file
    LOGGER.info('_dataset_to_sample - Write sample to /samples')
    _write_sample(sample, sample_filename)
    LOGGER.info('_dataset_to_sample - Finished')

def _generate_experiments(convert_datasets:bool=False, path_to_file:str="",params_sample_size:list|None=None, params_trace_length:list|None=None, params_type_length:int=-1, params_dimension:list|None=None) -> None:
    """
        Generates & writes samples with different size, trace- and typelength.

        Uses the SampleGenerator. Creates "done.txt" in the end. Creates a
        *.seq.mpack and a *.done.txt file for each experiment. Parameters can
        be extracted from the filename using FILENAME_REGEXP.

        Args:
            convert_datasets: Boolean which indicates whether datasets stored
                in 'datasets/' should be converted into samples. If set to True
                no synthetic samples are generated. Datasets should be stored
                as '*txt.gz'.

            path_to_file: String containing the path to a specific dataset
                which should be converted to a sample. Dataset should be stored
                as '*txt.gz'.

            params_sample_size: List of integer which represents the number of
                traces which has to be generated per sample.

            params_trace_length: List of integer tuples. The first element of
                each tuple defines the minimum trace length, the second defines
                the maximum trace length of each trace of the current sample.

            params_type_length: Integer which can be used to define the length
                of each type. Default -1 means that the length will be
                calculated depending on the maximal trace length and the size
                of the sample.

            params_dimension: List of integers representing the event dimension
                for each (multidimensional) sample. Default None implies that
                only onedimensional samples will be generated.
    """
    LOGGER.debug('_generate_experiments - Starting')

    if not os.path.exists('samples'):
        os.mkdir('samples')

    if os.path.exists('datasets') and convert_datasets is True:
        if len(path_to_file)>0 and os.path.exists(path_to_file):
            _dataset_to_sample(path_to_file)
        else:
            entries = os.listdir("datasets/")
            for entry in entries:
                if entry.split(".")[-1]=='gz':
                    entry_filename = "datasets/"+str(entry)
                    _dataset_to_sample(entry_filename)
        return

    multidim = True
    if params_sample_size is None:
        params_sample_size = [50,100,150,200,250,300,350,400,450,500]
    if params_trace_length is None:
        params_trace_length = [(10,50),(50,100),(100,150),(150,200)]
    if params_dimension is None:
        params_dimension = [1]
        multidim = False

    log_params_sample_size = '_generate_experiments - params sample sizes: '+" ".join(str(params_sample_size))
    log_params_trace_length = '_generate_experiments - params trace length: '+" ".join(str(params_trace_length))
    LOGGER.debug(log_params_sample_size)
    LOGGER.debug(log_params_trace_length)

    for sample_size, trace_length, event_dimension in [  (sample_size, trace_length, event_dimension)
                                        for sample_size in params_sample_size
                                        for trace_length in params_trace_length
                                        for event_dimension in params_dimension]:
        filename_base_random = f"{sample_size}.{str(trace_length[0])}.{str(trace_length[1])}.{str(params_type_length)}.{str(event_dimension)}.{'random'}"
        filename_base_fragmentation_gauss = f"{sample_size}.{str(trace_length[0])}.{str(trace_length[1])}.{str(params_type_length)}.{str(event_dimension)}.{'fragmentation-gauss'}"
        filename_base_fragmentation_quartered = f"{sample_size}.{str(trace_length[0])}.{str(trace_length[1])}.{str(params_type_length)}.{str(event_dimension)}.{'fragmentation-quartered'}"

        filename_random = filename_base_random + ".seq"
        filename_fragmentation_gauss = filename_base_fragmentation_gauss + ".seq"
        filename_fragmentation_quartered = filename_base_fragmentation_quartered + ".seq"

        filename_random_done = filename_base_random + ".done.txt"
        filename_fragmentation_gauss_done = filename_base_fragmentation_gauss + ".done.txt"
        filename_fragmentation_quartered_done = filename_base_fragmentation_quartered + ".done.txt"

        if os.path.isfile(filename_base_random + ".done.txt") and os.path.isfile(filename_base_fragmentation_gauss + ".done.txt"):
            continue

        if multidim is True:
            generator = MultidimSampleGenerator()
            sample_random = generator.generate_random_sample(sample_size=sample_size, min_trace_length=trace_length[0], max_trace_length=trace_length[1], type_length=params_type_length, event_dimension=event_dimension)
            sample_fragmentation_gauss = generator.generate_fragmentation_gauss_sample(sample_size=sample_size, min_trace_length=trace_length[0], max_trace_length=trace_length[1], type_length=params_type_length, event_dimension=event_dimension)
            sample_fragmentation_quartered = generator.generate_fragmentation_quartered_sample(sample_size=sample_size, min_trace_length=trace_length[0], max_trace_length=trace_length[1], type_length=params_type_length, event_dimension=event_dimension)
        else:
            generator = SampleGenerator()
            sample_random = generator.generate_random_sample(sample_size=sample_size, min_trace_length=trace_length[0], max_trace_length=trace_length[1], type_length=params_type_length)
            sample_fragmentation_gauss = generator.generate_fragmentation_gauss_sample(sample_size=sample_size, min_trace_length=trace_length[0], max_trace_length=trace_length[1], type_length=params_type_length)
            sample_fragmentation_quartered = generator.generate_fragmentation_quartered_sample(sample_size=sample_size, min_trace_length=trace_length[0], max_trace_length=trace_length[1], type_length=params_type_length)

        # check if filename is parsable
        parsed_params = re.match(FILENAME_REGEXP,filename_random)
        assert parsed_params is not None
        assert int(parsed_params.group('sample_size'))==sample_size
        assert int(parsed_params.group('min_trace_length'))==trace_length[0]
        assert int(parsed_params.group('max_trace_length'))==trace_length[1]
        assert int(parsed_params.group('type_length'))==params_type_length
        assert int(parsed_params.group('event_dimension'))==event_dimension
        parsed_params = re.match(FILENAME_REGEXP,filename_fragmentation_gauss)
        assert parsed_params is not None
        assert int(parsed_params.group('sample_size'))==sample_size
        assert int(parsed_params.group('min_trace_length'))==trace_length[0]
        assert int(parsed_params.group('max_trace_length'))==trace_length[1]
        assert int(parsed_params.group('type_length'))==params_type_length
        assert int(parsed_params.group('event_dimension'))==event_dimension
        parsed_params = re.match(FILENAME_REGEXP,filename_fragmentation_quartered)
        assert parsed_params is not None
        assert int(parsed_params.group('sample_size'))==sample_size
        assert int(parsed_params.group('min_trace_length'))==trace_length[0]
        assert int(parsed_params.group('max_trace_length'))==trace_length[1]
        assert int(parsed_params.group('type_length'))==params_type_length
        assert int(parsed_params.group('event_dimension'))==event_dimension

        _write_sample(sample_random._sample, filename_random)
        _write_sample(sample_fragmentation_gauss._sample, filename_fragmentation_gauss)
        _write_sample(sample_fragmentation_quartered._sample, filename_fragmentation_quartered)
        open('samples/' + filename_random_done, 'a', encoding='utf-8').close()
        open('samples/' + filename_fragmentation_gauss_done, 'a', encoding='utf-8').close()
        open('samples/' + filename_fragmentation_quartered_done, 'a', encoding='utf-8').close()

    done_file = open('samples/done.txt', 'w', encoding='utf-8')
    done_file.write("Generate experiment done!")
    done_file.close()
    LOGGER.debug('_generate_experiments - Finished')

def non_matching_sample(sample_size:int=2, trace_length:int = 20, domain_size:int = 3) ->list:
    """Returns a sample list with non-matching events.

    Args:
        sample_size (int, optional): Number of traces. Defaults to 2.
        trace_length (int, optional): Number of events per traces. Defaults to 20.
        domain_size (int, optional): Number of domains. Defaults to 3.

    Returns:
        list: list of traces
    """
    generator = SampleGenerator()
    generator.generate_sample_w_empty_queryset(sample_size=sample_size, min_trace_length=trace_length, max_trace_length=trace_length)
    sample_list= []
    for trace in generator._sample._sample:
        new_trace = []
        for event in trace.split():
            new_event= f"{event};" *domain_size
            new_trace.append(new_event)
        sample_list.append(' '.join(new_trace))
    return sample_list

def match_algos(sample_list: list, results: list, iterations: int, j: int|str, mod: str,file_path: str, 
                discovery: list|None = None, max_query_length: int = - 1, only_types: bool = False) -> list:
    if not discovery:
        discovery = ['uni', 'sep', 'ups', 'sps']
    
    sample = MultidimSample()
    sample.set_sample(sample_list)
    sample_size = len(sample_list)

    trace_sample = MultidimSample()
    trace_sample.set_sample([sample_list[0]])

    trace_length = sample_list[0].count(' ') + 1
    sample.get_vertical_sequence_database()
    # stats = sample.sample_stats()
    len_pattern = []
    patterns = []
    trace_pattern = 0
    pattern_cnt = 0
    avg_query_length = 0
    result_dict = {}
    supp =1.0
    timeout = 28800
    att_vsdb = sample.get_att_vertical_sequence_database()
    vsdb = {}
    domain_cnt = sample._sample_event_dimension
    alphabet = set()
    gen_event= ';' * domain_cnt
    gen_event_list = [i for i in gen_event]
    patternset ={}
    for domain, dom_vsdb in att_vsdb.items():
        patternset[domain] = set()
        for key, value in dom_vsdb.items():
            new_key = ''.join(gen_event_list[:domain] + [key] + gen_event_list[domain:])
            vsdb[new_key] = value
            if not only_types:
                for item in value.keys():
                    if len(value[item]) >= 2:
                        patternset[domain].add(key)
                        break

    pattern_list = [0]*sample_size
    for pos_dict in vsdb.values():
        for trace, positions in pos_dict.items():
            if len(positions) >= 2:
                pattern_list[trace] = pattern_list[trace] + 1

    sample_sized_support = sample_size
    alphabet = {symbol for symbol,value in vsdb.items() if len(value) >= sample_sized_support}
    # vsdb = sample._sample_att_vertical_sequence_database
    dim_sample_dict = sample.get_dim_sample_dict()
    dim_stats = {dim: dim_sample.sample_stats() for dim, dim_sample in dim_sample_dict.items()}
    dim_trace_sample_dict = trace_sample.get_dim_sample_dict()
    dim_trace_stats = {dim: dim_sample.sample_stats() for dim, dim_sample in dim_trace_sample_dict.items()}
    sum_pattern_list = []
    sum_type_list = []
    for idx, _ in enumerate(sample._sample):
        sum_pattern_dom_list = []
        sum_type_dom_list = []
        for dom_sample in dim_sample_dict.values():
            supported_alphabet = dom_sample.get_sample_supported_typeset()
            trace_list = dom_sample._sample[idx].split()
            trace_length = len(trace_list)
            event_counter = collections.Counter(trace_list)
            freq_dom_list = event_counter.most_common()
            pattern_sum  = sum(tup[1] for tup in freq_dom_list if tup[1] > 1)
            sum_pattern_dom_list.append(pattern_sum)
            type_sum = sum(tup[1] for tup in freq_dom_list if tup[0].replace(';','') in supported_alphabet)
            sum_type_dom_list.append(type_sum)
        domain_pattern_sums = sum(sum_pattern_dom_list)
        sum_pattern_list.append(domain_pattern_sums)
        domain_type_sums = sum(sum_type_dom_list)
        sum_type_list.append(domain_type_sums)

        for domain in att_vsdb.keys():
            for _, value in att_vsdb[domain].items():
                if list(value.keys())[0] == idx:
                    trace_pattern = max(trace_pattern, len(value[idx]))
                    if len(value[idx]) >= 2:
                        pattern_cnt +=1
        len_pattern.append(trace_pattern)
        patterns.append(pattern_cnt)
        trace_pattern = 0
        pattern_cnt = 0
        # max_type_occurence =[stats['sample type distribution ordered'][-1][1] for stats in dim_stats.values()]
        trace_max_type_occurence =[stats['sample type distribution ordered'][-1][1] for stats in dim_trace_stats.values()]
    
    max_sum_pattern = min(sum_pattern_list, default=0)
    max_sum_type = max(sum_type_list, default=0)
    min_pattern_len = min(len_pattern)

    for matching in discovery:
        # LOGGER.info('Started %s', matching)
        start= time.time()
        if matching == 'uni':
            copy_sample = deepcopy(sample)
            args = (copy_sample, 1.0, 'smarter', False, max_query_length, only_types)
            result_dict = runFunction(bu_discovery_multidim, timeout, args, {})
            # result_dict = bu_discovery_multidim(copy_sample, 1.0, 'smarter', domain_seperated=False, 
            #                                     max_query_length=max_query_length, only_types=only_types)
            result= time.time()-start
            if result_dict:
                queryset1 = result_dict['queryset']
                queryset = queryset1
                # for i, querystring in enumerate(queryset):
                #     print(querystring)
                querycount = result_dict['querycount']
                searchspace = len(result_dict['matchingset'])
                query_lengths = 0
                for query in queryset:
                    query_lengths += len(query.split())
                if len(queryset) > 0:
                    avg_query_length = query_lengths / len(queryset)
        elif matching == 'sep':
            copy_sample = deepcopy(sample)
            args = (copy_sample, 1.0, 'smarter', True, max_query_length)
            result_dict = runFunction(bu_discovery_multidim, timeout, args, {})
            # result_dict = bu_discovery_multidim(copy_sample, 1.0, 'smarter', domain_seperated=True,
            #                                     max_query_length=max_query_length)
            if result_dict:
                queryset2 = result_dict['queryset']
                queryset = queryset2
                result= time.time()-start
                searchspace = len(result_dict['matchingset'])
                # domain_string = ' '.join(str(e) for e in result_dict['domain_queries'])
        elif matching == 'sps':
            copy_sample = deepcopy(sample)
            args = (copy_sample, 1.0, 'pattern-split-sep', True, max_query_length)
            result_dict = runFunction(bu_discovery_multidim, timeout, args, {})
            # result_dict = bu_discovery_multidim(copy_sample, 1.0, 'pattern-split-sep', domain_seperated=True,
            #                                     max_query_length=max_query_length)
            if result_dict:
                result= time.time()-start
                queryset3 = result_dict['queryset']
                queryset = queryset3
                searchspace = len(result_dict['matchingset'])
                #max_query= sorted(queryset, key=len)[-1].count(' ') +1
        elif matching == 'ups':
            copy_sample = deepcopy(sample)
            args = (copy_sample, 1.0, True, True, 'type_first', max_query_length)
            result_dicts = runFunction(discovery_bu_pts_multidim, timeout, args, {})
            # result_dict = discovery_bu_pts_multidim(copy_sample, 1.0, use_smart_matching=True, discovery_order='type_first',
            #                                         use_tree_structure=True, max_query_length=max_query_length)[2]
            if result_dict:
                result_dict = result_dicts[2]
                result = time.time()-start
                queryset5 = result_dict['queryset'] - {''}
                searchspace = result_dict['querycount']
                queryset = queryset5
            # assert queryset5 == queryset1
        elif matching == 'ilm':
            copy_sample = deepcopy(sample)
            args = (copy_sample,True, max_query_length)
            result_dict = runFunction(il_miner, timeout, args, {})
            # result_dict = il_miner(copy_sample, complete =True, max_query_length=max_query_length)
            if result_dict:
                queryset4 = result_dict['queryset']
                queryset = queryset4
                result= time.time()-start
                searchspace = len(result_dict['matchingset'])
            # domain_string = ' '.join(str(e) for e in result_dict['domain_queries'])
        elif matching == 'ilm_lossy':
            copy_sample = deepcopy(sample)
            args = (copy_sample,False, max_query_length)
            result_dict = runFunction(il_miner, timeout, args, {})
            # result_dict = il_miner(copy_sample, complete =False, max_query_length=max_query_length)
            if result_dict:
                queryset5 = result_dict['queryset']
                queryset = queryset5
                result= time.time()-start
                searchspace = len(result_dict['matchingset'])
            # domain_string = ' '.join(str(e) for e in result_dict['domain_queries'])
        if result_dict:
            results.append([str(sample_size), str(trace_length),str(iterations), matching,str(result),
                        str(j), str(len(queryset)), queryset, mod, searchspace, str(max_sum_type),
                        str(trace_max_type_occurence), str(max_sum_pattern),
                        str(max_query_length), str(len(alphabet)), str(mean(pattern_list)), str(avg_query_length)])
            # generate_plots(results=results, file_name='in_progress', y="time", x="iterations", hue="algorithm",kind="scatter")
        columns = ['sample size', 'trace length','iteration', 'algorithm', 'time',
                   'iterations', 'queryset size', 'queryset', 'mode', 'searchspace', 'type sum',
                   'trace max type occurrence', 'pattern sum', 'max query length',
                   'supported types', 'pattern types', 'avg query length']
        dataframe = pd.DataFrame(results, columns=columns)
        dataframe.to_csv(file_path)
        # LOGGER.info('Finished %s', matching)
    return results, columns
def change_sample(sample_list:list, pattern_type:int, rand_domain:list, pos:int, pos_list:list) ->list:
    """Change a given sample list to a new sample list adding new patterns or supported types.

    Args:
        sample_list (list): list of traces
        pattern_type (int): 0 adds a pattern, 1 adds a type
        rand_domain (list): list of domains that will be changed
        pos (int): position of the event that will be copied
        pos_list (list): list of positions where the event will be copied to

    Returns:
        list: list of traces
    """
    if pattern_type == -1:
        new_trace = sample_list[0]
        for pos2 in pos_list:
            new_trace = ' '.join(new_trace.split()[:pos2] + [new_trace.split()[pos]] + new_trace.split()[pos2+1:])
            sample_list[0] = new_trace
        return sample_list
    domain_cnt = len(sample_list[0].split()[pos].split(';'))
    for idx, trace in enumerate(sample_list):
        
        if pattern_type == 0:
            new_trace = trace
            for pos2 in pos_list:
                if len(rand_domain) < domain_cnt:
                    for i in rand_domain:
                        new_event = new_trace.split()[pos2].split(';')
                        new_event[i] = new_trace.split()[pos].split(';')[i]
                        new_trace = ' '.join(new_trace.split()[:pos2] + [';'.join(new_event)] + new_trace.split()[pos2+1:])
                else:
                    new_trace = ' '.join(new_trace.split()[:pos2] + [new_trace.split()[pos]] + new_trace.split()[pos2+1:])
                sample_list[idx] = new_trace
        else:
            if idx > 0:
                trace0 = sample_list[0]
                trace0_list = trace0.split()
                new_trace = trace
                pos2 = pos_list[0]
                for dom in rand_domain:
                    letter = trace0_list[pos].split(';')[dom]
                    pos_event = new_trace.split()[pos2].split(';')
                    pos_event[dom] = letter
                    new_trace = ' '.join(new_trace.split()[:pos2] + [';'.join(pos_event)] + new_trace.split()[pos2+1:])
                    sample_list[idx] = new_trace
    return sample_list



def generate_plots(dataframe:pd.DataFrame, file_name:str, x:str, y:str, hue:str, 
                   col: int|None = None,row = None, col_wrap: int|None = None, kind='bar',
                   overwrite:bool = False, facet_kws:dict|None = None) -> pd.DataFrame:
    sns.set_context("paper", font_scale=4)
    plt.style.reload_library()
    plt.style.use(['fivethirtyeight'])
    plt.rcParams['lines.linewidth'] = 4
    # plt.rcParams['font.size'] = 20
    plt.rcParams["savefig.facecolor"] = 'white'
    plt.rcParams["axes.facecolor"] = 'white'
    plt.rcParams["figure.facecolor"]= "white"
    
    if kind not in ['sota', 'cluster', 'sota_acc', 'exclude', 'rl_compare']:
        dataframe = dataframe.astype({'time':'float', 'iterations':'int'})
    elif kind in ['interleaving']:
        dataframe = dataframe.astype({'time':'float', 'iterations':'float'})
    else:
        dataframe = dataframe.astype({'time':'float'})
    result_path = 'experiment_results'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    plot_path = 'experiment_results'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    dataframe.replace({'uni': 'D-U-C', 'sep': 'D-U-S', 'ilm': 'ILM',
                       'ups': 'B-S-C', 'sps': 'B-S-S', 'rl':'RL'}, inplace=True)
    hatches = {'D-U-C':'',
                   'D-U-S': '',
                   'B-S-C': '',
                   'B-S-S': '',
                   'ilm_lossy': '',
                   'ILM': '',
                   'RL': ''}
#                   'D-U-S': '/',
#                   'B-S-C': '|',
#                   'B-S-S': '//',
#                   'ilm_lossy': 'x',
#                   'ILM': 'o',
#                   'RL': 'x',}
    if kind in {'synt_queries'}:
        
        grid = sns.catplot(data=dataframe, y=y, x=x, hue=hue , kind='box', col=col, col_wrap=col_wrap)
        grid.set(yscale='log')
        grid.savefig(f'experiment_results/{file_name}.pdf')
    elif kind in ['sota']:
        dataframe.rename(columns={'time':'time[s]'}, inplace=True)
        y='time[s]'
        plt.rcParams['font.size'] = 20
        fig, grid = plt.subplots(1,2, layout='constrained', figsize=(15, 3))
        fig2 = plt.figure(figsize=(25, 6))
        gs = GridSpec(nrows=2, ncols=2)
        # First axes
        ax0 = fig2.add_subplot(gs[1, 1])
        # Second axes
        ax1 = fig2.add_subplot(gs[0, 0])
        # Third axes
        ax2 = fig2.add_subplot(gs[1, 0])
        # Fourth axes
        ax3 = fig2.add_subplot(gs[0, 1])
    
        grid[0].set(yscale='log')
        ax0.set(yscale='log', yticks=[.1, 10])
        ax1.set(yscale='log', yticks=[1000,10000, 100000])
        ax2.set(yscale='log', yticks=[.1, 10])
        ax3.set(yscale='log', yticks=[1000, 10000 ,100000])
        ax2.set_ylim(0.0001, 100)
        ax0.set_ylim(0.0001, 100)
        ax1.set_ylim(bottom=1000)
        ax3.set_ylim(bottom=1000)
        ax0.sharey(ax2)
        ax1.sharex(ax2)
        ax3.sharey(ax1)
        ax1.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)

        d = .01  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        adatped_dataframe = dataframe.loc[dataframe['algorithm'] != 'ilm_lossy']
        cart_product = list(product(adatped_dataframe['algorithm'].unique(), adatped_dataframe['iterations'].unique()))
        querystring_dict = {}

        for idx, mod in enumerate(['google', 'finance']):
            data_subplot = adatped_dataframe.loc[adatped_dataframe['mode'] == mod]
            sns.barplot(data=data_subplot, y=y, x=x, hue=hue, ax=grid[idx])
            if idx == 0:
                sns.barplot(data=data_subplot, y=y, x=x, hue=hue, ax=ax0)
                # ax0.bar_label(ax0.containers[4], fmt='timeout', label_type='edge')
                if 'ILM' in adatped_dataframe['algorithm'].unique():
                    # for container in ax0.containers[4]:
                    #     print(container._x0)
                    #     if isnan(container._height):
                    #         position = container._x0 #+ container._width/2
                    #         ax0.text(position, 0.0001, 'timeout',rotation=90, fontsize=35)
                    positions = [1.24,3.24,5.24]
                    for position in positions:
                        ax0.text(position, 0.0001, 'timeout',rotation=90, fontsize=35)
                
    
            else:
                sns.barplot(data=data_subplot, y=y, x=x, hue=hue, ax=ax1)
                sns.barplot(data=data_subplot, y=y, x=x, hue=hue, ax=ax2)
                # ax2.bar_label(ax2.containers[4], fmt='timeout', label_type='edge')
                if 'ILM' in adatped_dataframe['algorithm'].unique():
                    # for container in ax2.containers[4]:
                    #     print(container._x0)
                    #     if isnan(container._height):
                    #         position = container._x0 #+ container._width/4
                    #         ax2.text(position, 0.0001, 'timeout',rotation=90, fontsize=35)
                    positions = [1.24,3.24,5.24]
                    for position in positions:
                        ax2.text(position, 0.0001, 'timeout',rotation=90, fontsize=35)
                data_subplot = adatped_dataframe.loc[adatped_dataframe['mode'] == 'google']
                sns.barplot(data=data_subplot, y=y, x=x, hue=hue, ax=ax3)
            ax3_count = 0
            for count, (i, p) in enumerate(zip(cart_product, grid[idx].patches)):
                
                if count <=5:
                    hatch = 'D-U-C'
                elif count <=11:
                    hatch = 'D-U-S'
                elif count <=17:
                    hatch = 'B-S-C'
                elif count <=23:
                    hatch = 'B-S-S'
                else:
                    hatch = 'ILM'

                p.set(hatch=hatches[hatch])
                if idx == 0:
                    
                    grid[idx].legend([], [], frameon=False)
                    grid[idx].set(xlabel='Google')
                    ax0.set(xlabel='Google')

                    ax0.patches[count].set(hatch=hatches[hatch])

                    ax0.patches[27].set(hatch=hatches['D-U-C'])
                    ax0.patches[28].set(hatch=hatches['D-U-S'])
                    ax0.patches[29].set(hatch=hatches['B-S-C'])
                    ax0.patches[30].set(hatch=hatches['B-S-S'])
                    ax0.patches[31].set(hatch=hatches['ILM'])
                    # plt.bar_label(splot.containers[0], label_type='center')
                    # fig.legend(loc='outside right')
                    ax0.legend([], [], frameon=False)

                    # fig2.legend(loc='upper right',)
                    fig2.legend(bbox_to_anchor=(0.1, .9, .8, .07),loc='upper left',
                                mode='expand', ncols=5, borderaxespad=0.)

                else:
                    grid[idx].set(ylabel=None, xlabel='Finance')
                    grid[idx].legend([], [], frameon=False)
                    ax2.patches[count].set(hatch=hatches[hatch])
                    ax1.patches[count].set(hatch=hatches[hatch])

                    ax2.set(xlabel='Finance', ylabel='')
                    ax1.set(xlabel='', ylabel='')
                    ax2.legend([], [], frameon=False)
                    ax1.legend([], [], frameon=False)

                    # if i[1] in data_subplot.loc[data_subplot.algorithm== 'ILM'].iterations.unique():
                    ax3.legend([], [], frameon=False)
                    #     ax3.patches[count-ax3_count].set(hatch=hatches[i[0]])
                    # else:
                    #     ax3_count +=1
                    # sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))

                    # fig.legend()
        fig2.supylabel('time[s]', fontsize=35)
        fig2.subplots_adjust(wspace=0, hspace=0.1)
        # fig.savefig(f'experiments/results/plots/{file_name}.pdf')
        fig2.savefig(f'../experiments/experiment_results/{file_name}_broken_new.pdf')

    elif kind in ['sota_acc']:
        sns.set_context("paper", font_scale=3)
        plt.style.reload_library()
        # plt.style.use(['science'])
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('legend',fontsize='medium')

        grid = sns.relplot(data=dataframe, y=y, x=x, hue=hue,
                           kind='scatter', s=5)
        grid.set(xscale='log')

        grid.set_xlim(0.09, 11000)
        grid.savefig(f'experiment_results/{file_name}.pdf')

    elif kind in ['cluster', 'exclude']:
        sns.set_context("paper", font_scale=3)
        plt.rcParams['lines.linewidth'] = 4

        plt.rcParams['font.size'] = 40
        if kind == 'cluster':
            dataframe.replace({'google': 'G1', 'finance': 'F1'}, inplace=True)
        else:
            dataframe.replace({'google': 'G3', 'finance': 'F3'}, inplace=True)
        grid = sns.relplot(data=dataframe, y=y, x=x, hue=hue, style=hue, err_style='bars',
                           kind='line', col=col, col_wrap=1, facet_kws=facet_kws, legend=False)
        
        grid.set(xlabel='')  # remove the axis label
        grid.set_titles("{col_name}", fontsize=30)

        if kind == 'cluster':
            grid.set(yscale='log', yticks=[1, 10])

        # sns.move_legend(grid, "upper left", bbox_to_anchor=(.8, .99))
        plt.subplots_adjust(wspace = 0.1)

        grid.savefig(f'experiment_results/{file_name}.pdf')

    elif kind in ['synt']:
        dataframe.rename(columns={'time':'time[s]'}, inplace=True)
        dataframe.replace({'streams': r'$E_1:|D|$', 'stream length': r'$E_2:|S|$', 'domain size': r'$E_3: |\mathcal{A}|$',
                       'types': r'$E_4: |\Gamma_D|$', r'max type': r'$E_5: \rho_S$', 'pattern sum':r'$E_6: \rho_R$'}, inplace=True)
        sns.set_context("paper", font_scale=3)
        plt.rcParams['lines.linewidth'] = 4
        fig, grid = plt.subplots()

        grid = sns.relplot(data=dataframe, y=y, x=x, hue=hue, style=hue, err_style='bars',
                           kind='line', col=col, col_wrap=col_wrap, facet_kws=facet_kws)
        grid.set(yscale='log', yticks=[1, 10])
        grid.set_titles("{col_name}", y=-0.25, fontsize=30)
        grid.set_xlabels('')
        plt.rcParams['font.size'] = 40
        sns.move_legend(grid, loc='upper center', mode='expand',
                        ncols=4, bbox_to_anchor=(0.1,1.0, 0.5, 0),
                        borderaxespad=0., title=None, frameon=False)


        grid.savefig(f'experiment_results/{file_name}.pdf')

    elif kind in ['number_of_types', 'max_pattern', 'trace_length', 'number_of_patterns',
                'supp_alphabet', 'synt', 'domain_size', 'interleaving', 'max_type']:

        grid = sns.relplot(data=dataframe, y=y, x=x, hue=hue, style=hue, err_style='bars',
                           kind='line', col=col, col_wrap=col_wrap, facet_kws=facet_kws)


        grid.set(yscale='log', yticks=[1, 100])
        

        grid.savefig(f'experiment_results/{file_name}.pdf')

    
    elif kind == 'rl_compare':
        dataframe.rename(columns={'time':'time[s]'}, inplace=True)
        dataframe.replace({'google': 'G2', 'finance': 'F2'}, inplace=True)
        y='time[s]'
        fig, grid = plt.subplots(figsize=(8,7)) # figsize=(20, 15)
        # grid.set_xlabel(' ', fontdict={'fontsize': 40})


        grid.set(yscale='log')

        iteration = len(dataframe.iteration.unique())
        sns.barplot(data=dataframe, y=y, x=x, hue=hue, ax=grid, width=.8, order=['F2', 'G2'])
        disc_list = [disc for disc in dataframe['algorithm'].unique() for _ in range(iteration)]
        
        for disc,patch in zip(disc_list,grid.patches):
            patch.set_hatch(hatches[disc])
        
        handles, _ = grid.get_legend_handles_labels()
        handles[0].set_hatch(hatches['D-U-C'])
        handles[1].set_hatch(hatches['RL'])

        grid.legend([], [], frameon=False)
        grid.set_ylabel('time[s]' ,fontsize = 40)
        grid.tick_params(axis='x', labelsize=40)
        grid.set(xlabel=None)
        grid.set_yticks([1, 1000])
        grid.tick_params(axis='y', which='minor', direction='in', length=5, width=2)
        fig.legend(handles=handles, bbox_to_anchor=(0.05, .95, .85, .07),loc='upper left',
                               mode='expand', ncols=2, borderaxespad=0., frameon=False)
        plt.tight_layout()
        plt.savefig(f'experiment_results/{file_name}.pdf')

    return dataframe

def runFunction(f, max_wait, args, default_value):
    try:
        return func_timeout.func_timeout(timeout=max_wait, func=f, args=args)
    except func_timeout.FunctionTimedOut:
        pass
    return default_value