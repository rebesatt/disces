import os
import sys
from copy import deepcopy
import random
import collections
import logging

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

#from generator import SampleGenerator
#sys.path.append("./src")
CURRENT_WD = os.getcwd()

EXPERIMENT_DIR = os.path.abspath(__file__).replace("synt_sample_generator.py", "")
os.chdir(EXPERIMENT_DIR)
os.chdir("../../src")
SRC_DIR = os.getcwd()

os.chdir(EXPERIMENT_DIR)
sys.path.insert(0, SRC_DIR)

from testbench_helper_functions import _write_sample, non_matching_sample, change_sample
from sample_multidim import MultidimSample

def trace_length_sample(iterations: int):
    dir_name = f'trace_length'
    
    sample_list = non_matching_sample(sample_size=2, trace_length=iterations)
    new_event = sample_list[0].split()[0]
    new_trace_list = ' '.join(sample_list[0].split()[:1] + [new_event] + sample_list[0].split()[2:])
    sample_list[0] = new_trace_list
    
    new_sample_list = deepcopy(sample_list)
    for j in range(5, iterations+1, 10):
        #Modify Sample
        new_event = sample_list[1].split()[0]
        new_sample_list[1] = ' '.join(sample_list[1].split()[:j] + [new_event])

        #Save Sample
        file_name = f'sample_{iterations}_{j}'
        dirname = f'{dir_name}'
        _write_sample(filename=file_name, sample=new_sample_list, dirname=dirname)

def domain_size_sample(iterations: int):
    dir_name = f'domain_size'

    for j in range(iterations):
        sample_list = non_matching_sample(trace_length=20,domain_size=j+1)
        new_sample_list = deepcopy(sample_list)
        new_sample_list = change_sample(new_sample_list, pattern_type=1, pos=0, pos_list=[1], rand_domain=[0])
        new_sample_list = change_sample(new_sample_list, pattern_type=1, pos=0, pos_list=[2], rand_domain=[0])
        new_sample_list = change_sample(new_sample_list, pattern_type=1, pos=0, pos_list=[3], rand_domain=[0])
        new_sample_list = change_sample(new_sample_list, pattern_type=1, pos=0, pos_list=[4], rand_domain=[0])
        new_sample_list = change_sample(new_sample_list, pattern_type=0, pos=5, pos_list=[6,7], rand_domain=[0])
        new_sample_list = change_sample(new_sample_list, pattern_type=0, pos=8, pos_list=[9], rand_domain=[0])
        new_sample_list = change_sample(new_sample_list, pattern_type=1, pos=10, pos_list=[10], rand_domain=[0])
        
        #Save Sample
        if j%5 == 0:
            file_name = f'sample_{iterations}_{j}'
            dirname = f'{dir_name}'
            _write_sample(filename=file_name, sample=new_sample_list, dirname=dirname)

def types_sample(iterations: int):
    dir_name = f'types'
    
    trace_length = 102
    sample_list = non_matching_sample(sample_size=2, trace_length=trace_length, domain_size=1)

    new_sample_list = deepcopy(sample_list)
    for j in range(iterations):
        #Modify Sample
        if j%5 == 0:
            new_sample_list = change_sample(sample_list=new_sample_list, pattern_type=1, rand_domain=[0],
                                        pos=j, pos_list=[iterations-j-4])
        elif j%5 == 1:
            new_sample_list = change_sample(sample_list=new_sample_list, pattern_type=1, rand_domain=[0],
                                        pos=j, pos_list=[iterations-j-2])
        elif j%5 == 2:
            new_sample_list = change_sample(sample_list=new_sample_list, pattern_type=1, rand_domain=[0],
                                        pos=j, pos_list=[iterations-j])
        elif j%5 == 3:
            new_sample_list = change_sample(sample_list=new_sample_list, pattern_type=1, rand_domain=[0],
                                        pos=j, pos_list=[iterations-j+2])
        else:
            new_sample_list = change_sample(sample_list=new_sample_list, pattern_type=1, rand_domain=[0],
                                        pos=j, pos_list=[iterations-j+4])
        
        modulo = iterations//10
        if j%modulo == 0:
            #Save Sample
            file_name = f'sample_{iterations}_{j}'
            dirname = f'{dir_name}'
            _write_sample(filename=file_name, sample=new_sample_list, dirname=dirname)

def traces_sample(iterations: int):
    dir_name = f'streams'
    
    for j in range(2, iterations+1, iterations//10):
        #Modify sample
        sample_list = non_matching_sample(sample_size=j, trace_length=10)
        new_sample_list = change_sample(sample_list=sample_list, pattern_type=0, rand_domain=[0], pos=0, pos_list=[0])
        
        #Save Sample
        file_name = f'sample_{iterations}_{j}'
        dirname = f'{dir_name}'
        _write_sample(filename=file_name, sample=new_sample_list, dirname=dirname)

def max_pattern_sample(iterations: int):
    dir_name = f'max_pattern'

    sample_list = non_matching_sample()
    trace_length = len(sample_list[0].split())
    domain_cnt = sample_list[0].split()[0].count(';')
    new_sample_list = deepcopy(sample_list)
    pos_set = set(range(trace_length))
    for j in range(iterations):
        cur_domain = j% domain_cnt
        #Modify Sample
        if not pos_set:
            break

        pos_list = random.sample(sorted(pos_set), 1)
        if j == 0:
            pos = min(pos_set)
            pos_set.discard(pos)
        pos_set.difference_update(pos_list)

        new_sample_list = change_sample(new_sample_list, 0, [cur_domain], pos, pos_list)
        sample = MultidimSample()
        sample.set_sample(new_sample_list)
        sample.get_vertical_sequence_database()
        dim_sample_dict = sample.get_dim_sample_dict()
        sum_pattern_list = []
        for idx, _ in enumerate(sample._sample):
            sum_pattern_dom_list = []
            for dom_sample in dim_sample_dict.values():
                trace_list = dom_sample._sample[idx].split()
                trace_length = len(trace_list)
                event_counter = collections.Counter(trace_list)
                freq_dom_list = event_counter.most_common()
                pattern_sum  = sum(tup[1] for tup in freq_dom_list if tup[1] > 1)
                sum_pattern_dom_list.append(pattern_sum)
            domain_pattern_sums = sum(sum_pattern_dom_list)
            sum_pattern_list.append(domain_pattern_sums)

        max_sum_pattern = min(sum_pattern_list, default=0)
        #Save Sample
        file_name = f'sample_{iterations}_{max_sum_pattern}'
        dirname = f'{dir_name}'
        _write_sample(filename=file_name, sample=new_sample_list, dirname=dirname)

def max_type_sample(iterations: int):
    dir_name = f'max_type'
    sample_list = non_matching_sample(sample_size=2, trace_length=20, domain_size=3)

    new_sample_list = change_sample(sample_list=sample_list, pattern_type=0, rand_domain=[0,1,2],
                                            pos=0, pos_list=[0,1])
    new_sample_list = change_sample(sample_list=new_sample_list, pattern_type=1, rand_domain=[0,1,2],
                                            pos=0, pos_list=[0])
    new_sample_list = change_sample(sample_list=new_sample_list, pattern_type=1, rand_domain=[0,1,2],
                                            pos=0, pos_list=[1])
    first_event = new_sample_list[0].split()[0]

    # new_sample_list = deepcopy(sample_list)
    for j in range(iterations):
        #Modify Sample
        new_trace = new_sample_list[1] + ' ' + first_event
        new_sample_list[1] = new_trace

        #Save Sample
        if j%5 == 0:
            sample = MultidimSample()
            sample.set_sample(sample_list)
            sample.get_vertical_sequence_database()
            dim_sample_dict = sample.get_dim_sample_dict()
            sum_type_list = []
            for idx, _ in enumerate(sample._sample):
                sum_type_dom_list = []
                for dom_sample in dim_sample_dict.values():
                    supported_alphabet = dom_sample.get_sample_supported_typeset()
                    trace_list = dom_sample._sample[idx].split()
                    event_counter = collections.Counter(trace_list)
                    freq_dom_list = event_counter.most_common()
                    type_sum = sum(tup[1] for tup in freq_dom_list if tup[0].replace(';','') in supported_alphabet)
                    sum_type_dom_list.append(type_sum)
                domain_type_sums = sum(sum_type_dom_list)
                sum_type_list.append(domain_type_sums)

            max_sum_type = max(sum_type_list, default=0)
            file_name = f'sample_{iterations}_{max_sum_type}'
            dirname = f'{dir_name}'
            _write_sample(filename=file_name, sample=new_sample_list, dirname=dirname)

if __name__ == "__main__":
    LOGGER.info("Generating synthetic stream databases")
    trace_length_sample(iterations=100)
    LOGGER.info("1/6")
    domain_size_sample(iterations=101)
    LOGGER.info("2/6")
    traces_sample(iterations=100000)
    LOGGER.info("3/6")
    max_pattern_sample(iterations=10)
    LOGGER.info("4/6")
    max_type_sample(iterations=100)
    LOGGER.info("5/6")
    types_sample(iterations=100)
    LOGGER.info("6/6")