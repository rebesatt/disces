#!/usr/bin/python3
"""Contains functions implementing the Il-Miner."""
import logging
from collections import Counter
from itertools import product, chain, combinations

from discovery_bu_multidim import to_normalform, non_descriptive_queries_multidim

#Logger Configuration:
LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)


def il_miner(sample, complete, max_query_length=-1):
    """Adapted implementation of the Il-Miner.
    Lars George, Bruno Cadonna, and Matthias Weidlich. 2016.
    IL-miner: instance-level discovery of complex event patterns.
    Proc. VLDB Endow. 10, 1 (September 2016), 25–36.
    DOI:https://doi.org/10.14778/3015270.3015273


    Args:
        sample (MultidimSample): an instance of MultidimSample
        complete (boolean): Determines the variation of il-Miner. True resolves to all descriptive queries.
                            False to reduced set defined in il-miner paper. Default: True.
    """
    sample_set = sample._sample
    domain_cnt = sample._sample_event_dimension
    gen_event = ';' * domain_cnt
    eventset, event_temp_dict = eventset_vertdb(sample_set, complete)
    if max_query_length == -1:
        max_query_length = sample._sample[0].count(' ') + 1
    _,min_trace_length= sample.get_sample_min_trace()
    max_query_length = min(max_query_length, min_trace_length)

    # LOGGER.info('Phase 1: Sequence Template Learning.')
    #Phase 1: Sequence Template Learning.
    sequence_templates = learning_sequence_templates(sample_set, eventset, event_temp_dict, max_query_length=max_query_length)

    # LOGGER.info('Phase 2: Pattern Construction. - Linking Sequence Templates')
    #Phase 2: Pattern Construction.
    #Linking Sequence Templates
    if complete:
        poss_templates = [el for el in sequence_templates]
        while poss_templates:
            template = poss_templates.pop()
            template_list = template.split()
            old_event = ''
            for idx, event in enumerate(template_list):
                if event == gen_event and old_event == gen_event:
                    new_template = ' '.join(template_list[:idx] + template_list[idx+1:])
                    if new_template not in sequence_templates:
                        if max_query_length == -1 or len(new_template.split()) <= max_query_length:
                            sequence_templates.append(new_template)
                    if new_template not in poss_templates:
                        poss_templates.append(new_template)
                old_event = event
    trace_matches = linking_sequence_templates(sequence_templates, event_temp_dict)

    # LOGGER.info('Phase 2: Pattern Construction. - Extracting  and Merging Property and Relation Constraints')
    #Extracting  and Merging Property and Relation Constraints
    constraints = set(extract_constraints(trace_matches, sample_set))
    non_descriptives_dict={}
    queryset = {to_normalform(constraint) for constraint in constraints}
    # LOGGER.info('Discarding non-descriptive queries.')
    non_descriptive = set()
    for querystring in queryset:
        if querystring in non_descriptives_dict:
            quer_non_descr= non_descriptives_dict[querystring]
        else:
            quer_non_descr= non_descriptive_queries_multidim(querystring=querystring, parent_dict= queryset)
            non_descriptives_dict[querystring] = quer_non_descr
        non_descriptive.update(quer_non_descr)

    not_seen= non_descriptive - queryset

    while not_seen:
        querystring2 = not_seen.pop()
        if querystring2 in non_descriptives_dict:
            quer_non_descr= non_descriptives_dict[querystring2]
        else:
            quer_non_descr= non_descriptive_queries_multidim(querystring=querystring2, parent_dict=queryset)
            non_descriptives_dict[querystring2] = quer_non_descr
            non_descriptive.update(quer_non_descr)
            not_seen.update(quer_non_descr)
    return {
        'queryset': queryset - non_descriptive - {gen_event} - {''},
        'matchingset': queryset - {''},
    }



def learning_sequence_templates(sample_set, eventset, event_temp_dict, max_query_length):
    """Function that learns the sequence templates given a sample set and events.

    Args:
        sample_set (list): list of traces
        eventset (dictionary): {events : list of sub-eventsets}
        event_temp_dict (dictionary): {event_template with supp=1 : {trace_id : list of occurence positions}}

    Returns:
        list: list of strings of possible sequence templates
    """
    event_template_sets= set_of_event_templates(sample_set, eventset, event_temp_dict)
    #Maximum Frequent Sequence Mining
    sequence_templates_sets = pincer_search(event_template_sets, max_query_length=max_query_length)
    sequence_templates=[]
    for template_query in sequence_templates_sets:
        sequence_templates.extend(seq_templates(list(template_query)))
    return list(set(sequence_templates))

def linking_sequence_templates(sequence_templates, event_temp_dict):
    """Links the sequence templates to all the occurences in each trace

    Args:
        sequence_templates (list): list of tuples of possible sequence template sets
        event_temp_dict (dictionary): {event_template with supp=1 : {trace_id : list of occurence positions}}

    Returns:
        dictionary: {template string : {trace id : list of matching positions }}
    """
    trace_instances={}
    for template in sequence_templates:
        trace_instances[template]={}
        for idx, event_temp in enumerate(template.split()):
            for trace in event_temp_dict[(event_temp,)]:
                if idx == 0:
                    trace_instances[template][trace]= [[pos] for pos in event_temp_dict[(event_temp,)][trace]]
                else:
                    instances=[]
                    for instance in trace_instances[template][trace]:
                        last_position= instance[-1]
                        new_positions = event_temp_dict[(event_temp,)][trace]
                        for new_pos in new_positions[::-1]:
                            if new_pos > last_position:
                                instances.append(instance+ [new_pos])
                            else:
                                break
                    trace_instances[template][trace] = instances
    return trace_instances

def extract_constraints(trace_matches, sample_set):
    """Extracts the constraints from the trace matches and returns a maximal subset

    Args:
        trace_matches (dictionary): {template string : {trace id : list of matching positions }}
        sample_set (list): list of traces
    Returns:
        list: list of descriptive query strings
    """
    relation_constraints={}
    type_queries=[]
    mixed_queries=[]
    constraint_candidates= []
    for template in trace_matches:
        len_templ = len(template.split())
        for idx, trace in enumerate(trace_matches[template]):
            if idx == 0:
                constraint_candidates= []
                for match in trace_matches[template][trace]:
                    for idx2, pos in enumerate(match):
                        event = sample_set[trace-1].split()[pos]
                        for indx, pos2 in enumerate(match[idx2+1:], start=idx2+1):
                            template_pos= template.split()
                            event2 = sample_set[trace-1].split()[pos2]
                            if event.count(';') >1:
                                for domain_index, domain in enumerate(event.split(';')[:-1]):
                                    if not template_pos[idx2].split(';')[domain_index] and not template_pos[indx].split(';')[domain_index]:
                                        if event.split(';')[domain_index]== event2.split(';')[domain_index]:
                                            if [idx2, indx, domain_index] not in constraint_candidates:
                                                constraint_candidates.append([idx2, indx, domain_index])
                            else:
                                if event.split(';')[0]== sample_set[trace-1].split()[pos2].split(';')[0]:
                                    if [idx2, indx, 0] not in constraint_candidates:
                                        constraint_candidates.append([idx2, indx, 0])

            else:
                not_found = []
                for i, constraint in enumerate(constraint_candidates):
                    found = False
                    for match in trace_matches[template][trace]:
                        if not found:
                            pos1 = match[constraint[0]]
                            pos2 = match[constraint[1]]
                            event1 = sample_set[trace-1].split()[pos1]
                            event2 = sample_set[trace-1].split()[pos2]
                            if event1.split(';')[constraint[2]] == event2.split(';')[constraint[2]]:
                                found = True
                                break

                    if not found:
                        not_found.append(constraint)
                constraint_candidates = [constraint for constraint in constraint_candidates if constraint not in not_found]
                # if not_found:
                #     LOGGER.debug(not_found)
        if constraint_candidates:
            if len(constraint_candidates) > 1:
                possible_constr_subset = []
                possible_constr_subset.append(constraint_candidates)
                max_constr_subsets = []
                not_max_subsets= []
                possible_matches = {}
                while possible_constr_subset:
                    found = True
                    constr_candidate = possible_constr_subset.pop()
                    for idx, trace in enumerate(trace_matches[template]):
                        if constr_candidate in not_max_subsets:
                            continue
                        pos_trace_matches=[]
                        possible_matches[trace] = {}
                        for i, constraint in enumerate(constr_candidate):
                            domain_index = constraint[2]
                            if found:
                                if tuple(constraint) in possible_matches[trace]:
                                    pos_trace_matches.append(possible_matches[trace][tuple(constraint)])
                                else:
                                    possible_matches[trace][tuple(constraint)] = []
                                    for match in trace_matches[template][trace]:
                                        pos1 = match[constraint[0]]
                                        pos2 = match[constraint[1]]
                                        event1 = sample_set[trace-1].split()[pos1]
                                        event2 = sample_set[trace-1].split()[pos2]
                                        if event1.split(';')[domain_index] == event2.split(';')[domain_index]:
                                            pos_trace_matches.append(match)
                                            possible_matches[trace][tuple(constraint)].append(match)

                                if i != 0:
                                    copy_pos_trace_matches = [tuple(i) for i in pos_trace_matches]
                                    pos_trace_matches = [list(k) for k, v in Counter(copy_pos_trace_matches).items() if v > 1]

                                    if len(pos_trace_matches) == 0:
                                        subset_length= len(constr_candidate) -1
                                        for item in combinations(constr_candidate, subset_length):
                                            if item not in possible_constr_subset:
                                            #     constr_pos = set()
                                            #     for cond in item:
                                            #         constr_pos.add(cond[0])
                                            #         constr_pos.add(cond[1])
                                            #     if len(constr_pos) == len(set(range(len_templ))):
                                                possible_constr_subset.append(item)
                                        found = False
                                        # not_max_subsets.append(constr_candidate)
                                        break
                    if found and constr_candidate not in max_constr_subsets:
                        if constr_candidate not in not_max_subsets:
                            max_constr_subsets.append(constr_candidate)
                        if len(constr_candidate) > 1:
                            not_max_subsets.extend(list(chain.from_iterable(combinations(constr_candidate, r) for r in range(1,len(constr_candidate)))))
                relation_constraints[template] = [const_candidate for const_candidate in max_constr_subsets if const_candidate not in not_max_subsets]
            else:
                relation_constraints[template] = [constraint_candidates]

            for candidate in relation_constraints[template]:
                pattern_list = [domain.split(';') for domain in template.split()]
                var_cnt_max = 0
                for condition in candidate:
                    pos1 = condition[0]
                    pos2 = condition[1]
                    domain_index = condition[2]
                    if pattern_list[pos1][domain_index].count("$") !=0:
                        var_cnt = int(pattern_list[pos1][domain_index][-1])
                    elif pattern_list[pos2][domain_index].count("$") !=0:
                        var_cnt = int(pattern_list[pos2][domain_index][-1])
                    else:
                        var_cnt = var_cnt_max
                        var_cnt_max+=1
                    pattern_list[pos1][domain_index]= f'$x{var_cnt}'
                    pattern_list[pos2][domain_index]= f'$x{var_cnt}'

                pattern_string= " ".join([";".join(element) for element in pattern_list])
                if pattern_string not in mixed_queries:
                    mixed_queries.append(pattern_string)

        else:
            type_queries.append(template)
    return mixed_queries + type_queries

def powerset(event, complete):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        Given an event template string it calculates all substrings that contain the given number of
        semicolon (';').
        e.g  powerset('b;') --> [(';',) , ('b;',)]
    Args:
        event (String): String representation of an event
        complete (boolean): Determines the variation of il-Miner. True resolves to all descriptive queries.
                            False to reduced set defined in il-miner paper.
                            Default: True.
    Returns: List containing powerset of an event as tuples
    """
    event_list = []
    for element in event.split(';')[:-1]:
        event_list.append(element)
        event_list.append(';')
    subsets = list(set(chain.from_iterable(combinations(event_list, r) for r in range(1,len(event_list)+1))))
    templates= []
    if complete:
        for ev_temp in subsets:
            if ev_temp.count(';') == event.count(';'): #and len(ev_temp)> event.count(';'):
                templates.append(("".join(ev_temp),))
    else:
        for ev_temp in subsets:
            if ev_temp.count(';') == event.count(';') and len(ev_temp)> event.count(';'):
                templates.append(("".join(ev_temp),))
    return set(templates)
def set_of_event_templates(sample_set, eventset, event_temp_dict):
    """Translates traces into tuples of matching subevents for each event.

    Args:
        sample_set (list): list of traces
        eventset (dictionary): {occuring events : list of sub-eventsets}
        event_temp_dict (dictionary): {event_template with supp=1 : {trace_id : list of occurence positions}}

    Returns:
        (list): list of tuples of possible sequence template sets
    """
    event_template_sets = list()
    event_dict={}
    i=1
    for trace in sample_set:
        trace_list = []
        trace_spit = trace.split()
        for event in trace_spit:
            event_set = ()
            possible_event_templates= eventset[event]
            for possible_event in possible_event_templates:
                if possible_event in event_temp_dict:
                    event_set= event_set + (possible_event,)
            if event_set not in event_dict:
                event_dict[event_set]= i
                i+=1
            trace_list.append(event_set)
        event_template_sets.append(trace_list)
    return event_template_sets

def pincer_search(event_template_sets, max_query_length):
    """Adapted pincer search. Finds Maximal frequent sequence template sets.

    Args:
        event_template_sets (list): list of tuples of possible sequence template sets

    Returns:
        set: tuples of sequence template sets
    """
    min_trace = min(event_template_sets, key=len)
    sequence_templates= set()
    redundant= set()
    not_found = set()
    # if max_query_length == -1:
    # stack = [min_trace]
    # else:
    stack = list(set(combinations(min_trace, max_query_length)))
    # LOGGER.info(f"Stack size: {len(stack)}")
    while stack:
        seq_template = tuple(stack.pop())
        if seq_template in redundant or seq_template in not_found:
            continue
        for trace in event_template_sets:
            found= False
            trace_length= len(trace)
            seq_length = len(seq_template)
            if seq_length == 0:
                break
            trace_pos= 0
            seq_temp_pos= 0
            while trace_pos < trace_length:
                if seq_template:
                    if all(tuple(elem) in trace[trace_pos]  for elem in seq_template[seq_temp_pos]):
                        trace_pos+=1
                        if seq_temp_pos < seq_length-1:
                            seq_temp_pos+=1
                        else:
                            found =True
                            break
                    else:
                        trace_pos +=1

            if not found:
                break
        if not found:
            not_found.add(seq_template)
            for idx, events_temp in enumerate(seq_template):
                copy_seq_template = seq_template

                if len(events_temp) <= 1:
                    copy_seq_template = copy_seq_template[:idx] + copy_seq_template[idx+1:]
                    if copy_seq_template not in sequence_templates and copy_seq_template not in stack and copy_seq_template not in redundant:
                        stack.append(copy_seq_template)
                else:
                    for idx2, _ in enumerate(events_temp):
                        list_events_temp= list(events_temp)
                        del list_events_temp[idx2]
                        copy_seq_template = copy_seq_template[:idx]+ (tuple(list_events_temp),)  + copy_seq_template[idx+1:]
                        if copy_seq_template not in sequence_templates and copy_seq_template not in stack and copy_seq_template not in redundant:
                            stack.append(copy_seq_template)
        else:
            if seq_template not in sequence_templates and seq_template not in redundant:
                sublists= tuple([tuple(x) for x in tuple(sub_lists(seq_template))])
                for item in sublists:
                    if item in sequence_templates:
                        sequence_templates.remove(item)
                    else:
                        if item not in redundant:
                            all_lists= []
                            for idx, template in enumerate(item):
                                subsublist=  [tuple(x) for x in tuple(sub_lists(template))]
                                all_lists.append(list(subsublist))
                            res = tuple(product(*list(all_lists)))
                            for i in res:
                                redundant.add(i)
                                if i in sequence_templates:
                                    sequence_templates.remove(i)

                #All subsequences of a maximal frequent sequence cannot be maximal
                sequence_templates.add(seq_template)



    return sequence_templates

def sub_lists(my_list):
    """Calculates all sublists except the empty list from a given list.

    Args:
        my_list (list): list of objects

    Returns:
        list: list of all sub-lists
    """
    subs = []
    for i in range(0, len(my_list)+1):
        temp = [list(x) for x in combinations(my_list, i)]
        if len(temp)>0:
            subs.extend(temp)
    return subs[1:]

def seq_templates(arr):
    """Python3 program to find combinations from n arrays such that one element from each
        array is present

    Args:
        arr (List): List of sequence template sets

    Returns:
        List: list of sequence templates
    """
    # number of arrays
    num_arrays = len(arr)

    # to keep track of next element
    # in each of the n arrays
    indices = [0 for i in range(num_arrays)]
    template_set=[]
    while True:
        pos_template=[]
        # prcurrent combination
        for i in range(num_arrays):
            if arr[i]:
                pos_template.append(arr[i][indices[i]][0])
        template_set.append(" ".join(pos_template))

        # find the rightmost array that has more
        # elements left after the current element
        # in that array
        next_element = num_arrays - 1
        while (next_element >= 0 and
              (indices[next_element] + 1 >= len(arr[next_element]))):
            next_element-=1

        # no such array is found so no more
        # combinations left
        if next_element < 0:
            return template_set

        # if found move to next element in that
        # array
        indices[next_element] += 1

        # for all arrays to the right of this
        # array current index again points to
        # first element
        for i in range(next_element + 1, num_arrays):
            indices[i] = 0
def eventset_vertdb(sample_set, complete):
    """Calculates the vertical database for the given sample and returns 2 dictionaries.

    Args:
        sample_set (list): list of traces (Strings)
        complete (boolean): Determines the variation of il-Miner. True resolves to all descriptive queries.
                            False to reduced set defined in il-miner paper.
                            Default: True.

    Returns:
        dictionary: {eventset : list of all subsets}
        dictionary: {event_template with supp=1 : {trace_id : list of occurence positions}}
    """
    eventset = {}
    event_temp_dict = {}
    for idx, trace in enumerate(sample_set, 1):
        for event_pos, event in enumerate(trace.split()):
            if event not in eventset:
                possible_event_templates= powerset(event, complete)
                eventset[event] = possible_event_templates
                for event_temp in possible_event_templates:
                    if event_temp not in event_temp_dict and idx==1:
                        event_temp_dict[event_temp]={}
                        event_temp_dict[event_temp][idx]=[event_pos]
                    else:
                        if event_temp in event_temp_dict:
                            if len(event_temp_dict[event_temp])>= idx-1:
                                if idx not in event_temp_dict[event_temp]:
                                    event_temp_dict[event_temp][idx]=[event_pos]
                                else:
                                    event_temp_dict[event_temp][idx].append(event_pos)
            else:
                possible_event_templates = eventset[event]
                for event_temp in possible_event_templates:
                    if event_temp in event_temp_dict:
                        if len(event_temp_dict[event_temp])>= idx-1:
                            if idx not in event_temp_dict[event_temp]:
                                event_temp_dict[event_temp][idx]=[event_pos]
                            else:
                                event_temp_dict[event_temp][idx].append(event_pos)
    event_to_del = []
    for event_temp, value in event_temp_dict.items():
        if len(value) < len(sample_set):
            event_to_del.append(event_temp)

    for event_temp in event_to_del:
        del event_temp_dict[event_temp]

    return eventset, event_temp_dict
