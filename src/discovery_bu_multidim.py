#!/usr/bin/python3
"""Contains functions for discovering queries from samples by using bottom up algorithms."""

import logging
import time
from collections import deque, Counter
from itertools import combinations, product, chain, combinations_with_replacement, permutations
from copy import deepcopy
from math import ceil
import numpy as np

from sample import Sample
from discovery_bu_pts_multidim import discovery_bu_pts_multidim
from query_multidim import MultidimQuery
from query import Query
from hyper_linked_tree import HyperLinkedTree



#Logger Configuration:
LOG_FORMAT = '| %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

def bu_discovery_multidim(sample, supp, matchtest='smarter', domain_seperated= True, max_query_length = -1,
                          only_types = False) -> dict:
    """Query Discovery by using a bottom up depth-first search.

    Args:
        sample: Sample instance.
        supp: Float between 0 and 1 which describes the requested support.
        matchtest: String to determine the kind of matchtest to use for matching.
        domain_seperated: True iff each domain is mined separately.
        merge_option: int to define which merge function to use. Defaults to None.

    Returns:
        set of queries if a query has been discovered, None otherwise.

    Raises:
        InvalidQuerySupportError: Supp is less than 0 or greater than 1.
    """
    if max_query_length == -1:
        threshold = ceil(sample._sample_size * supp)
        trace_length = sorted([len(trace.split()) for trace in sample._sample])

        max_query_length = trace_length[sample._sample_size - threshold]
    if domain_seperated:
        return domain_seperated_discovery(sample=sample, supp=supp, matchtest=matchtest,
                                          max_query_length=max_query_length)

    return domain_unified_discovery(sample=sample, supp=supp, matchtest=matchtest, 
                                    max_query_length=max_query_length, only_types=only_types)

def domain_unified_discovery(sample, supp, matchtest, max_query_length, only_types) -> dict:
    """Query Discovery by using a unified bottom up depth-first search.

    Args:
        sample: Sample instance.
        supp: Float between 0 and 1 which describes the requested support.
        matchtest: String to determine the kind of matchtest to use for matching.

    Returns:
        Set of queries if a query has been discovered, None otherwise.
    """
    if matchtest == 'smarter':
        return domain_unified_discovery_smarter(sample, supp, max_query_length, only_types=only_types)


def domain_unified_discovery_smarter(sample, supp, max_query_length, only_types=False, find_descriptive_only=True,
                                     all_patternset = None) -> dict:
    """Query Discovery by using unified bottom up depth-first search with smarter matching.

    Args:
        sample: Sample instance.
        supp: Float between 0 and 1 which describes the requested support.

    Returns:
        Set of queries if a query has been discovered, None otherwise.
    """
    if max_query_length == -1:
        threshold = ceil(sample._sample_size * supp)
        trace_length = sorted([len(trace.split()) for trace in sample._sample])

        max_query_length = trace_length[sample._sample_size - threshold]
    query_dict= {}
    matching_dict = {}
    non_matching_dict = {}
    domain_cnt = sample._sample_event_dimension
    alphabet = set()
    if supp == 1.0:
        _,min_trace_length= sample.get_sample_min_trace()
        max_query_length = min(max_query_length, min_trace_length)
    gen_event= ';' * domain_cnt
    gen_event_list = [i for i in gen_event]
    att_vsdb = sample.get_att_vertical_sequence_database()
    # if not find_descriptive_only:
    sample_size = sample._sample_size
    vsdb = {}
    if all_patternset:
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
    else:
        patternset ={}
        all_patternset = {}
        for domain, dom_vsdb in att_vsdb.items():
            patternset[domain] = set()
            all_patternset[domain] = {trace_id: set() for trace_id in range(sample_size)}
            for key, value in dom_vsdb.items():
                new_key = ''.join(gen_event_list[:domain] + [key] + gen_event_list[domain:])
                vsdb[new_key] = value
                if not only_types:
                    for item in value.keys():
                        if len(value[item]) >= 2:
                            all_patternset[domain][item].add(key)
                            patternset[domain].add(key)
                            # break


    sample_sized_support = ceil(sample._sample_size * supp)
    alphabet = {symbol for symbol,value in vsdb.items() if len(value) >= sample_sized_support}
    parent_dict = {}
    alphabet=sorted(alphabet)
    query = MultidimQuery()
    query.set_query_string(gen_event)
    querystring= query._query_string
    matching_dict[querystring] = query
    stack= deque()
    dict_iter = {}
    matching = True
    querycount=1
    dictionary= {}
    parent_dict[querystring] = query

    children = _next_queries_multidim(query,alphabet, max_query_length, patternset)
    parent_dict.update({child._query_string: query for child in children})
    grand_children = []
    non_descriptive = set()
    
    stack.extend(children)
    query_tree = HyperLinkedTree(ceil(supp*sample._sample_size), event_dimension=sample._sample_event_dimension)
    
    start_time = time.time()
    last_print_time = start_time


    while stack:
        query = stack.pop()
        querystring = query._query_string
        query.set_query_matchtest('smarter')
        querycount+=1
        current_time = time.time()
        # if current_time - start_time > 43200: # 12 hours
        #     break
        if current_time - last_print_time > 300:
            LOGGER.info('Current query: %s; current stack size: %i; Current Query count: %i', querystring, len(stack), querycount)
            last_print_time = current_time
        parent = parent_dict[querystring]
        parentstring = parent._query_string
        matching= query.match_sample(sample = sample, supp= supp, dict_iter = dict_iter, patternset = all_patternset, parent_dict = parent_dict)
        dictionary.update({querystring:matching})

        if not matching:
            non_matching_dict[querystring] = query
            
        else:
            matching_dict[querystring] = query
            non_descriptive.add(parentstring)
            # query_dict.pop(parentstring, None)

            if parent_dict[querystring]._query_string == gen_event:
                parentstring = ''
            else:
                parentstring = parent_dict[querystring]._query_string
            parent_vertex = query_tree.find_vertex(parentstring)
            if not query_tree.find_vertex(querystring):
                vertex = query_tree.insert_query_string(parent_vertex, querystring, query=query, search_for_parents=False)
                vertex.matched_traces = query._query_matched_traces
            children = _next_queries_multidim(query,alphabet, max_query_length, patternset)
            if children:
                stack.extend(children)
                parent_dict.update({child._query_string: query for child in children})

            
    result_dict = {}
    if find_descriptive_only:
        
        queryset, query_tree = ht_descriptive_queries(query_tree, set(matching_dict.keys()))
        result_dict['queryset'] = queryset - {gen_event}
        
    else:
        result_dict['queryset'] = set(matching_dict.keys()) - {gen_event} - {''}

    result_dict['querycount'] =  querycount
    result_dict['parent_dict'] = parent_dict
    result_dict['matching_dict'] = matching_dict
    result_dict['dict_iter'] = dict_iter
    result_dict['query_tree'] = query_tree
    result_dict['non_matching_dict'] = non_matching_dict
    result_dict['patternset'] = patternset



    return result_dict



def domain_seperated_discovery(sample, supp, matchtest, max_query_length:int):
    """Query Discovery Algorithm.
        This algorithm uses a bottom-up approach.
        Every domain is discovered separately and
        merged at the end.
        The discovery that is used for each domain can be specified:
        Either: depth-first ->'smarter' (default) or
                breadt-first -> 'pattern-split-sep'
    Args:
        sample: Sample instance.
        supp: Float between 0 and 1 which describes the requested support.
        matchtest: String to determine the kind of matchtest to use for matching.

    Returns:
        Set of queries if a query has been discovered, None otherwise.
    """
    sample_set = sample._sample
    sample_size = len(sample_set)
    domain_cnt = sample_set[0].split(' ')[0].count(';')
    if domain_cnt == 1:
        if matchtest=='smarter':
            return domain_unified_discovery_smarter( sample=sample, supp=supp, max_query_length = max_query_length, find_descriptive_only=True)
        else:
            return discovery_bu_pts_multidim(sample, supp=supp, use_smart_matching=True, discovery_order='type_first',
                                                    use_tree_structure=True, max_query_length=max_query_length,
                                                    find_descriptive_only=True)[2]
    
    if supp == 1.0:
        _, min_trace_length= sample.get_sample_min_trace()
        max_query_length = min(max_query_length, min_trace_length)
    non_descriptives_dict={}
    query_dict = {}
    dict_iter= {}
    parent_dict = {}
    sample_list = []
    merge1 = 0
    merge2 = 0
    dim_sample_dict = sample.get_dim_sample_dict()

    domain_query_list=[]
    all_dictionary={}
    query_list= {}
    gen_event= ';' * domain_cnt
    gen_event_list = [i for i in gen_event]
    patternset = {}
    all_patternset = {}
    
    for domain, domain_sample in dim_sample_dict.items():
        
        domain_patternset = {}
        # domain_patternset = {dom:{} for dom in range(domain_cnt)}
        vert_db = domain_sample.get_att_vertical_sequence_database()
        # domain_patternset = set([key for key in vert_db[domain].keys() for item in vert_db[domain][key].keys() if len(vert_db[domain][key][item]) >= 2])
        all_patternset[domain] = {trace_id: set() for trace_id in range(sample_size)}
        for key, value in vert_db.items():
            for letter, pos_dict in value.items():
                for trace_id, positions in pos_dict.items():
                    # if len(value[item]) >= 2:
                    if len(positions) >=2:
                        all_patternset[domain][trace_id].add(letter)
                        # patternset[domain].add(key)
                        # break

        domain_patternset[domain] = all_patternset[domain]
        if matchtest=='smarter':
            result_dict = domain_unified_discovery_smarter( sample=domain_sample, supp=supp, max_query_length = max_query_length, find_descriptive_only=False,
                                                           all_patternset=domain_patternset)
        else:
            result_dict = discovery_bu_pts_multidim(domain_sample, supp=supp, use_smart_matching=True, discovery_order='type_first',
                                                    use_tree_structure=True, max_query_length=max_query_length,
                                                    find_descriptive_only=False, all_patternset=domain_patternset)[2]
        instance_dictionary = {}
        sample_size = domain_sample._sample_size

        trace_list=list(range(sample_size))

        for result_query in result_dict['matching_dict'].values():
            if supp < 1:
                trace_list = []
                for trace in range(sample_size):
                    querystring = result_query._query_string
                    if '$' in querystring:
                        if trace in result_dict['dict_iter'][querystring]:
                            trace_list.append(trace)

                    else:
                        if trace in result_dict['dict_iter'][querystring]:
                            if result_dict['dict_iter'][querystring][trace] != -1:
                                trace_list.append(trace)
            instance_dictionary = result_query.query_pos_dict(
                vert_db,
                domain_sample,
                instance_dictionary, trace_list=trace_list)

        pos_dict = instance_dictionary
        domain_queryset = set(result_dict['queryset'])
        query_list.update(result_dict['matching_dict'].items())  # type: ignore
        # if 'dict_iter' in result_dict and result_dict['matching_dict']:
        if result_dict['matching_dict']:
            dict_iter.update(result_dict['dict_iter'])  # type: ignore
            
        if 'parent_dict' in result_dict:
            parent_dict.update(result_dict['parent_dict'])  # type: ignore

        for domain_query in domain_queryset:
            if gen_event == domain_query or not domain_query:
                continue

            all_dictionary[domain_query] = {'trace_instances' :pos_dict[domain_query],
                                                'occurences': len(pos_dict[domain_query])}
        domain_query_list.append(list(domain_queryset))

        if domain == 0:
            mixed_query_tree = result_dict['query_tree']
        else:
            vertices = []
            domain_tree = result_dict['query_tree']
            root_vertex_domain = domain_tree.get_root()
            root_vertex = mixed_query_tree.get_root()
            child_vertices = root_vertex_domain.child_vertices
            for child_vertex in child_vertices:
                mixed_query_tree.insert_query_string(root_vertex, child_vertex.query_string, query=child_vertex.query, search_for_parents=False)



    queryset=set()

    descriptive_query_list = set()
    seen = set()
    non_empty_keys = {i : dom_list for i, dom_list in enumerate(domain_query_list)}
    empty_domains = set()
    for dom, dom_list in non_empty_keys.items():
        if matchtest == 'smarter':
            dom_list.append('')
        elif len(dom_list) == 1:
            empty_domains.add(dom)
        non_empty_keys[dom] = dom_list
    for dom in empty_domains:
       del non_empty_keys[dom]

    query_pairs_v2  = sorted(product(*non_empty_keys.values()))
    matchings = {}
    pair_dict_matching ={}
    pair_dict_non_matching ={}
    non_matching = set()
    seen = set()

    for pair in query_pairs_v2:
        pair_dict_matching[pair] = []
        pair_dict_non_matching[pair] = []
        parent_tuples = []
        parent_tuple_count = []
        parent_tuple_doms = []
        for idx, domain, dom_tuple in zip(range(len(pair)),non_empty_keys.keys(),pair):
            if dom_tuple:
                if dom_tuple in parent_dict:
                    parent_string = parent_dict[dom_tuple]._query_string
                    if parent_string == gen_event:
                        parent_string = ''
                else:
                    query = MultidimQuery()
                    query.set_query_string(dom_tuple, recalculate_attributes=False)
                    parent = query._parent()
                    parent_string = parent._query_string
            else:
                parent_string = ''
            parent_tuple_list = list(pair)
            parent_tuple_list[idx] = parent_string
            parent_tuple = tuple(parent_tuple_list)

            if parent_tuple != pair:
                parent_tuple_count.append(len([i for i in parent_tuple if i]))
                parent_tuples.append(parent_tuple)
                parent_tuple_doms.append([i for i,element in enumerate(parent_tuple) if element])
        poss_queries_v2= set()


        if '' in pair or gen_event in pair:
            domain_indeces_dict = {dom_idx: pair[i] for i, dom_idx in 
                                   enumerate(non_empty_keys.keys()) 
                                   if pair[i] not in ['', gen_event]}
            domain_indeces = list(non_empty_keys.keys())
        else:
            domain_indeces = list(non_empty_keys.keys())
            domain_indeces_dict = {domain_idx: domain_string for domain_idx, domain_string in zip(domain_indeces, pair) }
        if len(domain_indeces_dict) <= 1:
            matching_queryset = set()
            if domain_indeces_dict:
                querystring = list(domain_indeces_dict.values())[0]
                pair_dict_matching[pair].append(querystring)
                descriptive_query_list.add(querystring)
                if querystring in parent_dict:
                    parentstring = parent_dict[querystring]._query_string
                else:
                    parent = query._parent()
                    parentstring = parent._query_string
                if parentstring == gen_event and matchtest == 'smarter':
                    parentstring = ''
                parent_vertex = mixed_query_tree.find_vertex(parentstring)
                if not mixed_query_tree.find_vertex(querystring):
                    mixed_query_tree.insert_query_string(parent_vertex, querystring, query=query_list[querystring] , search_for_parents=False)


        else:
            poss_queries_v2 = _merge_domain_queries(domain_indeces_dict ,all_dictionary, max_query_length, supp)
            poss_query_list = set()
            if poss_queries_v2:
                adapted_querystring_dict = adapted_querystring(domain_indeces_dict, query_list)

                for pair2 in poss_queries_v2:
                    if pair2 and domain_indeces_dict:
                        querystring = pos2query(domain_indeces_dict, pair2, adapted_querystring_dict, max_query_length)
                        if querystring == gen_event or not querystring:
                            continue
                        else:
                            
                            poss_query_list.add(querystring)
                            # query_dict[querystring] = query
            merge2+=1

            matching_queryset = set()

            if poss_query_list:
                # for poss_query in poss_query_list:
                for querystring in poss_query_list:  
                    # querystring = poss_query._query_string
                    poss_query = MultidimQuery()
                    poss_query.set_query_string(querystring, recalculate_attributes= False)
                    parent = poss_query._parent()
                    querystring = poss_query._query_string
                    poss_query.set_query_matchtest('smarter')
                    parent_dict[querystring] = parent
                    if parent._query_string not in non_matching:
                        seen.add(querystring)
                        match = poss_query.match_sample(sample, supp, parent_dict=parent_dict, patternset=all_patternset, dict_iter=dict_iter)
                        matchings[querystring] = match
                        if match and len(querystring.split()) <= max_query_length:
                            query_dict[querystring] = poss_query
                            pair_dict_matching[pair].append(querystring)
                            matching_queryset.add(querystring)

                            
                            
                            mixed_query_tree = add_vertex2tree(poss_query, parent, mixed_query_tree, gen_event, parent_dict, matchtest)
                        else:
                            pair_dict_non_matching[pair].append(querystring)
                            non_matching.add(querystring)
            descriptive_query_list.update(matching_queryset)

    non_descriptive = set()
    
    result_dict = {}
    # result_dict['queryset'] = descriptive_query_list - non_descriptive
    result_dict['queryset'], mixed_query_tree = ht_descriptive_queries(mixed_query_tree, descriptive_query_list)
    result_dict['matching_dict'] = query_dict
    result_dict['domain_queries'] = [len(domain_list) for domain_list in domain_query_list]
    result_dict['merged queries'] = len(queryset)
    
    
    all_variables = dir() 
  
    return result_dict

def add_vertex2tree(poss_query, parent, mixed_query_tree, gen_event, parent_dict, matchtest):
    querystring = poss_query._query_string
    if parent._query_string == gen_event and matchtest == 'smarter':
        parentstring = ''
    else:
        parentstring = parent._query_string

    parent_vertex = mixed_query_tree.find_vertex(parentstring)
    while parent_vertex is None:
        mixed_query_tree = add_vertex2tree(parent, parent_dict[parentstring], mixed_query_tree, gen_event, parent_dict, matchtest)
    if not mixed_query_tree.find_vertex(querystring):
        mixed_query_tree.insert_query_string(parent_vertex, querystring, query=poss_query, search_for_parents=False)

    return mixed_query_tree

def _next_queries_multidim(query, alphabet, max_query_length, patternset, only_types = False):
    """Given a query the function calculates the children of that query which are the next queries that are more specialised adding one element following the rule set.

    Args:
        query (Query): an instance of Query
        alphabet (list): list of supported types
        max_query_length (int): maximal number of events in a query

    Returns:
        Children of the given query: list of query strings

    """
    querystring = query._query_string
    
    querylength= query._query_string_length
    querystring_list = query.get_query_list()
    domain_cnt = query._query_event_dimension
    variables=query._query_repeated_variables
    if variables:
        last_var = sorted(variables)[-1]
    typeset = query._query_typeset
    num_of_vars= len(variables)
    gen_event= ';' * domain_cnt
    gen_event_list = [i for i in gen_event]
    if querystring == '':
        return [MultidimQuery(gen_event)]
    children=[]
    children_strings = set()
    pos_last_type = query._pos_last_type_and_variable[0]
    pos_first_var = query._pos_last_type_and_variable[1]
    pos_last_var = query._pos_last_type_and_variable[2]


    #special case: most general query
    if querystring == gen_event:
        if max_query_length>= 2 and not only_types:
            for domain in range(domain_cnt):
                if patternset[domain]:
                    domain_var= ''.join(gen_event_list[:domain] + ['$x0'] + gen_event_list[domain:])
                    child=domain_var + ' ' + domain_var
                    child_query= MultidimQuery()
                    child_query._query_string = child
                    child_query._query_repeated_variables= {'x0'}
                    child_query._query_string_length= 2
                    child_query._pos_last_type_and_variable = np.array([-1, 0, 1])
                    child_query._query_event_dimension = query._query_event_dimension
                    children.append(child_query)

        if max_query_length>=1:
            for letter in alphabet:
                child=str(letter)
                child_query= MultidimQuery()
                child_query._query_string = child
                child_query._query_typeset= typeset | {letter}
                child_query._query_string_length= 1
                child_query._pos_last_type_and_variable = np.array([0, -1,-1])
                child_query._query_event_dimension = query._query_event_dimension
                children.append(child_query)



    #non-empty querystrings
    else:
        
        #insert new variable twice: after last occurence of type and after first occurence of last variable
        first_pos = max(pos_last_type, pos_first_var)
        first_pos_event= querystring_list[first_pos]
        first_pos_domains = query.non_empty_domain(first_pos_event)
        #first_pos_domain = domain_cnt
        for domain in first_pos_domains:
            att = first_pos_event.split(';')[domain]
            if first_pos == pos_last_type and att.count('$') == 0:
                first_pos_domain = domain
            if variables:
                if first_pos == pos_first_var and last_var in att:
                    first_pos_domain = domain

        querystring_split = querystring_list[first_pos:]
        if not only_types:
            for domain in range(domain_cnt):
                if patternset[domain]:
                    domain_var= ''.join(gen_event_list[:domain] + ['$x' + str(num_of_vars)] + gen_event_list[domain:])
                    var_domain = domain_var.find(domain_var.strip(';'))
                    var = 'x' + str(num_of_vars)
                    for idx, event in enumerate(querystring_split, start = first_pos):
                        if querylength+1 <= max_query_length:
                            if idx != querylength -1:
                                child = " ".join(querystring_list[:idx+1]) + ' '+domain_var +' ' + " ".join(querystring.split()[idx+1:])
                            else:
                                child = querystring + ' ' + domain_var

                            for idx2, event2 in enumerate(child.split()[idx+1:], start = idx+1):
                                if querylength+2 <= max_query_length:
                                    if idx2 != querylength:
                                        child2 = " ".join(child.split()[:idx2+1]) + ' '+domain_var +' ' + " ".join(child.split()[idx2+1:])
                                    else:
                                        child2 = child + ' ' + domain_var

                                    child_query= MultidimQuery()
                                    child_query._query_string = child2.strip()
                                    child_query._query_typeset= typeset
                                    child_query._query_repeated_variables = variables |{var}
                                    child_query._query_string_length= querylength + 2
                                    child_query._query_event_dimension = query._query_event_dimension
                                    child_query._pos_last_type_and_variable = np.array([pos_last_type, idx+1, idx2+1])
                                    assert child_query._query_string_length <= max_query_length
                                    if child_query._query_string not in children_strings:
                                        children.append(child_query)
                                        children_strings.add(child_query._query_string)

                                last_non_empty = query.non_empty_domain(event2)[-1]
                                if not event2.split(';')[var_domain] and idx2 <= querylength +1:
                                    #if  var_domain > last_non_empty:
                                    new_event = event2.split(';')
                                    new_event[var_domain] = domain_var.strip(';')
                                    # if idx == 0:
                                    #     child3 = ';'.join(new_event)
                                    if idx == querylength -1:
                                        child3 = " ".join(child.split()[:idx2]) + ' ' +  ';'.join(new_event)
                                    else:
                                        child3 = " ".join(child.split()[:idx2]) + ' ' +  ';'.join(new_event)+ ' ' + " ".join(child.split()[idx2+1:])
                                    child_query= MultidimQuery()
                                    child_query._query_string = child3.strip()
                                    child_query._query_typeset= typeset
                                    child_query._query_repeated_variables = variables |{var}
                                    child_query._query_string_length= querylength +1
                                    child_query._query_event_dimension = query._query_event_dimension
                                    child_query._pos_last_type_and_variable = np.array([pos_last_type, idx+1, idx2])
                                    assert child_query._query_string_length <= max_query_length
                                    if child_query._query_string not in children_strings:
                                        children.append(child_query)
                                        children_strings.add(child_query._query_string)


                        #last_non_empty = non_empty_domain(event)[-1]
                        if not event.split(';')[var_domain]:
                            if idx != first_pos or var_domain > first_pos_domain:
                                new_event = event.split(';')
                                new_event[var_domain] = domain_var.strip(';')
                                # if idx == 0:
                                #     child = ';'.join(new_event)
                                if idx == querylength -1:
                                    child = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)
                                else:
                                    child = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)+ ' ' + " ".join(querystring_list[idx+1:])

                                for idx2, event2 in enumerate(child.split()[idx:], start = idx):
                                    if querylength+1 <= max_query_length:
                                        if idx2 != querylength -1:
                                            child2 = " ".join(child.split()[:idx2+1]) + ' '+domain_var +' ' + " ".join(child.split()[idx2+1:])
                                        else:
                                            child2 = child + ' ' + domain_var
                                        child_query= MultidimQuery()
                                        child_query._query_string = child2.strip()
                                        child_query._query_typeset= typeset
                                        child_query._query_repeated_variables = variables |{var}
                                        child_query._query_string_length= querylength + 1
                                        child_query._query_event_dimension = query._query_event_dimension
                                        child_query._pos_last_type_and_variable = np.array([pos_last_type, idx, idx2])
                                        assert child_query._query_string_length <= max_query_length

                                        if child_query._query_string not in children_strings:
                                            children.append(child_query)
                                            children_strings.add(child_query._query_string)

                                    last_non_empty = query.non_empty_domain(event2)[-1]
                                    if not event2.split(';')[var_domain]:
                                        if idx2 != first_pos: #or var_domain > last_non_empty:
                                            new_event = event2.split(';')
                                            new_event[var_domain] = domain_var.strip(';')
                                            # if idx == 0:
                                            #     child3 = ';'.join(new_event)
                                            if idx == querylength -1:
                                                child3 = " ".join(child.split()[:idx2]) + ' ' +  ';'.join(new_event)
                                            else:
                                                child3 = " ".join(child.split()[:idx2]) + ' ' +  ';'.join(new_event)+ ' ' + " ".join(child.split()[idx2+1:])
                                            child_query= MultidimQuery()
                                            child_query._query_string = child3.strip()
                                            child_query._query_typeset= typeset
                                            child_query._query_repeated_variables = variables |{var}
                                            child_query._query_string_length= querylength
                                            child_query._query_event_dimension = query._query_event_dimension
                                            child_query._pos_last_type_and_variable = np.array([pos_last_type, idx, idx2])
                                            assert child_query._query_string_length <= max_query_length

                                            if child_query._query_string not in children_strings:
                                                children.append(child_query)
                                                children_strings.add(child_query._query_string)


        #insert last inserted variable again
        if pos_first_var>= pos_last_type:
            if pos_first_var !=-1: # and pos_first_var:
                var_numb=0
                for domain, letter in enumerate(querystring_list[pos_first_var].split(';')):
                    if '$' in letter and var_numb <= int(letter.strip('$x;')):
                        last_variable_domain = domain
                        var_numb = int(letter.strip('$x;'))

                last_variable = querystring_list[pos_first_var].split(';')[last_variable_domain]
                num_of_vars= int(last_variable.strip('$x;')) +1
                domain_var= ''.join(gen_event_list[:last_variable_domain] + [last_variable] + gen_event_list[last_variable_domain:])
            # else:
            #     last_variable = '$x0'
            #     num_of_vars= 0
            no_letter= True
            if pos_first_var == pos_last_type:
                for event in querystring_list[pos_first_var].split(';')[last_variable_domain+1:-1]:
                    if event.count('$') == 0 and event:
                        no_letter = False
            if no_letter and not only_types:
                first_pos = max(pos_last_type, pos_last_var)
                querystring_split = querystring_list[first_pos:]
                for idx, event in enumerate(querystring_split, start = first_pos):
                    if querylength+1 <= max_query_length:
                        if idx != querylength -1:
                            child = " ".join(querystring_list[:idx+1]) + ' '+domain_var +' ' + " ".join(querystring_list[idx+1:])
                        else:
                            child = querystring + ' ' + domain_var
                        child_query= MultidimQuery()
                        child_query._query_string = child.strip()
                        child_query._query_typeset= typeset
                        child_query._query_repeated_variables = variables
                        child_query._query_string_length= querylength + 1
                        child_query._query_event_dimension = query._query_event_dimension
                        child_query._pos_last_type_and_variable = np.array([pos_last_type, pos_first_var, idx+1])
                        assert child_query._query_string_length <= max_query_length

                        if child_query._query_string not in children_strings:
                            children.append(child_query)
                            children_strings.add(child_query._query_string)
                    var_domain = domain_var.find(domain_var.strip(';'))
                    last_non_empty = query.non_empty_domain(event)[-1]
                    if not event.split(';')[var_domain]:
                        #if idx != first_pos or var_domain > last_non_empty:
                        new_event = event.split(';')
                        new_event[var_domain] = domain_var.strip(';')
                        if idx == 0:
                            child2 = ';'.join(new_event)
                        elif idx == querylength -1:
                            child2 = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)
                        else:
                            child2 = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)+ ' ' + " ".join(querystring_list[idx+1:])
                        child_query= MultidimQuery()
                        child_query._query_string = child2.strip()
                        child_query._query_typeset= typeset
                        child_query._query_repeated_variables = variables
                        child_query._query_string_length= querylength
                        child_query._query_event_dimension = query._query_event_dimension
                        child_query._pos_last_type_and_variable = np.array([pos_last_type, pos_first_var, idx])
                        assert child_query._query_string_length <= max_query_length

                        if child_query._query_string not in children_strings:
                            children.append(child_query)
                            children_strings.add(child_query._query_string)


        #insert types: after last occurence of type and after first occurence of last variable
        first_pos = max(pos_last_type, pos_first_var)
        first_pos_event= querystring_list[first_pos]
        if 'last_variable_domain' in locals():
            if pos_first_var != pos_last_type:
                last_symbol_domain = last_variable_domain
            else:
                for domain, letter in enumerate(first_pos_event.split(';')):
                    if letter:
                        last_symbol_domain = domain
        else:
            for domain, letter in enumerate(first_pos_event.split(';')):
                if letter and '$' not in letter:
                    last_symbol_domain = domain

        querystring_split = querystring_list[first_pos:]
        for letter in alphabet:
            for idx, event in enumerate(querystring_split, start = first_pos):
                if querylength+1 <= max_query_length:
                    if idx != querylength -1:
                        child = " ".join(querystring_list[:idx+1]) + ' '+letter +' ' + " ".join(querystring_list[idx+1:])
                    else:
                        child = querystring + ' ' + letter
                    child_query= MultidimQuery()
                    child_query._query_string = child.strip()
                    child_query._query_typeset= typeset | {letter}
                    child_query._query_repeated_variables = variables
                    child_query._query_string_length= querylength + 1
                    child_query._query_event_dimension = query._query_event_dimension
                    if idx < pos_last_var:
                        child_query._pos_last_type_and_variable = np.array([idx+1, pos_first_var, pos_last_var+1])
                    else:
                        child_query._pos_last_type_and_variable = np.array([idx+1, pos_first_var, pos_last_var])
                    assert child_query._query_string_length <= max_query_length
                    if child_query._query_string not in children_strings:
                        children.append(child_query)
                        children_strings.add(child_query._query_string)
                letter_domain = letter.find(letter.strip(';'))
                last_non_empty = query.non_empty_domain(event)[-1]
                if not event.split(';')[letter_domain]:
                    if idx != first_pos or letter_domain > last_non_empty or '$' in event.split(';')[last_non_empty]:
                        if letter_domain < last_symbol_domain and idx == first_pos:
                            continue
                        new_event = event.split(';')
                        new_event[letter_domain] = letter.strip(';')
                        # if idx == 0:
                        #     child2 = ';'.join(new_event)
                        if idx == querylength -1:
                            child2 = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)
                        else:
                            child2 = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)+ ' ' + " ".join(querystring_list[idx+1:])
                        child_query= MultidimQuery()
                        child_query._query_string = child2.strip()
                        child_query._query_typeset= typeset | {letter}
                        child_query._query_repeated_variables = variables
                        child_query._query_string_length= querylength
                        child_query._query_event_dimension = query._query_event_dimension
                        child_query._pos_last_type_and_variable = np.array([idx, pos_first_var, pos_last_var])
                        assert child_query._query_string_length <= max_query_length

                        if child_query._query_string not in children_strings:
                            children.append(child_query)
                            children_strings.add(child_query._query_string)


    return children


def _merge_domain_queries(querystring_dict , pos_dict, max_query_length, supp=1.0):
    """Given a set of querystrings from different domains, it returns possible merged queries.

    Args:
        querystring_dict (dict): dictionary with domain index as key and querystring as value.
        pos_dict (dict): dictionary containing positions for each query and each trace.
        query_dict (dict): dictionary containing the query object to the querystring.
        max_query_length (int): max length of the resulting queries. Defaults to 10.
        supp (float): Support value. Defaults to 1.0.

    Returns:
        Set of merged queries
    """
    query_list = list(querystring_dict.values())
    domain_indeces = list(querystring_dict.keys())
    querystring1 = query_list[0]
    domain_cnt = querystring1.split()[0].count(';')
    sample_size = len(pos_dict[querystring1]['trace_instances'])
    gen_event= ';' * domain_cnt
    # domain_cnt = -1
    all_positions = {}
    for domain, querystring in querystring_dict.items():
        all_positions[domain]= pos_dict[querystring]['trace_instances']


    if domain_cnt == -1:
        #return set()
        return {}

    
    min_length = -1
    trace_id = 0
    number_of_traces = ceil(sample_size - supp*sample_size) + 1
    trace_id_list = []
    if supp == 1.0:
        for idx in range(sample_size):
            # query_occ_list = [pos_dict[querystring]['occurences'] for querystring in query_list]
        
            query_occ_list = [len(pos_dict[querystring]['trace_instances'][idx]) for querystring in 
                            query_list]
            trace_product = np.prod(query_occ_list)
            if trace_product <= 3*len(query_occ_list):
                trace_id_list= [idx]
                break
            if min_length == -1 or min_length > trace_product:
                min_length = trace_product
                trace_id_list = [idx]
        
            

    else:
        # Get the list of keys in all inner dictionaries
        all_trace_ids = [set(inner_dict.keys()) for inner_dict in all_positions.values()]

        # Find the common keys using set intersection
        query_occ_list = set.intersection(*all_trace_ids)

        # Convert the result to a list
        trace_id_list = list(query_occ_list)


    all_instance_pairs = []
    if len(trace_id_list) < supp*sample_size and supp < 1.0:
        return all_instance_pairs

    for idx in trace_id_list[:number_of_traces]:
        instance_trace_list = [all_positions[domain][idx] for domain in domain_indeces]
        # instance_trace_list = [all_positions[domain] for domain in domain_indeces]
        instance_pairs= list(product(*instance_trace_list))
        all_instance_pairs.extend(instance_pairs)
        

    return all_instance_pairs

def _merge_domain_queries2(querystring_dict, query_dict, max_query_length):
    """Given a set of querystrings from different domains, it returns possible merged queries.

    Args:
        querystring_dict (dict): dictionary with domain index as key and querystring as value.
        query_dict (dict): dictionary containing the query object to the querystring.
        max_query_length (int): max length of the resulting queries. Defaults to 10.

    Returns:
        Set of merged queries
    """
    if len(querystring_dict) <= 1:
        
        if querystring_dict:
            return tuple([tuple([tuple(tuple(range(list(querystring_dict.values())[0].count(' ') +1)))])])
        else:
            return tuple([tuple()])

    query_list = list(querystring_dict.values())
    domain_cnt = -1
    for querystring in query_list:
        querystring_list = query_dict[querystring].get_query_list()
        if querystring:
            domain_cnt = querystring_list[0].count(';')
            break

    if domain_cnt == -1:
        return tuple([tuple()])


    dom_query_length = [query_dict[qstring]._query_string_length for qstring in query_list]
    dom_number = len(query_list)
    max_dom_query_length = sum(dom_query_length)
    min_length = max(dom_query_length)
    max_length = min(max_query_length, max_dom_query_length)

    sliced_instance_positions = set()
    for q_length in range(min_length,max_length+1):
        overlap = max_dom_query_length - q_length
        stable_query_length = int(max_dom_query_length/(q_length + overlap))
        query_bases = stable_query_length * tuple(range(q_length))

        if overlap != 0:
            pos_query_lengths = list(combinations_with_replacement(range(q_length), r=overlap))
            pos_positions=[]

            for el in pos_query_lengths:
                c= Counter(el)
                most_repeat = c.most_common(1)[0]
                if most_repeat[1]<= dom_number - stable_query_length:
                    pos_positions.append(el)
            instance_positions = [query_bases + el for el in pos_positions]

        else:
            pos_query_lengths= []
            instance_positions = [query_bases]

        for instance in instance_positions:
            instances = calc_instance_pairs(sorted(instance), dom_query_length, max_dom_query_length)
            for inst in instances:
                sliced_instance_positions.add(inst)

    
    return sliced_instance_positions

def _merge_domain_queries1(parent_queries, parent_dict= None, all_dictionary = None, max_query_length=10):
    """

    Args:
        parent_queries (_type_): _description_
    """
    sliced_instance_positions = set()
    children = []
    dom_length = [i for i in range(len(parent_queries))]
    dim_dict = {}

    for dim, dim_list in enumerate(parent_queries):
        # parent_list = []
        dim_dict[dim] = {}
        for querystring in dim_list:
            domain_cnt = all_dictionary[querystring]._query_event_dimension
            gen_event= ';' * domain_cnt
            gen_event_list = [i for i in gen_event]
            parentstring = parent_dict[querystring]._query_string
            dim_dict[dim][parentstring] = set()
            if all_dictionary[querystring]._last_inserted_ele:
                dim_dict[dim][parentstring].add(all_dictionary[querystring]._last_inserted_ele)
            else:
                last_inserted_pos = max(all_dictionary[querystring]._pos_last_type_and_variable[:-1])
                last_event = querystring.split()[last_inserted_pos].split(';')[:-1]
                last_element = ''
                for domain, att in reversed(list(enumerate(last_event))):
                    if att:
                        if not last_element or '$' not in att:
                            last_element = ''.join(gen_event_list[:domain] + [att] + gen_event_list[domain:])
                        elif int(att.strip('$x')) > int(last_element.strip(';$x')):
                            last_element = ''.join(gen_event_list[:domain] + [att] + gen_event_list[domain:])
                        else:
                            pass
                        if '$' not in att:
                            break
                dim_dict[dim][parentstring].add(last_element)


    parent_set = []
    for query_dict in dim_dict.values():
        parent_set.extend(query_dict.keys())

    most_common = Counter(parent_set).most_common()
    for tup in most_common:
        if tup[1] == 1:
            pass
        else:
            alphabet_sets = [query_dict[tup[0]] for query_dict in dim_dict.values() if tup[0] in query_dict]
            set_combinations = list(chain(*[list(permutations(att_list))  for att_list in  product(*alphabet_sets)]))
            for event_list in set_combinations:
                query_list = [all_dictionary[tup[0]]]
                for event in event_list:
                    queries = []
                    for query in query_list:
                        if '$' in event:
                            # if query._query_string not in seen:
                            if event.strip(';') in query._query_string:
                                event_count = domain_cnt - event.lstrip(';').count(';')
                                for ev in query._query_string.split():
                                    if event.strip(';') in ev:
                                        ev_count = ev.split(';').index(event.strip(';'))
                                        break
                                if event_count == ev_count:
                                    queries.extend(_next_queries_multidim(query= query, alphabet=[], max_query_length=max_query_length))
                                else:
                                    queries.extend(_next_queries_multidim(query= query, alphabet=[], max_query_length=max_query_length))

                            else:
                                event_count = event.lstrip(';').count(';') -1
                                queries.extend(_next_queries_multidim(query= query, alphabet=[], max_query_length=max_query_length))
                            # seen.add(query._query_string)
                        else:
                            queries.extend(_next_queries_multidim(query= query, alphabet=[event], max_query_length=max_query_length))
                    query_list = queries
                children.extend(query_list)
    return children



def pos2query(querystring_dict, pair, adapted_querystring_dict, max_query_length):
    """_summary_

    Args:
        instance_positions (_type_): _description_
    """
    
    new_query = ''
    new_query_list = []
    instance_positions = sorted(set(chain(*pair)))
    if len(instance_positions) > max_query_length:
        return new_query
    query_list = list(querystring_dict.values())
    domain_indeces = list(querystring_dict.keys())
    domain_cnt = -1
    for querystring in query_list:
        querystring_list = querystring.split()
        if querystring:
            domain_cnt = querystring_list[0].count(';')
            break

    # if len(instance_positions) == instance_positions[-1] + 1:
    for pos in instance_positions:
        domain_pos = [idx for idx, p in zip(domain_indeces, pair) if pos in p]
        last_domain = -1
        instance_count = 0
        for domain in range(domain_cnt):
            if domain in domain_indeces:

                if domain in domain_pos:
                    dom_instance = pair[domain_indeces.index(domain)]
                    instance_type = dom_instance.index(pos)
                    instance_count+=1
                    if last_domain >=0:
                        # new_query = new_query + adapted_querystring_dict[domain].split()[instance_type].strip(';') + ';'
                        new_query_list.append(adapted_querystring_dict[domain].split()[instance_type].strip(';'))
                        new_query_list.append(';')
                    elif new_query:
                        # ew_query =new_query+ ' ' +  adapted_querystring_dict[domain].split()[instance_type].strip(';') + ';'
                        new_query_list.append(' ')
                        new_query_list.append(adapted_querystring_dict[domain].split()[instance_type].strip(';'))
                        new_query_list.append(';')
                    else:
                        # new_query = adapted_querystring_dict[domain].split()[instance_type].strip(';') + ';'
                        new_query_list.append(adapted_querystring_dict[domain].split()[instance_type].strip(';'))
                        new_query_list.append(';')
                else:
                    if last_domain >= 0:
                        # new_query = new_query + ';'
                        new_query_list.append(';')
                    elif new_query:
                        # new_query = new_query+ ' ;'
                        new_query_list.append(' ;')
                    else:
                        # new_query = ';'
                        new_query_list.append(';')

            else:
                if last_domain >= 0:
                    # new_query = new_query + ';'
                    new_query_list.append(';')
                elif new_query:
                    # new_query = new_query+ ' ;'
                    new_query_list.append(' ;')
                else:
                    # new_query = ';'
                    new_query_list.append(';')

            if domain == domain_cnt -1 and pos != instance_positions[-1]:
                # new_query = new_query + ';'
                new_query_list.append(' ')
            last_domain+=1
    # if new_query:
    if new_query_list:
        new_query = ''.join(new_query_list)
        # normal_form = to_normalform(new_query)
        normal_form = reposition_vars(new_query)
        return normal_form
    else:
        return ''



def adapted_querystring(querystring_dict, query_dict):
    """_summary_

    Args:
        querystring_dict (_type_): _description_
        domain_indeces (_type_): _description_
        query_list (_type_): _description_
        query_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    query_list = list(querystring_dict.values())
    domain_indeces = list(querystring_dict.keys())
    var_domains = [domain for domain, querystring in querystring_dict.items() if querystring.count('$') !=0]
    adapted_querystring_dict = deepcopy(querystring_dict)
    if len(var_domains) >1:
        for idx, var_domain in enumerate(var_domains):
            var_domain_idx = domain_indeces.index(var_domain)
            if idx == 0:
                
                var_querystring = querystring_dict[var_domain].replace(';', '')
                var_queryslist = var_querystring.split()
                var_list =[int(event[2:])  for event in var_queryslist if event.startswith('$x')]
                max_var = max(var_list)
                
                new_var = max_var +1
            else:
                
                var_querystring2 = querystring_dict[var_domain].replace(';', '')
                var_queryslist2 = var_querystring2.split()
                var_list2 =[int(event[2:])  for event in var_queryslist2 if event.startswith('$x')]
                max_var2 = max(var_list2)
                new_var2 = max_var2 +1
                # querystring = querystring_dict[last_variable_domain2]
                querystring = querystring_dict[var_domain]
                for old_var in range(new_var2):
                    var_shift = old_var + new_var
                    querystring = querystring.replace(f'$x{old_var}', f'$x_{var_shift}')
                querystring = querystring.replace('_', '')
                new_var += new_var2
                adapted_querystring_dict[var_domain] = querystring
    return adapted_querystring_dict

def calc_instance_pairs(instance, dom_query_length, max_dom_query_length):
    """_summary_

    Args:
        instance (_type_): _description_
        dom_query_length (_type_): _description_
        max_dom_query_length (_type_): _description_

    Returns:
        _type_: _description_
    """

    combs = sorted(combinations(instance, dom_query_length[0]))
    rcombs = sorted(combinations(instance, max_dom_query_length-dom_query_length[0]), reverse=True)

    if len(dom_query_length) == 2:
        inst_pairs = tuple((x,y) for x,y in zip(combs, rcombs) if len(set(x)) == len(x) and len(set(y)) == len(y))
        return set(inst_pairs)
    else:
        inst_pairs = []
        for x,y in zip(combs, rcombs):
            if len(set(x)) == len(x):
                for z in calc_instance_pairs(y, dom_query_length[1:], max_dom_query_length-dom_query_length[0]):
                    inst_pairs.append((x,*z))
        return set(inst_pairs)

def to_normalform(querystring):
    """Returns the normalform of a given querystring.

    Args:
        querystring (String)

    Returns:
        querystring in normalform (String)
    """
    if not querystring:
        return querystring


    normal_query = MultidimQuery()
    normal_query.set_query_string(querystring, recalculate_attributes=False)
    normal_query.query_string_to_normalform()
    
    return normal_query._query_string



def non_descriptive_queries_multidim(query= None, querystring= None, parent_dict= None):
    """Given a querystring it generates more general querystring that are not descriptive.

    Args:
        querystring (string): string representation of query

    Returns:
        Set containing non descriptive querystrings
    """
    if query:
        querystring = query._query_string
        variables=query._query_repeated_variables
        if not variables and querystring.count('$') != 0:
            query.set_query_repeated_variables()
            variables=query._query_repeated_variables
        var_domains = []
    else:
        query = MultidimQuery()
        query.set_query_string(querystring)
        variables=query._query_repeated_variables
        var_domains= []

    non_descriptive_set= set()
    if not querystring:
        return non_descriptive_set
    query_liste = query.get_query_list()
    query_length = len(query_liste)

    num_of_vars= len(variables)
    domain_cnt = query_liste[0].count(';')
    gen_event= ';' * domain_cnt
    gen_event_list = [i for i in gen_event]

    typeset= {}
    for domain in range(domain_cnt):
        domain_types=set()
        domain_query= ' '.join([event.split(';')[domain] for event in query_liste])
        for letter in domain_query.split():
            if '$' not in letter:
                domain_types.add(letter)
        if domain_types:
            typeset[domain] = domain_types



    if variables:
        for idx, variable in enumerate(variables):
            #replace variables by new variables if they occur more than 4 times
            variable_count = querystring.count(variable)
            var_pos= querystring.find(variable)-1
            if var_domains:
                var_domain = var_domains[idx]
            else:
                var_domain= querystring[:var_pos].count(';') % domain_cnt
            domain_query_liste = [event.split(';')[var_domain] for event in query_liste]
            counter=2
            if variable_count >=4:
                pos_list = {i for i in range(len(domain_query_liste))  if domain_query_liste[i]== '$'+variable }
                # print(pos_list)
                pos_pairs= set()
                while counter <= variable_count-2:
                    pos_pairs.update(set(combinations(pos_list, counter)))
                    # print(pos_pairs)
                    counter+=1

                for pos_pair in pos_pairs:
                    if var_domain == 0:
                        gen_querystring= " ".join([
                            ''.join(
                                [';'.join(query_liste[i].split(';')[:var_domain])] +
                                [f"$x{num_of_vars};"] +
                                [';'.join(query_liste[i].split(';')[var_domain+1:])])
                            if i in pos_pair else query_liste[i]
                            for i in range(len(query_liste)) ])
                    else:
                        gen_querystring= " ".join([
                            ''.join(
                                [';'.join(query_liste[i].split(';')[:var_domain])] +
                                [f";$x{num_of_vars};"] +
                                [';'.join(query_liste[i].split(';')[var_domain+1:])])
                            if i in pos_pair else query_liste[i]
                            for i in range(len(query_liste)) ])

                    if gen_querystring not in parent_dict:
                        gen_querystring = reposition_vars(gen_querystring)
                        # assert gen_querystring == to_normalform(gen_querystring)
                        # gen_querystring=to_normalform(gen_querystring)

                    if gen_querystring!= querystring:
                        non_descriptive_set.add(gen_querystring)
    if typeset:
        for domain, letters in typeset.items():
            #replace type by new variables if they occur more than once
            for letter in letters:
                domain_query_liste = [event.split(';')[domain] for event in query_liste]
                letter_count = domain_query_liste.count(letter)
                counter= 2
                pos_list = {i for i in range(len(domain_query_liste))  if domain_query_liste[i]== letter }
                # print(pos_list)
                pos_pairs= set()
                while counter <= letter_count:
                    pos_pairs.update(set(combinations(pos_list, counter)))
                    # print(pos_pairs)
                    counter+=1

                    for pos_pair in pos_pairs:
                        if domain == 0:
                            gen_querystring= " ".join([
                            ''.join(
                                [';'.join(query_liste[i].split(';')[:domain])] +
                                [f"$x{num_of_vars};"] +
                                [';'.join(query_liste[i].split(';')[domain+1:])])
                            if i in pos_pair else query_liste[i]
                            for i in range(len(query_liste)) ])
                        else:
                            gen_querystring= " ".join([
                            ''.join(
                                [';'.join(query_liste[i].split(';')[:domain])] +
                                [f";$x{num_of_vars};"] +
                                [';'.join(query_liste[i].split(';')[domain+1:])])
                            if i in pos_pair else query_liste[i]
                            for i in range(len(query_liste)) ])
                        if gen_querystring not in parent_dict:
                            gen_querystring=to_normalform(gen_querystring)

                        if gen_querystring!= querystring:
                            non_descriptive_set.add(gen_querystring)

    for event in query_liste:
        for domain in range(domain_cnt):
            domain_query = " ".join([event.split(';')[domain] for event in query_liste])
            dom_query_list=[]
            for item in domain_query.split():
                cur_event_list= gen_event_list[:domain] + [item] + gen_event_list[domain:]
                cur_event=''.join(cur_event_list)
                dom_query_list.append(cur_event)
            dom_query_string= ' '.join(dom_query_list)
            if dom_query_string:
                if dom_query_string not in parent_dict:
                    dom_query_string = rename_variables(dom_query_string, variables)
                    normal_form = dom_query_string
                    # normal_form = to_normalform(dom_query_string)
                    # assert normal_form == dom_query_string
                else:
                    normal_form= dom_query_string
                if normal_form != querystring:
                    non_descriptive_set.add(normal_form)
    subqueries = combinations(query_liste, query_length-1)
    subquerystrings = { ' '.join(subquery) for subquery in subqueries}
    subsubqueries = []
    for event in query_liste:
        event_set= set()

        for idx, symbol in enumerate(event):
            if symbol == ';' or event[idx-1]!= ';':
                continue
            new_event= event
            while new_event[idx]!= ';':
                new_event= new_event[:idx] + new_event[idx+1:]
            if new_event != gen_event:
                event_set.add(new_event)
        event_set.add(event)
        subsubqueries.append(event_set)
    subsubs= [' '.join(subquery) for subquery in set(product(*subsubqueries))]
    subquerystrings.update(subsubs)
    for gen_querystring in subquerystrings:
        if gen_querystring not in ['', gen_event]:
            if gen_querystring.count('$') >= 1:
                for i in range(len(variables)):
                    var = f'$x{i}'
                    if gen_querystring.count(var) == 1:
                        gen_querystring= gen_querystring.replace( var, '')
                gen_querystring = ' '.join([event for event in gen_querystring.split() if event!= gen_event])
                if not gen_querystring:
                    continue
                gen_querystring = rename_variables(gen_querystring, variables)
                # if gen_querystring[gen_querystring.find('$')+2] != '0':
                gen_querystring = reposition_vars(gen_querystring)
            #normal_form = to_normalform(gen_querystring)
            normal_form = gen_querystring
            # assert normal_form == gen_querystring
            if normal_form!= querystring:
                non_descriptive_set.add(normal_form)
    if '' in non_descriptive_set:
        non_descriptive_set.remove('')
    if gen_event in non_descriptive_set:
        non_descriptive_set.remove(gen_event)
    return non_descriptive_set

def ht_descriptive_queries(query_tree:HyperLinkedTree, matching_queries:set):
    ht_non_descriptive = set()
    for vertex in query_tree.vertices_to_list(frequent_items_only=False):
        querystring = vertex.query_string
        if querystring not in matching_queries:
            continue
        if vertex.parent_vertices:
            parent_vertices = vertex.parent_vertices
        else:
            parent_vertices = query_tree.find_parent_vertices(vertex)
            vertex.parent_vertices = parent_vertices
        parent_querystrings = [parent_vertex.query_string for parent_vertex in parent_vertices]

        ht_non_descriptive.update(parent_querystrings)
    pos_descriptive = matching_queries - ht_non_descriptive
    splitted_event_qs_set = [[event.split(";") for event in qs.split()] for qs in pos_descriptive]
    # descriptive_query_set = set()
    non_descriptive_query_set = set()
    qs_set_pairs = list(combinations(splitted_event_qs_set, 2))
    for cur_tuple in qs_set_pairs:
        curr_qs = cur_tuple[0]
        splitted_query_string = cur_tuple[1]

        curr_qs_is_descriptive = True
        splitted_event_is_descriptive = True
        if len(curr_qs) < len(splitted_query_string):
            if _syntactically_contained(curr_qs, splitted_query_string):
                curr_qs_is_descriptive = False
                childstring = ' '.join([';'.join(event) for event in splitted_query_string])
                new_child = query_tree.find_vertex(childstring)
                if new_child:
                    non_desc_string = ' '.join([';'.join(event) for event in curr_qs])
                    non_desc_vertex = query_tree.find_vertex(non_desc_string)
                    new_child.parent_vertices.add(non_desc_vertex)

        elif len(splitted_query_string) < len(curr_qs):
            if _syntactically_contained(splitted_query_string, curr_qs):
                splitted_event_is_descriptive = False
                childstring = ' '.join([';'.join(event) for event in curr_qs])
                non_desc_string = ' '.join([';'.join(event) for event in splitted_query_string])
                new_child = query_tree.find_vertex(childstring)
                non_desc_vertex = query_tree.find_vertex(non_desc_string)
                new_child.parent_vertices.add(non_desc_vertex)
                
        else:
            if _syntactically_contained(curr_qs, splitted_query_string):
                curr_qs_is_descriptive = False
                childstring = ' '.join([';'.join(event) for event in splitted_query_string])
                new_child = query_tree.find_vertex(childstring)
                if new_child:
                    non_desc_string = ' '.join([';'.join(event) for event in curr_qs])
                    non_desc_vertex = query_tree.find_vertex(non_desc_string)
                    new_child.parent_vertices.add(non_desc_vertex)
                
            if _syntactically_contained(splitted_query_string, curr_qs):
                splitted_event_is_descriptive = False
                childstring = ' '.join([';'.join(event) for event in curr_qs])
                non_desc_string = ' '.join([';'.join(event) for event in splitted_query_string])
                new_child = query_tree.find_vertex(childstring)
                non_desc_vertex = query_tree.find_vertex(non_desc_string)
                new_child.parent_vertices.add(non_desc_vertex)
                
        if not curr_qs_is_descriptive:
            non_descriptive_query_set.add(non_desc_string)
        elif not splitted_event_is_descriptive:
            non_descriptive_query_set.add(non_desc_string)
        
    return pos_descriptive - non_descriptive_query_set, query_tree
    

def _syntactically_contained(qs_array_1:list, qs_array_2:list, assignments:dict|None=None) -> bool:
    """
        Decides whether on of the arrays of arrays of events is contained in the other following a given varibale mapping.
        The relation is symmetric.
        Example:
            _syntactically_contained([['a','b']], [['a','']]) === True
            _syntacitcally_contained([['$x0',''],['$x0','']], [['a',''], ['a','']]) === True
            _syntacitcally_contained([['$x0',''],['$x0','']], [['a',''], ['a','']], {'$x0' : 'a'}) === True
            _syntacitcally_contained([['$x0',''],['$x0','']], [['a',''], ['a','']], {'$x0' : 'b'}) === False

        Args:
            qs_array_1: an array of array containg attributes
                e.g. [event.split(';') for event in query_string.split()]

            qs_array_2: an array of array containg attributes

            assignments [=None]: optional parameter to give an already defined dictionary as varibale mapping.

        Returns:
            bool:
                True, if one qs_array contains the other
                False, else.

        Raises:
            TypeError: if a given assignment dictionary is not of type <dict>.
            TypeError if a given assignment dictionary is not of type <dict>.

        Passes:
            ValueError from _syntactically_contained_event(...)
    """
    if len(qs_array_1) == 0:
        return True
    if len(qs_array_2) == 0:
        return False
    if assignments is None:
        assignments = {}
        assignments_cp = {}
    else:
        if not isinstance(assignments, dict):
            raise TypeError("A given assignment dictionary must be of type <dict>!")
        assignments_cp = deepcopy(assignments)
    event_counter = 0
    for i, ev_array_2 in enumerate(qs_array_2):
        ev_array_1 = qs_array_1[event_counter]
        equals, changed = _syntactically_contained_event(ev_array_1, ev_array_2, assignments_cp)
        if not equals:
            continue
        if not changed:
            event_counter += 1
        else:
            if _syntactically_contained(qs_array_1[event_counter+1:],qs_array_2[i+1:],assignments_cp):
                return True
            assignments_cp = deepcopy(assignments)
        if event_counter == len(qs_array_1):
            return True
    return False

def reposition_vars(gen_querystring):
    gen_querylist= gen_querystring.split()
    counter2 = 0
    seen_vars= set()
    for event in gen_querylist:
        for domain_letter in event.split(';'):
            if '$x' in domain_letter and domain_letter not in seen_vars:
                if gen_querystring.count(domain_letter) > 1:
                    if int(domain_letter[2:]) == counter2:
                        pass
                    else:
                        gen_querystring = gen_querystring.replace(domain_letter, f"$x_{counter2}")
                    seen_vars.add(domain_letter)
                    counter2+=1
    gen_querystring = gen_querystring.replace('_', '')
    return gen_querystring

def _syntactically_contained_event(ev_array_1:list, ev_array_2:list, assignments:dict|None=None) -> tuple:
    """
        Decides whether ev_array_1 is contained in ev_array_2 or not.
        The relation checks only one way.
        Example:
            _syntactically_contained(['a','b'], ['a','']) === False
            _syntactically_contained(['a',''], ['a','b']) === True

        Args:
            ev_array_1: an array with attributes or variables (i.e. representing an event)

            ev_array_2: an array with attributes or variables (i.e. representing an event)

            assignments [=None]: optional parameter to give an already defined dictionary as varibale mapping.

        Returns:
            tuple: (value_1, value_2)
                value_1 = True, if ev_array_2 contains ev_array_1
                value_1 = False, else
                value_2 = True, if the assignment has be changed
                value_2 = False, else

        Raises:
            ValueError: if the dimension of both ev_arrays does not match

        Passes:
            None
    """
    if not len(ev_array_1) == len(ev_array_2):
        raise ValueError("Dimension of events does not match!")
    changed_assignments = False
    for dim, value in enumerate(ev_array_1):
        if value == "":
            continue
        if ev_array_2[dim] == "":
            return False, changed_assignments
        if not value[0] == "$":
            if value == ev_array_2[dim]:
                continue
            return False, changed_assignments
        else:
            if value in assignments:
                if assignments[value] == ev_array_2[dim]:
                    continue
                return False, changed_assignments
            else:
                assignments[value] = ev_array_2[dim]
                changed_assignments = True
    return True, changed_assignments

def rename_variables(gen_querystring, variables):
    total_count= gen_querystring.count('$')
    cur_count = 0
    for i in range(len(variables)-1):
        var = f'$x{i}'
        cur_count += gen_querystring.count(var)
        if cur_count == total_count:
            break
        if gen_querystring.count(var) == 0:
            for j in range(i+1, len(variables)):
                if gen_querystring.count(f'$x{j}') >= 1:
                    old_var = f'$x{j}'
                    break
            gen_querystring= gen_querystring.replace(old_var, var)
    return gen_querystring

def adapt_sample_multidim(sample):
    """For types that occur more than mean+ std times in the sample a counter is added to those
        types to exclude them from mining.
    Args:
        sample: an instance of Sample

    Returns:
        Adapted instance of Sample
    """
    trace_dict = {}
    sample_set = sample._sample
    sample_list = []
    domain_cnt = sample_set[0].split(' ')[0].count(';')

    for trace_id , trace in enumerate(sample_set):
        domain_list=[]
        trace_list = [domain.split(';')[:-1] for domain in trace.split()]

        for i in range(domain_cnt):
            current_domain=[]
            for event in trace_list:
                current_domain.append(event[i])
            domain_list.append(current_domain)
        sample_list.append(domain_list)

    for domain in range(len(sample_list[0])):

        domain_sample_list= []
        for trace_id, trace in enumerate(sample_list):
            domain_sample_list.append(sample_list[trace_id][domain])
        domain_sample_set= [' '.join(trace) for trace in domain_sample_list]
        domain_sample=Sample()
        domain_sample.set_sample(domain_sample_set)
        domain_sample.set_sample_typeset()
        domain_sample.adapt_sample()
        domain_sample_set= domain_sample._sample

        for trace_id, trace in enumerate(sample_set):
            if trace_id in trace_dict:
                current_trace = trace_dict[trace_id]
                trace_dict[trace_id] = ' '.join([';'.join([cur_event] + [new_ev])  for cur_event, new_ev in zip(current_trace.split(), domain_sample_set[trace_id].split())])
            else:
                trace_dict[trace_id] = domain_sample_set[trace_id]
    new_sampleset= []

    for trace_id, trace in trace_dict.items():
        new_sampleset.append(trace)
    sample = Sample()
    sample.set_sample(new_sampleset)
    sample.set_sample_typeset()
    return sample
