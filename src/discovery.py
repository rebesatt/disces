#!/usr/bin/python3
"""Contains shared functionalities for different discovery algorithms"""
import logging
from typing import Iterable
from query import Query
from sample import Sample
from error import ShinoharaInvalidPositionError

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

def _find_next_position_in_query_string(query_string:str, position_count:int) -> int:
    """
        Finds the next position for discovery algorithms in the query string.

        Note the difference between the real position of a variable in the
        query string and the position_count: E.g. in a query string "$A $B $C"
        the position of variable $C is requested. The position_count is 3 while
        pos should be 6.

        Args:
            query_string: String which contains the query string as described
                in class Query.

            position_count: Integer which represents the number of the variable
                we want to check next within Shinoharas Algorithm.

        Returns:
            The next position for Shinoharas Algorithm as an integer.

        Raises:
            ShinoharaInvalidPositionError if the position_count does not fit
            the length of the query string.

    """
    if position_count <= 0 or position_count > query_string.count(" ")+1:
        raise ShinoharaInvalidPositionError(f"position_count ({position_count}) is less or greater than the number of events in the given query_string!")
    count=1
    pos=0
    while pos < len(query_string):
        if count == position_count:
            break
        if query_string[pos]==" ":
            count = count + 1
            pos = pos+1
        else:
            pos = pos+1
    return pos

def _find_next_position_in_query_string_multidim(query_string:str, position_count:int) -> int:
    """
        Finds the next position for discovery algorithms in the query string.

        This function is similar to _find_next_position_in_query_string() but
        adapted for multidimensional queries.

        Note the difference between the real position of a variable in the
        query string and the position_count: E.g. in a query string
            "$A;$B; $C;$D; $E;$F"
        the position of variable $C is requested. Then the position_count is 3
        while pos should be 7.

        Args:
            query_string: String which contains the multidimensional query
                string as described in class MultidimQuery.

            position_count: Integer which represents the number of the variable
                we want to check next within Shinoharas Algorithm.

        Returns:
            The next position for Shinoharas Algorithm as an integer.

        Raises:
            ShinoharaInvalidPositionError if the position_count does not fit
            the length of the query string.
    """
    if position_count <= 0 or position_count > query_string.count(" ")+query_string.count(";")+1:
        raise ShinoharaInvalidPositionError(f"position_count ({position_count}) is less or greater than the number of events in the given query_string!")
    count=1
    pos=0
    while pos < len(query_string):
        if count == position_count:
            break
        if query_string[pos]==";":
            count = count + 1
            if pos < len(query_string)-1 and query_string[pos+1]==" ":
                pos = pos+2
            else:
                pos = pos+1
        else:
            pos = pos+1
    return pos

def _extract_var_pre_suf(query_string:str, pos:int) -> tuple:
    """
        Extracts the variable and its pre- and suffix from query string.

        Args:
            query_string: String which contains the query string as described
                in class Query.

            pos: An integer which represents the position of the beginning of
                the variable we want to extract from the query string.

        Returns:
            A tuple of strings consisting of the variable, the variable and its
                suffix, the prefix and the suffix.
    """
    if pos < 0 or pos >= len(query_string):
        raise ShinoharaInvalidPositionError(f"position_count ({pos}) is less or greater than the number of events in the given query_string!")
    if query_string[pos] != '$':
        raise ShinoharaInvalidPositionError(f"No variable starts at position_count ({pos})!")
    current_variable_and_suffix = query_string[pos:]
    current_variable = current_variable_and_suffix.split(' ')[0]
    if pos == 0:
        prefix = ""
    else:
        prefix = query_string[0:pos-1]
    suffix = query_string[len(prefix)+len(current_variable)+1:]
    return (current_variable, current_variable_and_suffix, prefix, suffix)

def _extract_var_pre_suf_multidim(query_string:str, pos:int) -> tuple:
    """
        Extracts the variable and its pre- and suffix from query string.

        This function is similar to _extract_var_pre_suf() but adapted for
        multidimensional queries.

        Args:
            query_string: String which contains the query string as described
                in class Query.

            pos: An integer which represents the position of the beginning of
                the variable we want to extract from the query string.

        Returns:
            A tuple of strings consisting of the variable, the variable and its
                suffix, the prefix and the suffix.
    """
    if pos < 0 or pos >= len(query_string):
        raise ShinoharaInvalidPositionError(f"pos ({pos}) is less or greater than the number of events in the given query_string!")
    if query_string[pos] != '$':
        raise ShinoharaInvalidPositionError(f"No variable starts at pos ({pos})!")
    current_variable_and_suffix = query_string[pos:]
    current_variable = current_variable_and_suffix.split(';')[0] + ";"
    if pos == 0:
        prefix = ""
    else:
        prefix = query_string[0:pos]
    suffix = query_string[len(prefix)+len(current_variable):]
    return (current_variable, current_variable_and_suffix, prefix, suffix)

def _find_attribute_index(query_string:str, position_count:int) -> int:
    """
        Calculates the attribute index at the given position pos.

        Assumes that all events share the same dimension.

        Args:
            query_string: String which contains the query string as described
                in class Query.

            position_count: Integer which represents the relative starting
                position of a variable or type.

        Returns:
            The attribute index of the corresponding variable or type at pos as
            an integer.
    """
    if position_count <= 0 or position_count > query_string.count(" ")+query_string.count(";")+1:
        raise ShinoharaInvalidPositionError(f"pos ({position_count}) is less or greater than the number of events in the given query_string!")
    dimension = query_string.split(' ')[0].count(';')
    if dimension == 0:
        dimension = 1
    attribute_index = (position_count%dimension)-1
    if attribute_index == -1:
        attribute_index = dimension-1
    assert attribute_index > -1
    return attribute_index

def combine_all(iteratable1:list, iteratable2:list) -> Iterable:
    """
        Merges 2 given iterables (list, array, set) in any combination
        preserving the existent order.

        Example combine_all([0,1],[a,b]) ==
        [[0,1,a,b],[0,a,1,b],[0,a,b,1],[a,0,1,b],[a,0,b,1],[a,b,0,1]]

        Args:
            xs, xy: 2 sets which should be merged in any combination.

        Returns:
            An array containing all combinations of xs, xy

        Raises:
            None
    """
    if iteratable1 == []:
        return [iteratable2]
    if iteratable2 == []:
        return [iteratable1]
    xs_first, *xs_tail = iteratable1
    ys_first, *ys_tail = iteratable2
    return [ [xs_first] + item for item in combine_all(xs_tail, iteratable2) ] + [ [ys_first] + item for item in combine_all(ys_tail, iteratable1) ]

def merge_event_arrays(event1:list, event2:list) -> list|None:
    """
        Merges to events, if no dimension has entries in both events.
        The result is an array that contains all values <= 2*dim(event).

        Example:
        merge_event_arrays(["", "x", "", ""], ["","", "y", "z"]) == ["","x","y", "z"] (e.g. ";x;;;" + ";;y;z;" == ";x;y;z;")

        Args:
            event1, event2: of type <list> and is an event splitted by ' '.

        Returns:
            An array contaning the merged event.
            None, if there is at least one dimension that contains values in both event1 and event2

        Raises
            ValueError, if not both events are of type <list>!.
            IndexError, if both events have different dimensions.
    """
    if not isinstance(event1, type(event2)):
        raise ValueError("Both events have to be of type <list>!")
    if not isinstance(event1, list):
        raise ValueError("Both events have to be of type <list>!")
    if not len(event1) == len(event2):
        raise IndexError("Both events have to have the same length!")
    merged_event = []
    for dim, value in enumerate(event1):
        if value == '':
            merged_event.append(event2[dim])
        elif event2[dim] == '':
            merged_event.append(value)
        else:
            return None
    return merged_event

def matching_smarter(querystring:str, sample:Sample, dict_iter:dict, patternset:set, supp:float, parentstring:str):
    """
        Matches a query against all traces in the sample

        Args:
            querystring (String): an instance of Query
            supp (float): between 0 and 1 which describes the requested support.
                Default: 1
            dict_iter (dictionary): nested dictionary for each query and trace the last matching position is value. Default None
            patternset (set): set of types occurring twice in at least one trace. Default None
            parentstring (str): String by which the current querystring was generated.


        Returns:
            (Result, trace_matches, trace_index_list)
            Trace dictionary containing trace index as keys and a dictionary of
            groups with the matched string and span as values.

        Raises:
            ValueError: Grandparent is needed but not available
    """
    trace_matches={}
    sample_size = len(sample._sample)

    if querystring.count('$x') == 0:
        return_match_result = True
        num_trace_match = sample_size
        for trace_idx, trace in enumerate(sample._sample):
            idx, dict_iter = smart_trace_match(querystring, trace, trace_idx, dict_iter)
            if trace_idx not in trace_matches:
                trace_matches[trace_idx]= {}
            trace_matches[trace_idx]= idx
            if idx == -1:
                num_trace_match -= 1
            if num_trace_match/sample_size < supp:
                return_match_result = False
                break

        if return_match_result:
            for trace_index, value in trace_matches.items():
                if querystring not in dict_iter:
                    dict_iter[querystring]= {}
                dict_iter[querystring][trace_index]= value
        return return_match_result, trace_matches, list(trace_matches.keys())

    if parentstring.count('$') == 0:
        if not parentstring:
            parent_traces = list(range(sample_size))
        else:
            parent_traces = list(dict_iter[parentstring].keys())
        trace_list= parent_traces + list(range(parent_traces[-1]+1,sample_size))
        num_trace_match = len(trace_list)
        return_match_result = True
        for trace in trace_list:
            for letter in patternset:
                letter_querystring = querystring.replace('$x0', letter)
                if letter_querystring in dict_iter:
                    if trace in dict_iter[letter_querystring]:
                        if dict_iter[letter_querystring][trace] !=-1:
                            if trace not in trace_matches:
                                trace_matches[trace]= {}
                            trace_matches[trace][(letter,)]= dict_iter[letter_querystring][trace]
                    else:
                        idx, dict_iter = smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                        if idx != -1:
                            if trace not in trace_matches:
                                trace_matches[trace]= {}
                            trace_matches[trace][(letter,)]= idx
                else:
                    idx, dict_iter = smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                    if idx != -1:
                        if trace not in trace_matches:
                            trace_matches[trace]= {}
                        trace_matches[trace][(letter,)]= idx
            if num_trace_match/sample_size < supp:
                return_match_result = False
                break

            if trace not in trace_matches:
                num_trace_match-=1
        if return_match_result:
            for trace_index, value in trace_matches.items():
                if querystring not in dict_iter:
                    dict_iter[querystring]= {}
                dict_iter[querystring][trace_index]= value
        return return_match_result, trace_matches, list(trace_matches.keys())

    else:
        parent = Query()
        parent.set_query_string(parentstring)
        parent_variables=sorted(list(parent._query_repeated_variables))
        if parentstring in dict_iter:
            parent_traces= list(dict_iter[parentstring].keys())
        else:
            LOGGER.info(querystring)
            LOGGER.info(parentstring)
            LOGGER.info(dict_iter)
            raise ValueError("Yet grand parent is needed!")
            
        trace_list= parent_traces
        num_trace_match = len(trace_list)
        return_match_result = True
        for trace in trace_list:
            group_list= list(dict_iter[parentstring][trace].keys())
            for group in group_list:
                letter_querystring=querystring
                assert len(group) == len(parent_variables)
                for val, letter in enumerate(group):
                    letter_querystring = letter_querystring.replace(f'${parent_variables[val]}', letter)
                if letter_querystring.count('$')>0:
                    for letter in patternset:
                        letter_querystring2 = letter_querystring.replace(f'$x{len(group)}', letter)
                        if letter_querystring2 in dict_iter:
                            if trace in dict_iter[letter_querystring2]:
                                if dict_iter[letter_querystring2][trace] != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group + (letter,)]= dict_iter[letter_querystring2][trace]

                            else:
                                idx, dict_iter = smart_trace_match(letter_querystring2, sample._sample[trace], trace, dict_iter)
                                if idx != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group + (letter,)]= idx

                        else:
                            idx, dict_iter = smart_trace_match(letter_querystring2, sample._sample[trace], trace, dict_iter)
                            if idx != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace]= {}
                                trace_matches[trace][group + (letter,)]= idx

                else:
                    if letter_querystring in dict_iter:
                        if trace in dict_iter[letter_querystring]:
                            if dict_iter[letter_querystring][trace] != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace]= {}
                                trace_matches[trace][group]= dict_iter[letter_querystring][trace]
                        else:
                            idx, dict_iter = smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                            if idx != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace]= {}
                                trace_matches[trace][group]= idx
                    else:
                        idx, dict_iter = smart_trace_match(letter_querystring, sample._sample[trace], trace, dict_iter)
                        if idx != -1:
                            if trace not in trace_matches:
                                trace_matches[trace]= {}
                            trace_matches[trace][group]= idx

            if trace not in trace_matches:
                num_trace_match -= 1
            if num_trace_match/sample_size < supp:
                return_match_result = False
                break
        if return_match_result:
            for trace_index, value in trace_matches.items():
                if querystring not in dict_iter:
                    dict_iter[querystring]= {}
                dict_iter[querystring][trace_index]= value
        return return_match_result, trace_matches, list(trace_matches.keys())

def smart_trace_match(querystring:str, trace:str, trace_idx:int, dict_iter:dict) -> tuple:
    """
        Given a trace and a querystring the matching position is calculated and
        in case of a match the dict_iter is updated.

        Args:
            querystring (String): querystring for query

            trace (String): trace from the sample

            trace_idx (int): index of the given trace

            dict_iter (dictionary): nested dictionary for each query and trace
                the last matching position is value.

        Returns:
            [tuple]: containing last matching position and updated dict_iter

        Raises:
            None
    """
    parentstring = ' '.join(querystring.split()[:-1])
    if querystring not in dict_iter:
        dict_iter[querystring]= {}
    if not querystring:
        dict_iter[querystring][trace_idx]=0
        return 0, dict_iter
    if parentstring not in dict_iter:
        idx, dict_iter= smart_trace_match(parentstring, trace, trace_idx, dict_iter)
    if trace_idx in dict_iter[parentstring]:
        parent_end_pos = dict_iter[parentstring][trace_idx]
    else:
        idx, dict_iter= smart_trace_match(parentstring, trace, trace_idx, dict_iter)
        parent_end_pos = dict_iter[parentstring][trace_idx]
    if not parentstring:
        try:
            end_pos = trace.split().index(querystring)
        except ValueError:
            end_pos = -1
        dict_iter[querystring][trace_idx]= end_pos
        return end_pos, dict_iter
    if parent_end_pos !=-1:
        trace_list = trace.split()[parent_end_pos+1:]
        try:
            idx = trace_list.index(querystring.split()[-1])
        except ValueError:
            idx = -1
        if idx != -1:
            if trace_idx not in dict_iter[querystring]:
                dict_iter[querystring][trace_idx]= {}
            end_pos = dict_iter[parentstring][trace_idx] + idx +1
            dict_iter[querystring][trace_idx]= end_pos
            return end_pos, dict_iter
        else:
            dict_iter[querystring][trace_idx]=-1
            return -1, dict_iter
    else:
        dict_iter[querystring][trace_idx]=-1
        return -1, dict_iter
