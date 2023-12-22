#!/usr/bin/python3
""" Contains the class for handling Samples"""
import logging
from copy import deepcopy
from math import ceil
import numpy as np
from error import EmptySampleError,InvalidTraceError,InvalidQuerySupportError

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

class Sample:
    """
        Base class for representing a Sample.

        Attributes:
            _sample: List of strings. Each string is called a trace. Types
                occurring in a trace must by separated by one whitespace.

            _sample_size: Number of traces within the _sample.

            _sample_typeset: Set of types occurring within the _sample. This
                can be seen as the alphabet for query discovery, since a query
                can not match a sample while containing further types.

            _sample_supported_typeset: Dict mapping support thresholds to a
                corresponding set of types occurring within the _sample and
                satisfying the given support threshold.

            _self._sample_vertical_sequence_database: contains a tracewise
                vertical map from type to occurence in trace
    """
    def __init__(self, given_sample=None) -> None:
        LOGGER.debug('Creating an instance of Sample')
        if not given_sample:             # checks if given_sample == None:
            given_sample = []
        self._sample:list = given_sample
        """List of strings representing traces. Default: []"""
        self._sample_size:int = len(self._sample)
        """Number of traces within _sample. Default: 0"""
        self._sample_typeset:set = set()
        """List of types occuring in _sample. Default: set()"""
        if self._sample_size:           # checks if len(givenSample) != 0
            self.set_sample_typeset()
        self._sample_supported_typeset:dict = {}
        """Dict of types satisfying a gven support. Default: {}"""
        self._sample_vertical_sequence_database:dict = {}
        """Vertical map from type to occurence in trace. Default: {}}"""

    ##################################################

    def sample_stats(self, support:float=1.0, factors:list|None=None) -> dict:
        """
            Collects statistics regarding sample and returns them as a dict.

            Args:
                support: Float between 0 and 1. If greater than zero the
                    function should collect all types which satisfy the given
                    support.

                factors: Expects a list of integers which represents the
                    factors the user is interested in concerning a
                    distribution. Default is None, i.e. no factor distribution
                    gets calculated.

            Returns:
                A dict which contains all collected statistics.

            Raises:
                InvalidQuerySupportError: Support is less than 0 or greater
                    than 1.
        """
        if support < 0.0 or support > 1.0:
            raise InvalidQuerySupportError(f'Support {support} has to be between 0 and 1.')

        self.set_sample_size()
        self.set_sample_typeset()
        min_max_avg_trace = self.get_sample_max_min_avg_trace()
        min_trace = min_max_avg_trace[0]
        max_trace = min_max_avg_trace[1]
        avg_trace_len = min_max_avg_trace[2]
        type_distribution = self._get_type_distribution()

        statistic = {}
        statistic["sample size"] = self._sample_size
        statistic["sample typeset"] = self._sample_typeset
        statistic["sample support"] = support
        if support==0.0:
            statistic["sample supported typeset"] = self._sample_typeset
        else:
            statistic["sample supported typeset"] = self.get_supported_typeset(support)

        statistic["sample #types"] = len(self._sample_typeset)
        statistic["sample min trace"] = min_trace
        statistic["sample min trace len"] = min_trace.count(" ")+1
        statistic["sample max trace"] = max_trace
        statistic["sample max trace len"] = max_trace.count(" ")+1
        statistic["sample average trace len"] = avg_trace_len

        trace_length_distribution = self._get_trace_length_distribution()
        statistic["sample trace length distribution"] = sorted(trace_length_distribution.items(), key=lambda x: x[1])

        statistic["sample type distribution"] = type_distribution
        statistic["sample type distribution ordered"] = sorted(type_distribution.items(), key=lambda x: x[1])
        if factors is not None:
            for factor in factors:
                factor_distribution=self._get_factor_distribution(factor_length=factor)
                statistic[f"sample {str(factor)} factor distribution"] = factor_distribution
                statistic[f"sample {str(factor)} factor distribution ordered"] = sorted(factor_distribution.items(), key=lambda x: x[1])
                statistic[f"sample {str(factor)} #factors"] = len(factor_distribution)
        return statistic

    def _get_supported_typeset(self, support:float) -> set:
        """
            Collects all types which satisfy the given support, i.e. which
            occur in enough traces in the sample.

            Args:
                support: Float between 0 and 1.

            Returns:
                A set consisting all types which satisfy the given support.

            Raises:
                InvalidQuerySupportError: Support is less than 0 or greater
                    than 1.
        """
        if support < 0.0 or support > 1.0:
            raise InvalidQuerySupportError(f'Support {support} has to be between 0 and 1.')
        self.set_sample_size()
        supported_typeset = set()
        typeset = self._sample_typeset
        for elem in typeset:
            supp = 0
            for trace in self._sample:
                if str(trace).count(str(elem))>0:
                    supp = supp+1
            if (supp/self._sample_size) >= support:
                supported_typeset.add(elem)
        return supported_typeset

    def _get_trace_length_distribution(self) -> dict:
        """
            Computes the distribution of trace length in the sample.

            Returns:
                Dict which contains the mapping from traces length to
                corresponding relative frequencies.
        """
        trace_length_distribution = {}
        # count the absolute number of trace length occurrences in the sample
        for trace in self._sample:
            trace_length = trace.count(" ")+1
            if str(trace_length) in trace_length_distribution:
                trace_length_distribution[str(trace_length)] = trace_length_distribution[str(trace_length)] +1
                LOGGER.debug('_get_trace_length_distribution - key contained, new value: %s', str(trace_length_distribution[str(trace_length)] +1))
            else:
                trace_length_distribution[str(trace_length)] = 1
                LOGGER.debug('_get_trace_length_distribution - key not contained, first value: %s', str(trace_length_distribution[str(trace_length)]))

        overall_type_sum = sum(trace_length_distribution.values())
        # compute the relative frequency for each trace length
        for key in trace_length_distribution:
            trace_length_distribution[key] = [trace_length_distribution[key]/overall_type_sum]
        return trace_length_distribution

    def _get_type_distribution(self) -> dict:
        """
            Computes the distribution of types in the sample.

            Assumes that self._sample_typeset is correct regarding
            self._sample.

            Returns:
                Dict which contains the mapping from types to the corresponding
                relative frequencies.
        """
        type_distribution = {}
        # initialization of dict
        for type_block in self._sample_typeset:
            type_distribution[str(type_block)] = 0
        # count the absolute number of occurrences in the sample
        for type_block in self._sample_typeset:
            for trace in self._sample:
                count = trace.count(type_block)
                type_distribution[str(type_block)] = type_distribution[str(type_block)] + count

        overall_type_sum = sum(type_distribution.values())

        # compute the relative frequency for each type
        for key in type_distribution:
            type_distribution[key] = [type_distribution[key]/overall_type_sum]
        return type_distribution

    def _get_factor_distribution(self, factor_length:int=10) -> dict:
        """
            Computes the distribution of factors of given length in the sample.

            Returns:
                Dict which contains the mapping from factors to the
                corresponding relative frequencies.
        """
        factors = self._get_factors(factor_length)
        factor_distribution = {}
        # initialization of dict
        for factor in factors:
            factor_distribution[factor] = 0
        # count the absolute number of occurrences in the sample
        for factor in factors:
            for trace in self._sample:
                count = trace.count(factor)
                factor_distribution[str(factor)] = factor_distribution[str(factor)] + count
        # compute the relative frequency for each factor
        overall_factor_sum = sum(factor_distribution.values())
        for key in factor_distribution:
            factor_distribution[key] = factor_distribution[key]/overall_factor_sum
        return factor_distribution

    def _get_factors(self, factor_length:int) -> set:
        """
            Computes all factors of given length of all traces in the sample.

            Args:
                factor_length: Length of the factors, excluding whitespaces, as
                    integer.

            Returns:
                A set of all factors of the requested length.

            Raises:
                InvalidTraceError: If a trace ends with a whitespace.
        """
        type_starts = True
        factors = set()
        for trace in self._sample:
            for i in range(len(trace)):
                if (i>=1 and trace[i-1] == " ") or i==0:
                    type_starts = True
                else:
                    type_starts = False
                if type_starts is True and i<len(trace)-1:
                    for j in range(i + 1, len(trace)):
                        # we assume that the last position in each trace is not a whitespace
                        if j==len(trace)-1 and trace[j]==" ":
                            raise InvalidTraceError("Trace ends with whitespace.")
                        if j==len(trace)-1 and ((trace[i:j+1].count(" ")+1)==factor_length or ((trace[i:j+1].count(" "))==0 and factor_length==1)):
                            factors.add(trace[i:j+1])
                        if j < len(trace)-1 and trace[j]==" " and ((trace[i:j].count(" ")+1)==factor_length or ((trace[i:j].count(" "))==0 and factor_length==1)):
                            factors.add(trace[i:j])
                        if j < len(trace)-1 and trace[j+1]==" " and ((trace[i:j+1].count(" ")+1)==factor_length or ((trace[i:j+1].count(" "))==0 and factor_length==1)):
                            factors.add(trace[i:j+1])
        return factors

    def _create_vertical_sequence_database(self, trace_index_list:set|None=None) -> None:
        """
            Creating the vertical database of the sample.

            Vertical means the mapping is change from 'trace number -> trace'
            to 'type -> [[pos in 0], [pos in 1], ...]' positions per trace. The
            result is storted in "_sample_vertical_sequence_database".

            Args:
                trace_index_list: [optional] list, which traces of
                    'sequence_database' shall be used to build the vertical
                    database. Enumeration begins with 0 and an 'traces == []'
                    includes all sequences in 'sequence_database'.

            Returns:
                None

            Raises:
                EmptySampleError: If the given sample is empty.
                IndexError: If an index occurs in trace_index_list that is not between 0 and len(sequence_database)-1
        """
        if self._sample == []:
            raise EmptySampleError("Can't create a vertical database for an empty sample!")
        if trace_index_list is None:
            trace_index_list = set(range(0,self._sample_size))
        else:
            trace_index_list = set(trace_index_list)
            for item in trace_index_list:
                if item not in range(0,len(self._sample)):
                    raise IndexError

        vertical_representation = {}

        for trace_index in trace_index_list:
            trace = self._sample[trace_index].split()
            for item_index, item in enumerate(trace):
                if item not in vertical_representation:
                    vertical_representation[item] = {}
                if trace_index in vertical_representation[item]:
                    vertical_representation[item][trace_index].append(item_index)
                else:
                    vertical_representation[item][trace_index] = [item_index]
        self._sample_vertical_sequence_database = vertical_representation

    ##################################################

    def get_sample_min_trace(self) -> str:
        """
            Returns the trace with minimal number of events within the sample.

            Returns:
                A trace with minimal length as a string.

            Raises:
                EmptySampleError: The given sample is empty.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')

        min_trace = self._sample[0]
        for trace in self._sample:
            if trace.count(" ") < min_trace.count(" "):
                min_trace = trace
        return min_trace

    def get_sample_max_trace(self) -> str:
        """
            Returns the trace with maximal number of events within the sample.

            Returns:
                A trace with maximal length as a string.

            Raises:
                EmptySampleError: The given sample is empty.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')
        max_trace = ""
        for trace in self._sample:
            if trace.count(" ") > max_trace.count(" "):
                max_trace = trace
        return max_trace

    def get_sample_max_min_avg_trace(self) -> tuple:
        """
            Returns min and max trace as well as the average trace length.

            Returns:
                A tuple of traces and an integer where the first or second
                trace has minimal or maximal length, whereby the integer stores
                the average trace length within the sample.

            Raises:
                EmptySampleError: The given sample is empty.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')

        trace_length = 0
        trace_count = self._sample_size
        min_trace = self._sample[0]
        max_trace = ""
        for trace in self._sample:
            trace_length = trace_length+trace.count(" ")+1
            if trace.count(" ") >= max_trace.count(" "):
                max_trace = trace
            if trace.count(" ") < min_trace.count(" "):
                min_trace = trace
        avg_trace_length = trace_length/trace_count
        return tuple((min_trace,max_trace,avg_trace_length))

    def get_l_w_tuples(self, query_string_length_interval:tuple|None=None) -> list[tuple]:
        """
            Returns a list of all possible l-w-tuples for the current sample,
            with equal w-tuples for each position.

            Args:
                query_string_length_interval: Optional argument to fix a lower
                    and an upper bound for the query string length.

            Returns:
                A list of tuples. The first element of each tuple is an integer
                representing a possible query string length l, the second entry
                is a list of integer tuples, representing the local window
                sizes dependent on l.

            Raises:
                EmptySampleError: The given sample is empty.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')
        l_w_list = []
        min_trace_len = self.get_sample_min_trace().count(" ")+1
        for query_string_length in range(1,min_trace_len+1):
            if query_string_length_interval is not None and query_string_length<query_string_length_interval[0]:
                continue
            if query_string_length_interval is not None and query_string_length>query_string_length_interval[1]:
                break
            if query_string_length == 1:
                l_w_list.append(tuple((1,[tuple((-1,-1)),tuple((-1,-1))])))
                continue
            for lower_bound in range(0,min_trace_len):
                for upper_bound in range(lower_bound,min_trace_len):
                    sum_of_upper_bound = query_string_length*upper_bound
                    # if we found a valid window size tuple:
                    if lower_bound<=upper_bound and sum_of_upper_bound+query_string_length<=min_trace_len:
                        w_list = []
                        # add placeholder window size tuple at the beginning
                        w_list.append(tuple((-1,-1)))
                        # add window size tuple query_string_length-1 times
                        for _ in range(1,query_string_length):
                            w_list.append(tuple((lower_bound,upper_bound)))
                        # add placeholder window size tuple at eh end:
                        w_list.append(tuple((-1,-1)))
                        l_w_list.append(tuple((query_string_length,deepcopy(w_list))))
        return l_w_list

    def get_supported_typeset(self, support:float) -> set:
        """
            Collects all types which satisfy the given support, i.e. which
            occur in enough traces in the sample.

            For different supports the supported typesets are stored in a
            dictionary and won't be recalculated untill the sample changes.

            Args:
                support: Float between 0 and 1.

            Returns:
                set: supported typeset

            Raises:
                InvalidQuerySupportError: Support is less than 0 or greater than 1.
        """
        sample_sized_support = ceil(self._sample_size * support)
        if sample_sized_support not in self._sample_supported_typeset:
            if support < 0.0 or support > 1.0:
                raise InvalidQuerySupportError(f'Support {support} has to be between 0 and 1.')
            if len(self._sample_vertical_sequence_database)==0:
                self._create_vertical_sequence_database()

            vsdb = self._sample_vertical_sequence_database
            supported_type_set = {symbol for symbol in vsdb if len(vsdb[symbol]) >= sample_sized_support}
            self._sample_supported_typeset[sample_sized_support] =  supported_type_set

        return self._sample_supported_typeset[sample_sized_support]

    def get_vertical_sequence_database(self) -> dict:
        """
            Returns the vertical sequence database of the sample.

            If it is not computed yet, it will be calculated.

            Args:
                None

            Returns:
                Dict: "type" -> [positions per traces]

            Raises:
                None
        """
        if len(self._sample_vertical_sequence_database)==0:
            self._create_vertical_sequence_database()
        return self._sample_vertical_sequence_database

    def set_sample(self, sample:list) -> None:
        """
            Sets _sample to sample.

            Args:
                sample: List of stringe which represent traces.
        """
        self._sample = sample
        if isinstance(sample, list):
            self.set_sample_size()
            if self._sample_size > 0:
                self.set_sample_typeset()
            else:
                self._sample_typeset = set()
            self._sample_supported_typeset = {}
            self._sample_vertical_sequence_database = {}

    def set_sample_size(self) -> None:
        """
            Determines and sets _sample_size for current _sample.
        """
        self._sample_size = len(self._sample)

    def set_sample_typeset(self) -> None:
        """
            Determines and sets the set of types which occur in the _sample.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')

        self._sample_typeset = set()
        for trace in self._sample:
            trace_set = set(trace.split())
            for current_type in trace_set:
                self._sample_typeset.add(current_type)

    ##################################################
    def adapt_sample(self) -> None:
        """
            For types that occur more than mean+ std times in the sample a
            counter is added to those types to exclude them from mining.
        """
        sample_set = self._sample
        type_distribution = self._get_type_distribution()
        type_values = list(type_distribution.values())
        stand_dev = np.std(type_values)
        mean = 1./len(type_distribution)
        threshold = mean + stand_dev
        typeset = self._sample_typeset

        type_count= {letter: 0 for letter in typeset}
        new_sample_set = []
        for trace in sample_set:
            new_trace= []
            for event in trace.split():
                if type_distribution[event] > threshold:
                    counter= type_count[event]
                    new_trace.append(event +',' + str(counter))
                    type_count[event] = counter +1
                else:
                    new_trace.append(event)


            new_sample_set.append(' '.join(new_trace))
        self.set_sample(new_sample_set)
