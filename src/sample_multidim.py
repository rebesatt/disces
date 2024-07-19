#!/usr/bin/python3
""" Contains the class for handling Samples"""
import logging
from copy import deepcopy
from math import ceil
from sample import Sample
from error import EmptySampleError,InvalidTraceError,InvalidQuerySupportError,InvalidSampleDimensionError

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

class MultidimSample(Sample):
    """
        Base class for representing a Sample.

        Attributes:
            _sample: List of strings. Each string is called a trace and
            consists of events, separated by a whitespace. Events may be
            multidimensional and attributes (i.e. the single types of the
            event) are separated by semicolons.

            _sample_size: Number of traces within the _sample.

            _sample_event_dimension: Integer which represents the max. number
            of attributes per event.

            _sample_typeset: Set of types occurring within the _sample. This
            can be seen as the alphabet for query discovery, since a query can
            not match a sample while containing further types.

            _sample_att_typesets: Mapping of attributes to typesets, stored as
            a dict, i.e. the set at index 0 contains all types occuring as a
            first type / attribute value of some event.

            _sample_supported_typeset: Dict mapping support thresholds to a
            corresponding set of types occurring within the _sample satisfying
            the given support threshold.

            _sample_att_supported_typeset: Dict mapping attributes to typesets,
            s.t. each type in a set satisfies the given support for the corres-
            ponding attribute.

            _sample_vertical_sequence_database: contains a tracewise
            vertical map from type to occurence in trace

            _dim_sample_dict: Dictionary with dimensions as key and corres-
            ponding one-dim Sample as value.
    """
    def __init__(self, given_sample=None, uniform_dimension=False) -> None:
        LOGGER.debug('Creating an instance of Sample')
        ### private
        if not given_sample:             # checks if given_sample == None:
            given_sample = []
        self._sample:list = given_sample
        """List of strings representing traces. Default: []"""
        self._sample_size:int = len(self._sample)
        """Number of traces within _sample. Default: 0"""
        self._sample_event_dimension:int = 1
        """Max. number of attributes in an event. Default: 1"""
        self._sample_supported_typeset:dict = {}
        """Dict mapping support thresholds to sets ot types satisfying the corresponding support. Default: {}"""
        self._sample_att_supported_typesets:dict = {}
        """Dict mapping attributes to set of types satisfying a given support. Default: {}"""
        self._dim_sample_dict:dict = {}
        """Dictionary with dimensions as key and corresponding one-dim Sample as value."""
        self._sample_typeset = None
        self._sample_att_typesets = {}
        self._sample_vertical_sequence_database = None
        self._sample_att_vertical_sequence_database = None

        ### public
        self.sample_typeset = None
        """List of types occuring in _sample. Default: None"""
        self.sample_att_typesets = {}
        """List of types occuring in _sample. Default: None"""
        self.sample_vertical_sequence_database = None
        """Vertical map from type to occurence in trace. Default: None"""
        self.sample_att_vertical_sequence_database = None
        

        self.sample_type_pattern_disjointness_list = None
        self.sample_type_pattern_disjointness_score = None

        ### routines
        if self._sample_size:           # checks if len(given_sample) != 0
            self.set_sample_event_dimension(uniform_dimension=uniform_dimension)

    ##################################################

    def sample_stats(self, support:float=1.0, factors:list|None=None, event_factors:bool=True) -> dict:
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

                event_factors: Boolean which indicates whether the factor
                    length refers to the number of whole events or single
                    attributes. Default is True, i.e. for a given factor length
                    i each factor is made of i consecutive events.

            Returns:
                A dict which contains all collected statistics.

            Raises:
                InvalidQuerySupportError: Support is less than 0 or greater than 1.
        """
        if support < 0.0 or support > 1.0:
            raise InvalidQuerySupportError(f'Support {support} has to be between 0 and 1.')

        self.set_sample_size()
        self.calc_sample_typeset()
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
            statistic["sample supported typeset"] = self.get_sample_supported_typeset(support)

        assert self._sample_typeset is not None
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
                factor_distribution=self._get_factor_distribution(factor_length=factor,event_factors=event_factors)
                statistic[f"sample {str(factor)} factor distribution"] = factor_distribution
                statistic[f"sample {str(factor)} factor distribution ordered"] = sorted(factor_distribution.items(), key=lambda x: x[1])
                statistic[f"sample {str(factor)} #factors"] = len(factor_distribution)
        return statistic

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

    def _get_type_distribution(self, attribute:int=-1) -> dict:
        """
            Computes the distribution of types in the sample.

            Assumes that self._sample_typeset is correct regarding
            self._sample.

            Args:
                attribute: Integer which determines the attribute for which the
                    type distribition should be calculated. Use attribute = 0
                    to access the first attribute. Default is -1, implying the
                    type distributions are calculated across all attributes.

            Returns:
                Dict which contains the mapping from types to the corresponding
                relative frequencies.
        """
        if attribute == -1:
            self.calc_sample_typeset()
            return Sample._get_type_distribution(self)

        type_distribution = {}
        # initialization of dict
        for type_block in self.get_sample_typeset(attribute):
            type_distribution[str(type_block)] = 0
        # count the absolute number of occurrences in the sample
        # at the requested attribute
        for type_block in self._sample_att_typesets[attribute]:
            for trace in self._sample:
                count = 0
                event_list = trace.split(' ')
                for current_event in event_list:
                    # Extract the requested attribute
                    type_list = current_event.split(';')
                    if len(type_list)>attribute and str(type_list[attribute])==type_block:
                        count = count + 1
                type_distribution[str(type_block)] = type_distribution[str(type_block)] + count

        overall_type_sum = sum(type_distribution.values())

        # compute the relative frequency for each type
        for key in type_distribution:
            type_distribution[key] = [type_distribution[key]/overall_type_sum]
        return type_distribution

    def _get_factor_distribution(self, factor_length:int=10, event_factors:bool=True) -> dict:
        """
            Computes the distribution of factors of given length in the sample.

            Args:
                factor_length: Length of the factors, excluding whitespaces, as
                    integer.

                event_factors: Boolean which indicates whether the factor
                    length refers to the number of whole events or single
                    attributes. Default is True, i.e. for a given factor length
                    i each factor is made of i consecutive events.

            Returns:
                Dict which contains the mapping from factors to the
                corresponding relative frequencies.
        """
        factors = self._get_factors(factor_length,event_factors)
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

    def _get_factors(self, factor_length:int, event_factors:bool=True) -> set:
        """
            Computes all factors of given length of all traces in the sample.

            Args:
                factor_length: Length of the factors, excluding whitespaces, as
                integer.

                event_factors: Boolean which indicates whether the factor
                    length refers to the number of whole events or single
                    attributes. Default is True, i.e. for a given factor length
                    i each factor is made of i consecutive events.

            Returns:
                A set of all factors of the requested length.

            Raises:
                InvalidTraceError: If a trace ends with a whitespace.
        """
        if len(self._sample)==0:
            return set()
        if event_factors is True:
            return Sample._get_factors(self,factor_length=factor_length)

        factors = set()
        for trace in self._sample:
            for i in range(len(trace)):
                type_starts = False
                if (i>=1 and (trace[i-1] == " " or trace[i-1] == ";")) or i==0:
                    type_starts = True
                if type_starts is True and i<len(trace)-1:
                    for j in range(i + 1, len(trace)):
                        factor=""
                        # we assume that the last position in each trace is not a whitespace
                        if j==len(trace)-1 and trace[j]==" ":
                            raise InvalidTraceError("Trace ends with whitespace.")
                        if j==len(trace)-1 and ((trace[i:j+1].count(";"))==factor_length or ((trace[i:j+1].count(";"))==0 and factor_length==1)):
                            factor = trace[i:j+1]
                        if j < len(trace)-1 and trace[j]==";" and ((trace[i:j].count(";")+1)==factor_length or ((trace[i:j].count(";"))==0 and factor_length==1)):
                            factor = trace[i:j+1]
                        if len(factor)>0:
                            if factor[0]==" ":
                                factor=factor[1:]
                            factors.add(factor)
                            break
        return factors

    def calc_vertical_sequence_databases(self, trace_index_list:set|None=None) -> None:
        """
            Creating the vertical database of the sample.

            Vertical means the mapping is change from '{trace number -> trace}'
            to '{att -> {type -> {trace -> [pos 0, ...]}} }' positions per
            trace. The result is storted in
            "sample_att_vertical_sequence_database".

            Meanwhile a second mapping with '{type -> {trace -> {pos -> att}}}'
            is created. The result is stored in
            "sample_vertical_sequence_database".

            Args:
                trace_index_list [= None]: optional list, which traces of
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
        att_vertical_representation = {att : {} for att in range(0,self._sample_event_dimension)}

        for trace_index in trace_index_list:
            events = self._sample[trace_index].split()
            for event_index, event in enumerate(events):
                values = event.split(";")
                for attribute, value in enumerate(values):
                    if value == "":
                        continue
                    if value not in att_vertical_representation[attribute]:
                        att_vertical_representation[attribute][value] = {}
                        if value not in vertical_representation:
                            vertical_representation[value] = {}
                    if trace_index not in att_vertical_representation[attribute][value]:
                        att_vertical_representation[attribute][value][trace_index] = [event_index]
                        if trace_index not in vertical_representation[value]:
                            vertical_representation[value][trace_index] = {}
                    else:
                        att_vertical_representation[attribute][value][trace_index].append(event_index)

                    if event_index not in vertical_representation[value][trace_index]:
                        vertical_representation[value][trace_index][event_index] = [attribute]
                    else:
                        vertical_representation[value][trace_index][event_index].append(attribute)

        self._sample_vertical_sequence_database = vertical_representation
        self._sample_att_vertical_sequence_database = att_vertical_representation

    def calc_sample_typeset(self, attribute:int=-1, calculate_all:bool=False) -> None:
        """
            Determines and sets _sample_typeset and _sample_att_typesets.

            Args:
                attribute [=-1]: Optional if only a specific domain shall be
                    returned. For "-1" the whole typeset is returned.

            Returns:
                None

            Raises:
                IndexError: if attribute is less than 0 or greater than or
                    equals the event dimension, or is not -1.
        """
        if calculate_all:
            if self._sample_att_vertical_sequence_database is None:
                self.calc_vertical_sequence_databases()
            att_vsdb = self._sample_att_vertical_sequence_database
            assert att_vsdb is not None

            for dim in range(0, self._sample_event_dimension):
                if dim not in self._sample_att_typesets:
                    self._sample_att_typesets[dim] = set(att_vsdb[dim].keys())

        if attribute == -1:
            if self._sample_vertical_sequence_database is None:
                self.calc_vertical_sequence_databases()

            vsdb = self._sample_vertical_sequence_database
            assert vsdb is not None
            self._sample_typeset = set(vsdb.keys())

        elif attribute in range(0, self._sample_event_dimension):
            if self._sample_att_vertical_sequence_database is None:
                self.calc_vertical_sequence_databases()
            if attribute not in self._sample_att_typesets:
                att_vsdb = self._sample_att_vertical_sequence_database
                assert att_vsdb is not None
                self._sample_att_typesets[attribute] = set(att_vsdb[attribute].keys())
        else:
            raise IndexError("Attribute must be an interger from '-1' and 'event dimension -1'")

    #TODO (all): should calc-functions have a return value?
    def calc_sample_supported_typeset(self, support:float, attribute:int=-1) -> set:
        """
            Collects all types which satisfy the given support (for the given
            attribute).

            Args:
                support: Float between 0 and 1.

                attribute: Integer which determines the attribute for which the
                    supported types should get collected. Use attribute = 0 to
                    access the first attribute. Default is -1, implying types
                    are collected across all attributes.

            Returns:
                A set consisting all types which satisfy the given support.

            Raises:
                InvalidQuerySupportError: Support is less than 0 or greater
                    than 1.
                IndexError: If attribute is less than 0 or greater than or equals
                    the event dimension, or is not -1.
        """
        if support < 0.0 or support > 1.0:
            raise InvalidQuerySupportError(f'Support {support} has to be between 0 and 1.')
        if attribute > self._sample_event_dimension:
            raise InvalidSampleDimensionError(f'Attribute {attribute} has to be smaller than or equal to the sample event dimension {self._sample_event_dimension}')
        sample_sized_support = ceil(self._sample_size * support)

        if attribute == -1:
            if sample_sized_support not in self._sample_supported_typeset:
                if self._sample_vertical_sequence_database is None:
                    self.calc_vertical_sequence_databases()
                vsdb = self._sample_vertical_sequence_database
                assert vsdb is not None
                supported_type_set = {symbol for symbol in vsdb if len(vsdb[symbol]) >= sample_sized_support}
                self._sample_supported_typeset[sample_sized_support] =  supported_type_set
            return self._sample_supported_typeset[sample_sized_support]
        else:
            if attribute not in range(0, self._sample_event_dimension):
                raise IndexError
            if sample_sized_support not in self._sample_att_supported_typesets:
                self._sample_att_supported_typesets[sample_sized_support] = {}
            if attribute not in self._sample_att_supported_typesets[sample_sized_support]:
                if self._sample_att_vertical_sequence_database is None:
                    self.calc_vertical_sequence_databases()
                assert self._sample_att_vertical_sequence_database is not None
                vsdb = self._sample_att_vertical_sequence_database[attribute]
                supported_type_set = {symbol for symbol in vsdb if len(vsdb[symbol]) >= sample_sized_support}
                self._sample_att_supported_typesets[sample_sized_support][attribute] =  supported_type_set
            return self._sample_att_supported_typesets[sample_sized_support][attribute]

    def calc_dim_sample_dict(self) -> None:
        """
            Calculates the one-dim Sample for each dimension 
            and sets the attribute self.dim_sample_dict.
        """
        sample_list = self._sample
        dim_count = self._sample_event_dimension
        gen_event= ';' * dim_count
        gen_event_list = [i for i in gen_event]
        dim_samples_list = []
        dim_sample_dict = {}
        for trace_id , trace in enumerate(sample_list):
            domain_list=[]
            trace_list = [domain.split(';')[:-1] for domain in trace.split()]

            for i in range(dim_count):
                current_domain=[]
                for event in trace_list:
                    cur_event_list= gen_event_list[:i] + [event[i]] + gen_event_list[i:]
                    cur_event=''.join(cur_event_list)
                    # current_domain.append(event[i] + ';')
                    current_domain.append(cur_event)
                domain_list.append(current_domain)
            dim_samples_list.append(domain_list)

        for dim in range(dim_count):
            dim_sample_list = []
            for trace_id, trace in enumerate(dim_samples_list):
                dim_sample_list.append(dim_samples_list[trace_id][dim])
            dim_sample_list = [' '.join(trace) for trace in dim_sample_list]
            dim_sample=MultidimSample()
            dim_sample.set_sample(dim_sample_list)
            dim_sample.get_sample_typeset()
            dim_sample.get_vertical_sequence_database()
            dim_sample_dict[dim] = dim_sample
        self._dim_sample_dict = dim_sample_dict

    def calc_type_pattern_disjointness(self, stat_method:str='advanced') -> None:
        disjointness_list = [0]*self._sample_size
        supported_typeset = self.get_sample_supported_typeset(support=1.0)
        repeated_types = {}
        disjointness_min = -1
        disjointness_max = -1

        for trace_num, trace in enumerate(self._sample):
            first_type_index = -1
            last_type_index = -1
            first_pattern_index = -1
            last_pattern_index = -1
            number_of_types_in_pattern_range = 0
            number_of_patterns_in_type_range = 0
            number_of_types = 0
            number_of_patterns = 0
            number_of_patterns_before_type = 0
            number_of_types_before_pattern = 0
            number_of_patterns_behind_type = 0
            number_of_types_behind_pattern = 0
            temp_number_of_types = 0
            temp_number_of_patterns = 0

            complete_vsdb = self.get_att_vertical_sequence_database()
            for (att_dim, single_dim_vsdb) in complete_vsdb.items():
                repeated_types[att_dim] = {key for key in single_dim_vsdb if trace_num in single_dim_vsdb[key] and len(single_dim_vsdb[key][trace_num]) >= 2}

            event_array = [event[:-1].split(';') for event in trace.split()]

            for event_num, event in enumerate(event_array):
                for dim, value in enumerate(event):
                    if isinstance(first_type_index, int) and isinstance(first_pattern_index,int):
                        if value in supported_typeset:
                            first_type_index = (event_num, dim)
                            if value in repeated_types[dim]:
                                first_pattern_index = (event_num, dim)
                                number_of_patterns_in_type_range = 1
                                number_of_types_in_pattern_range = 1
                            else:
                                number_of_types_before_pattern = 1
                        elif value in repeated_types[dim]:
                            first_pattern_index = (event_num, dim)
                            number_of_patterns_before_type = 1
                    elif first_type_index == -1:
                        if value in supported_typeset:
                            first_type_index = (event_num, dim)
                            if value in repeated_types[dim]:
                                number_of_types_in_pattern_range = 1
                                number_of_patterns_in_type_range = 1
                            else:
                                temp_number_of_types = 1
                        elif value in repeated_types[dim]:
                            last_pattern_index = (event_num, dim)
                            number_of_patterns_before_type += 1
                    elif first_pattern_index == -1:
                        if value in repeated_types[dim]:
                            first_pattern_index = (event_num, dim)
                            if value in supported_typeset:
                                number_of_types_in_pattern_range = 1
                                number_of_patterns_in_type_range = 1
                            else:
                                temp_number_of_patterns = 1
                        elif value in supported_typeset:
                            last_type_index = (event_num, dim)
                            number_of_types_before_pattern += 1
                    else:
                        if value in supported_typeset:
                            last_type_index = (event_num, dim)
                            number_of_patterns_in_type_range += temp_number_of_patterns
                            temp_number_of_patterns = 0
                            if value in repeated_types[dim]:
                                last_pattern_index = (event_num, dim)
                                number_of_types_in_pattern_range += temp_number_of_types + 1
                                number_of_patterns_in_type_range += 1
                                temp_number_of_types = 0
                            else:
                                temp_number_of_types += 1
                        elif value in repeated_types[dim]:
                            last_pattern_index = (event_num, dim)
                            number_of_types_in_pattern_range += temp_number_of_types
                            temp_number_of_types = 0
                            temp_number_of_patterns += 1
            number_of_types_behind_pattern = temp_number_of_types
            number_of_patterns_behind_type = temp_number_of_patterns

            LOGGER.debug("first_type_index: %s", first_type_index)
            LOGGER.debug("last_type_index:  %s", last_type_index)
            LOGGER.debug("first_pattern_index: %s", first_pattern_index)
            LOGGER.debug("last_pattern_index:  %s", last_pattern_index)
            LOGGER.debug("number_of_types_before_pattern:   %s", number_of_types_before_pattern)
            LOGGER.debug("number_of_types_in_pattern_range: %s", number_of_types_in_pattern_range)
            LOGGER.debug("number_of_types_behind_pattern:   %s", number_of_types_behind_pattern)
            LOGGER.debug("number_of_patterns_before_type:   %s", number_of_patterns_before_type)
            LOGGER.debug("number_of_patterns_in_type_range: %s", number_of_patterns_in_type_range)
            LOGGER.debug("number_of_patterns_behind_type:   %s", number_of_patterns_behind_type)

            if stat_method == 'advanced':
                t_ges = number_of_types_before_pattern + number_of_types_in_pattern_range + number_of_types_behind_pattern
                p_ges = number_of_patterns_before_type + number_of_patterns_in_type_range + number_of_patterns_behind_type

                LOGGER.debug(trace)
                LOGGER.debug(supported_typeset)
                LOGGER.debug(repeated_types)
                if p_ges == 0:
                    disjointness_list[trace_num] = 1
                elif t_ges == 0:
                    disjointness_list[trace_num] = -1
                else:


                    t_diff = (number_of_types_before_pattern**1 - number_of_types_behind_pattern**1)/t_ges
                    p_diff = (number_of_patterns_before_type**1 - number_of_patterns_behind_type**1)/p_ges

                    t_dev = (number_of_types_in_pattern_range - t_ges/2)/(t_ges/2)
                    p_dev = (number_of_patterns_in_type_range - p_ges/2)/(p_ges/2)

                    disjointness_list[trace_num] = abs(t_dev)*abs(p_dev) * (t_diff - p_diff)/2

                    LOGGER.debug("Term t_dev: %s", abs(t_dev))
                    LOGGER.debug("Term p_dev: %s", abs(p_dev))
                    LOGGER.debug("Term diff : %s", (t_diff-p_diff)/2)
                    LOGGER.debug("Term t_dif: %s", t_diff)
                    LOGGER.debug("Term p_dif: %s", p_diff)

                if trace_num == 0:
                    disjointness_min = disjointness_list[0]
                    disjointness_max = disjointness_list[0]
                else:
                    if disjointness_list[trace_num] < disjointness_min:
                        disjointness_min = disjointness_list[trace_num]
                    elif disjointness_list[trace_num] > disjointness_max:
                        disjointness_max = disjointness_list[trace_num]
            elif stat_method == 'simple':
                t_ges = number_of_types_before_pattern + number_of_types_in_pattern_range + number_of_types_behind_pattern
                p_ges = number_of_patterns_before_type + number_of_patterns_in_type_range + number_of_patterns_behind_type

                if p_ges == 0:
                    disjointness_list[trace_num] = 1
                elif t_ges == 0:
                    disjointness_list[trace_num] = -1
                else:
                    t_diff = (number_of_types_before_pattern**1 - number_of_types_behind_pattern**1)/t_ges
                    p_diff = (number_of_patterns_before_type**1 - number_of_patterns_behind_type**1)/p_ges

                    t_dev = ((number_of_types_in_pattern_range - t_ges/2)/(t_ges/2))**(1/2)
                    p_dev = ((number_of_patterns_in_type_range - p_ges/2)/(p_ges/2))**(1/2)

                    if t_diff - p_diff != 0:
                        disjointness_list[trace_num] = (t_diff - p_diff)/abs(t_diff - p_diff)*abs(t_dev)*abs(p_dev)
                    else:
                        disjointness_list[trace_num] = 0

                    LOGGER.debug("Term t_dev: %s", abs(t_dev))
                    LOGGER.debug("Term p_dev: %s", abs(p_dev))

                if trace_num == 0:
                    disjointness_min = disjointness_list[0]
                    disjointness_max = disjointness_list[0]
                else:
                    if disjointness_list[trace_num] < disjointness_min:
                        disjointness_min = disjointness_list[trace_num]
                    elif disjointness_list[trace_num] > disjointness_max:
                        disjointness_max = disjointness_list[trace_num]
            else:
                t_ges = number_of_types_before_pattern + number_of_types_in_pattern_range + number_of_types_behind_pattern
                p_ges = number_of_patterns_before_type + number_of_patterns_in_type_range + number_of_patterns_behind_type

                if p_ges == 0:
                    disjointness_list[trace_num] = 1
                elif t_ges == 0:
                    disjointness_list[trace_num] = -1
                else:


                    t_diff = (number_of_types_before_pattern**1 - number_of_types_behind_pattern**1)/t_ges
                    p_diff = (number_of_patterns_before_type**1 - number_of_patterns_behind_type**1)/p_ges

                    t_dev = (number_of_types_in_pattern_range - t_ges/2)/(t_ges/2)
                    p_dev = (number_of_patterns_in_type_range - p_ges/2)/(p_ges/2)

                    if t_diff-p_diff != 0:
                        disjointness_list[trace_num] = abs(t_dev)*abs(p_dev) * abs((t_diff - p_diff)/2)**(1/64)
                    else:
                        disjointness_list[trace_num] = 0

                    LOGGER.debug("Term t_dev: %s", abs(t_dev))
                    LOGGER.debug("Term p_dev: %s", abs(p_dev))
                    LOGGER.debug("Term diff : %s", (t_diff-p_diff)/2)
                    LOGGER.debug("Term t_dif: %s", t_diff)
                    LOGGER.debug("Term p_dif: %s", p_diff)

                if trace_num == 0:
                    disjointness_min = disjointness_list[0]
                    disjointness_max = disjointness_list[0]
                else:
                    if disjointness_list[trace_num] < disjointness_min:
                        disjointness_min = disjointness_list[trace_num]
                    elif disjointness_list[trace_num] > disjointness_max:
                        disjointness_max = disjointness_list[trace_num]

        self.sample_type_pattern_disjointness_list = disjointness_list
        self.sample_type_pattern_disjointness_score = (disjointness_max - disjointness_min)/2

    ##################################################

    def get_sample_min_trace(self) -> tuple[str,int]:
        """
            Returns the trace with minimal number of events within the sample.

            Returns:
                A trace with minimal length as a string.

            Raises:
                EmptySampleError: The given sample is empty.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')
        else:
            min_trace = self._sample[0]
            min_trace_spaces = min_trace.count(" ")
            for trace in self._sample:
                if trace.count(" ") < min_trace_spaces:
                    min_trace = trace
                    min_trace_spaces = trace.count(" ")
            return min_trace, min_trace_spaces + 1

    def get_sample_max_trace(self) -> str|None:
        """
            Returns the trace with maximal number of events within the sample.

            Returns:
                A trace with maximal length as a string.

            Raises:
                EmptySampleError: The given sample is empty.
        """
        Sample.get_sample_max_trace(self)

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
        return Sample.get_sample_max_min_avg_trace(self)

    def get_l_w_tuples(self, query_string_length_interval:tuple|None=None) -> list[tuple]:
        """
            Returns a list of all possible l-w-tuples for the current sample.

            The tuples haven equal w-tuples for each gap between events and
            (0,0)-tuples between attributes of the same event.

            We assume the same event dimension for each event in the sample.

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
        _, min_trace_len = self.get_sample_min_trace()
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

        self.set_sample_event_dimension()
        sample_event_dimension = self._sample_event_dimension

        if sample_event_dimension == 1:
            return l_w_list

        # Add (0,0)-tuples between attributes
        lwtuple_index = 0
        for lwtuple in l_w_list:
            current_query_string_length = lwtuple[0]
            current_local_window_sizes = lwtuple[1]
            for i in range(current_query_string_length):
                for _ in range(sample_event_dimension-1):
                    if i == 0:
                        index = 1
                    else:
                        index = (i*sample_event_dimension)+1
                    current_local_window_sizes.insert(index, tuple((0,0)))
            # Set adapted local window sizes
            l_w_list[lwtuple_index] = tuple((current_query_string_length, current_local_window_sizes))
            lwtuple_index = lwtuple_index +1
        return l_w_list

    def get_sample_typeset(self, attribute:int=-1) -> set:
        """
            Returns the typeset of the sample.
            If it is not computed yet, it will be calculated.

            Args:
                attribute [=-1]: Optional if only a specific domain shall be
                    returned. For "-1" the whole typeset is returned.

            Returns:
                set: containing all types

            Raises:
                None:

            Passes:
                IndexError: if attribute is less than 0 or greater than or
                    equals the event dimension, or is not -1.
        """
        if attribute == -1:
            if self._sample_typeset is None:
                self.calc_sample_typeset()
            assert self._sample_typeset is not None
            return self._sample_typeset
        else:
            if attribute not in self._sample_att_typesets:
                self.calc_sample_typeset(attribute)
            return self._sample_att_typesets[attribute]

    def get_sample_supported_typeset(self, support:float=1.0, attribute:int=-1) -> set:
        """
            Returns the supported typeset of the sample.
            If it is not computed yet, it will be calculated.

            Args:
                support [=1.0]: Optional if any other support is needed

                attribute [=-1]: Optional if only a specific domain shall be returned
                    For "-1" the whole typeset is returned

            Returns:
                set: all types that statisfy the support

            Raises:
                None

            Passes:
                IndexError: If attribute is less than 0 or greater than or equals
                    the event dimension, or is not -1.
                InvalidQuerySupportError: Support is less than 0 or greater
                    than 1.
        """
        sample_sized_support = ceil(self._sample_size * support)
        if attribute == -1:
            if self._sample_supported_typeset is None or sample_sized_support not in self._sample_supported_typeset:
                self.calc_sample_supported_typeset(support, attribute)
            return self._sample_supported_typeset[sample_sized_support]
        else:
            if sample_sized_support not in self._sample_att_supported_typesets or attribute not in self._sample_att_supported_typesets[sample_sized_support]:
                self.calc_sample_supported_typeset(support, attribute)
            return self._sample_att_supported_typesets[sample_sized_support][attribute]

    def set_sample(self, sample:list) -> None:
        """
            Sets _sample to sample and updates corresponding attributes.

            Args:
                sample: List of strings which represent traces.
        """
        if not isinstance(sample, list):
            raise TypeError("sample must be of type <list>!")
        self._sample = sample
        self.set_sample_size()
        self.set_sample_event_dimension()

        self._sample_supported_typeset = {}
        self._sample_att_supported_typesets = {}
        self.sample_typeset = None
        self.sample_att_typesets = {}
        self.sample_vertical_sequence_database = None
        self.sample_att_vertical_sequence_database = None

    def set_sample_size(self) -> None:
        """
            Determines and sets _sample_size for current _sample.
        """
        self._sample_size = len(self._sample)

    def set_sample_event_dimension(self,uniform_dimension:bool=False) -> None:
        """
            Determines and set the maximum event dimension.

            Args:
                uniform_dimension [=False]: Boolean which indicates whether the
                    dimension of all events is the same.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')

        dimension = -1
        if uniform_dimension is True:
            trace=self._sample[0]
            event = trace.split(' ')[0]
            dimension = event.count(";")
        else:
            for trace in self._sample:
                while len(trace)>0:
                    if trace[0] == " ":
                        trace = trace[1:]
                    else:
                        event = trace.split(' ')[0]
                        event_dimension = event.count(";")
                        if event_dimension>dimension:
                            dimension = event_dimension
                        trace = trace[len(event):]
        self._sample_event_dimension = dimension

    ##################################################

    
    def set_sample_typeset(self, typeset) -> None:
        """
            Stores the typeset of the sample.

            Args:
                None

            Returns:
                None

            Raises:
                TypeError: if typeset is not of type <set>
        """
        if typeset is not None:
            if not isinstance(typeset,set):
                raise TypeError
        self._sample_typeset = typeset

    def get_sample_att_typesets(self) -> dict:
        """
            Returns the typesets of the sample seperated by the attributes.
            If it is not computed yet, it will be calculated.

            Args:
                None

            Returns:
                Dict: "attribute" -> {types per dimension}

            Raises:
                None

            Passes:
                EmptySampleError: If the given sample is empty.

                IndexError: If an index occurs in trace_index_list that is not
                    between 0 and len(sequence_database)-1.
        """
        if not len(self._sample_att_typesets) == self._sample_event_dimension:
            for attribute in range(0, self._sample_event_dimension):
                self.calc_sample_typeset(attribute)
        return self._sample_att_typesets

    
    def set_sample_att_typesets(self, typesets) -> None:
        """
            Sets the typesets of the sample seperated by the attributes.

            Args:
                None

            Returns:
                None

            Raises:
                None

            Passes:
                TypeError, if "typesets" is not of type <dict>
        """
        if not isinstance(typesets,dict):
            raise TypeError("'Typesets' have to be of type <dict>")
        self._sample_vertical_sequence_database = typesets

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

            Passes:
                EmptySampleError: If the given sample is empty.

                IndexError: If an index occurs in trace_index_list that is not
                    between 0 and len(sequence_database)-1.
        """
        if self._sample_vertical_sequence_database is None:
            self.calc_vertical_sequence_databases()
        assert self._sample_vertical_sequence_database is not None
        return self._sample_vertical_sequence_database

    def set_vertical_sequence_database(self, vsdb:dict) -> None:
        """
            Set _sample_vertical_sequences_database.

            Args:
                vsdb: a vertical sequence database of type <dict>

            Returns:
                None

            Raises:
                TypeError: if vsdb is not of type <dict>
        """
        if vsdb is not None:
            if not isinstance(vsdb,dict):
                raise TypeError
        self._sample_vertical_sequence_database = vsdb

    def get_att_vertical_sequence_database(self) -> dict:
        """
            Returns the vertical sequence database of the sample.

            If it is not computed yet, it will be calculated.

            Args:
                None

            Returns:
                Dict: "type" -> [positions per traces]

            Raises:
                None

            Passes:
                EmptySampleError: If the given sample is empty.

                IndexError: If an index occurs in trace_index_list that is not
                    between 0 and len(sequence_database)-1.
        """
        if self._sample_att_vertical_sequence_database is None:
            self.calc_vertical_sequence_databases()
        assert self._sample_att_vertical_sequence_database is not None
        return self._sample_att_vertical_sequence_database

    def set_att_vertical_sequence_database(self, vsdb:dict) -> None:
        """
            Set _sample_vertical_sequences_database

            Args:
                vsdb: a vertical sequence database of type <dict>

            Returns:
                None

            Raises:
                TypeError: if vsdb is not of type <dict>
        """
        if vsdb is not None:
            if not isinstance(vsdb,dict):
                raise TypeError
        self._sample_att_vertical_sequence_database = vsdb

    def get_dim_sample_dict(self) ->dict:
        """
            Returns the Dictionary containing for each dimension its one-dim Sample.

            Returns:
                dict: {dimension: one-dim Sample}
        """
        if not self._dim_sample_dict:
            self.calc_dim_sample_dict()
        return self._dim_sample_dict

    def set_dim_sample_dict(self, dim_sample_dict:dict):
        """Set _dim_sample_dict

        Args:
            dim_sample_dict (dict): Dictionary containing for each dimension its one-dim Sample
        """
        self._dim_sample_dict = dim_sample_dict

    def get_type_pattern_disjointness_list(self) -> list:
        if self._sample_type_pattern_disjointness_list is None:
            self.calc_type_pattern_disjointness()
        assert self._sample_type_pattern_disjointness_list is not None
        return self._sample_type_pattern_disjointness_list

    def set_type_pattern_disjointness_list(self, type_pattern_disjointness_list:list) -> None:
        if type_pattern_disjointness_list is not None:
            if len(type_pattern_disjointness_list) != self._sample_size:
                raise TypeError
        self._sample_type_pattern_disjointness_list = type_pattern_disjointness_list

    def get_type_pattern_disjointness_score(self) -> float:
        if self._sample_type_pattern_disjointness_score is None:
            self.calc_type_pattern_disjointness()
        assert self._sample_type_pattern_disjointness_score is not None
        return self._sample_type_pattern_disjointness_score

    def set_type_pattern_disjointness_score(self, type_pattern_disjointness_score:float) -> None:
        if type_pattern_disjointness_score is not None:
            if abs(type_pattern_disjointness_score) > 1.0:
                raise TypeError
        self._sample_type_pattern_disjointness_score = type_pattern_disjointness_score

    ##################################################

    sample_typeset = property(get_sample_typeset, set_sample_typeset)
    sample_att_typesets = property(get_sample_att_typesets, set_sample_att_typesets)
    sample_vertical_sequence_database = property(get_vertical_sequence_database, set_vertical_sequence_database)
    sample_att_vertical_sequence_database = property(get_att_vertical_sequence_database, set_att_vertical_sequence_database)
    sample_type_pattern_disjointness_list = property(get_type_pattern_disjointness_list, set_type_pattern_disjointness_list)
    sample_type_pattern_disjointness_score = property(get_type_pattern_disjointness_score, set_type_pattern_disjointness_score)
