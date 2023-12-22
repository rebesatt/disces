#!/usr/bin/python3
"""Contains a class for generating multidimensional samples for experiments"""
import logging
from random import randrange, sample
import numpy as np
from sample_multidim import MultidimSample
from generator import SampleGenerator
from copy import deepcopy

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

class MultidimSampleGenerator(SampleGenerator):
    """
        Class for generating multidim. samples with different properties.
    """

    def __init__(self):
        self._msample:MultidimSample = MultidimSample()
        self._settings = None
        self._setting_count = None

    def generate_random_sample(self, sample_size:int=100, min_trace_length:int=100, max_trace_length:int=150, type_length:int=-1, type_count:int=-1, event_dimension:int=1) -> MultidimSample:
        """
            Generates a random sample for the given parameters.

            For each trace the trace length and each type is picked randomly.

            Args:
                sample_size: Integer which represents the number of traces
                    which has to be generated.

                min_trace_length: Minimum length of each trace in the generated
                    sample as an integer.

                max_trace_length: Maximum length of each trace in the generated
                    sample as an integer.

                type_length: Integer which can be used to define the length of
                    each type. Default -1 means that the length will be
                    calculated depending on max_trace_length and sample_size.

                type_count: Integer which can be used to define how many types
                    the typeset should contain.

                event_dimension: Integer which represents the number of
                    attributes per event.

            Returns:
                An instance of class MultidimSample.

            Raises:
                AssertionError: min_trace_length has to be greater than zero and
                less equal than max_trace_length.
        """
        LOGGER.debug('generate_random_sample - Starting')
        LOGGER.debug('generate_random_sample - sample size %s min trace length %s, max trace length %s', str(sample_size), str(min_trace_length), str(max_trace_length))

        assert min_trace_length<=max_trace_length
        assert min_trace_length>0

        # Build a typeset (list, because sampling from sets is deprecated)
        upper_bound = max_trace_length*sample_size
        typeset = list(self.build_typeset(upper_bound,type_length,type_count))

        self._msample = MultidimSample()
        new_sample = []

        for _ in range(sample_size):
            random_trace_length = min_trace_length
            if min_trace_length<max_trace_length:
                random_trace_length = randrange(min_trace_length,max_trace_length)
            random_trace = ""
            for type_count in range(random_trace_length):
                for _ in range(event_dimension):
                    random_type = sample(typeset,1)
                    random_trace = random_trace + str(random_type[0]) + ";"
                #Separate random types by whitespace if it is not the last type
                if type_count < random_trace_length-1:
                    random_trace = random_trace + " "
            new_sample.append(random_trace)

        self._msample.set_sample(new_sample)
        self._msample.calc_sample_typeset(calculate_all=True)
        LOGGER.debug('generate_random_sample - Finished')
        return self._msample

    def generate_sample_w_empty_queryset(self, sample_size:int=100, min_trace_length:int=100, max_trace_length:int=150, event_dimensions:int=1) -> MultidimSample:
        """
            Generates a sample with no repeated types for the given parameters.
            The resulting queryset will be empty.

            Args:
                sample_size: Integer which represents the number of traces
                    which has to be generated.

                min_trace_length: Minimum length of each trace in the generated
                    sample as an integer.

                max_trace_length: Maximum length of each trace in the generated
                    sample as an integer.

                event_dimension: Integer which represents the number of
                    attributes per event.

            Returns:
                An instance of class Sample.

            Raises:
                AssertionError: min_trace_length has to be greater than zero
                    and less equal than max_trace_length.
        """
        LOGGER.debug('generate_random_sample - Starting')
        LOGGER.debug('generate_random_sample - sample size %s min trace length %s, max trace length %s', str(sample_size), str(min_trace_length), str(max_trace_length))

        assert min_trace_length<=max_trace_length
        assert min_trace_length>0
        generator = SampleGenerator()
        generator.generate_sample_w_empty_queryset(sample_size=sample_size, min_trace_length=min_trace_length, max_trace_length=min_trace_length)
        sample_set= []
        for trace in generator._sample._sample:
            new_trace = []
            for event in trace.split():
                new_event= f"{event};" *event_dimensions
                new_trace.append(new_event)
            sample_set.append(' '.join(new_trace))
        self._msample = MultidimSample()
        self._msample.set_sample(sample_set)

        LOGGER.debug('generate_random_sample - Finished')
        return self._msample

    def generate_fragmentation_gauss_sample(self, sample_size:int=100, min_trace_length:int=100, max_trace_length:int=150, type_length:int=-1, type_count:int=-1, event_dimension:int=1) -> MultidimSample:
        """
            Generates a random sample for the given parameters, using different
            probabilities for each type in the generated typeset.

            For each trace the trace length and each type is picked randomly.

            Args:
                sample_size: Integer which represents the number of traces
                    which has to be generated.

                min_trace_length: Minimum length of each trace in the generated
                    sample as an integer.

                max_trace_length: Maximum length of each trace in the generated
                    sample as an integer.

                type_length: Integer which can be used to define the length of
                    each type. Default -1 means that the length will be
                    calculated depending on max_trace_length and sample_size.

                type_count: Integer which can be used to define how many types
                    the typeset should contain.

                event_dimension: Integer which represents the number of
                    attributes per event.

            Returns:
                An instance of class MultidimSample.

            Raises:
                AssertionError: min_trace_length has to be greater than zero
                    and less equal than max_trace_length.
        """
        LOGGER.debug('generate_fragmentation_gauss_sample - Starting')
        LOGGER.debug('generate_fragmentation_gauss_sample - sample size %s min trace length %s, max trace length %s', str(sample_size), str(min_trace_length), str(max_trace_length))

        assert min_trace_length<=max_trace_length
        assert min_trace_length>0

        # Build a typeset
        upper_bound = max_trace_length*sample_size
        typelist = list(self.build_typeset(upper_bound,type_length,type_count))

        # Build a numpy array to store the indices which corresponds to the types.
        typelist_index=[]
        for counter in range(0,len(typelist)):
            typelist_index.append(counter)
        np_typelist_index = np.array(typelist_index)

        # Build distribution and transform it into a numpy array
        distribution_gauss = self._build_fragmentation_of_1_gauss(typelist)
        np_distribution_gauss = np.array(distribution_gauss)

        self._msample = MultidimSample()
        new_sample = []

        for _ in range(sample_size):
            random_trace_length = min_trace_length
            if min_trace_length<max_trace_length:
                random_trace_length = randrange(min_trace_length,max_trace_length)
            random_trace = ""
            for type_count in range(random_trace_length):
                for _ in range(event_dimension):
                    random_type_index = int(np.random.choice(np_typelist_index, 1, p=np_distribution_gauss))
                    random_type = typelist[random_type_index]
                    random_trace = random_trace + str(random_type) + ';'
                #Separate random types by whitespace if it is not the last type
                if type_count < random_trace_length-1:
                    random_trace = random_trace + " "
            new_sample.append(random_trace)

        self._msample.set_sample(new_sample)
        self._msample.calc_sample_typeset(calculate_all=True)
        LOGGER.debug('generate_fragmentation_gauss_sample - Finished')
        return self._msample

    def generate_fragmentation_quartered_sample(self, sample_size:int=100, min_trace_length:int=100, max_trace_length:int=150, type_length:int=-1, type_count:int=-1, event_dimension:int=1) -> MultidimSample:
        """
            Generates a random sample for the given parameters, using the
            probabilities for each type in the generated typeset which are
            assigned by the function _build_fragmentation_of_1_quartered().

            For each trace the trace length and each type is picked randomly.

            Args:
                sample_size: Integer which represents the number of traces
                    which has to be generated.

                min_trace_length: Minimum length of each trace in the generated
                    sample as an integer.

                max_trace_length: Maximum length of each trace in the generated
                    sample as an integer.

                type_length: Integer which can be used to define the length of
                    each type. Default -1 means that the length will be
                    calculated depending on max_trace_length and sample_size.

                type_count: Integer which can be used to define how many types
                    the typeset should contain.

                event_dimension: Integer which represents the number of
                    attributes per event.

            Returns:
                An instance of class Sample or MultidimSample.

            Raises:
                AssertionError: min_trace_length has to be greater than zero
                    and less equal than max_trace_length.
        """
        LOGGER.debug('generate_fragmentation_quartered_sample - Starting')
        LOGGER.debug('generate_fragmentation_quartered_sample - sample size %s min trace length %s, max trace length %s', str(sample_size), str(min_trace_length), str(max_trace_length))

        assert min_trace_length<=max_trace_length
        assert min_trace_length>0

        # Build a typeset
        upper_bound = max_trace_length*sample_size
        typelist = list(self.build_typeset(upper_bound,type_length,type_count))

        # Build a numpy array to store the indices which corresponds to the types.
        typelist_index=[]
        for counter in range(0,len(typelist)):
            typelist_index.append(counter)
        np_typelist_index = np.array(typelist_index)

        # Build distribution and transform it into a numpy array
        distribution_quartered = self._build_fragmentation_of_1_quartered(typelist)
        np_distribution_quartered = np.array(distribution_quartered)

        self._msample = MultidimSample()
        new_sample = []

        for _ in range(sample_size):
            random_trace_length = min_trace_length
            if min_trace_length<max_trace_length:
                random_trace_length = randrange(min_trace_length,max_trace_length)
            random_trace = ""
            for type_count in range(random_trace_length):
                for _ in range(event_dimension):
                    random_type_index = int(np.random.choice(np_typelist_index, 1, p=np_distribution_quartered))
                    random_type = typelist[random_type_index]
                    random_trace = random_trace + str(random_type) + ';'
                #Separate random types by whitespace if it is not the last type
                if type_count < random_trace_length-1:
                    random_trace = random_trace + " "
            new_sample.append(random_trace)

        self._msample.set_sample(new_sample)
        self._msample.calc_sample_typeset(calculate_all=True)
        LOGGER.debug('generate_fragmentation_quartered_sample - Finished')
        return self._msample

    def build_typeset(self, upper_bound:int, type_length:int=-1, type_count:int=-1) -> set:
        """
            Creates a set of possible types.

            This function builds all combinations of symbols ascii_letters of a
            length

            Args:
                upper_bound: Integer which represents the upper bound for
                    possible types in the type set

                type_length: Integer which can be used to define the length of
                    each type. Default -1 means that the length will be
                    calculated depending on max_trace_length and sample_size.

                type_count: Integer which can be used to define how many types
                    the typeset should contain.

            Returns:
                A set of strings. Each string represents a type.
        """
        return SampleGenerator.build_typeset(self, upper_bound, type_length=type_length, type_count=type_count)

    def _build_fragmentation_of_1_gauss(self, typelist:list) -> list:
        """
            Finds n distinct numbers in the range from 0 to 1 s.t. their sum
            equals 1.

            Args:
                typelist: List of length n which contains types.

            Returns:
                A list of n probabilities (floats).
        """
        return SampleGenerator._build_fragmentation_of_1_gauss(self, typelist=typelist)

    def _build_fragmentation_of_1_quartered(self, typelist:list) -> list:
        """
            Assgins each quarter of the typeset a different probability.

            The probabilty of the first/second/third/fourth len(typelist)//4
            types, i.e. the first/second/third/fourth quarter, sums up to
            0.6/0.25/0.1/0.05.

            Args:
                typelist: List of length n which contains types.

            Returns:
                A list of n probabilities (floats).
        """
        return SampleGenerator._build_fragmentation_of_1_quartered(self, typelist=typelist)

    def generate_disjoint_type_pattern_interleaving(self, trace_length:int=50, number_of_types:int=25, event_dimensions:int=1) -> MultidimSample | None:
        if self._setting_count is None:
            assert number_of_types <= trace_length
            number_of_vars = trace_length - number_of_types
            number_of_different_vars = max(int(number_of_vars/3), 1)
            number_of_different_vars_mod = number_of_vars % 3

            number_of_characters = len(str(trace_length - 1))
            traces = []

            for trace_num in [0,1]:
                new_trace = []
                for index in range(0, trace_length):
                    new_event = []
                    for dim in range(0, event_dimensions):
                        new_event.append(str(trace_num) + (number_of_characters - len(str(index)))*"0" + str(index) + str(dim))
                    new_trace.append(new_event)
                traces.append(new_trace)

            settings = []
            for trailing_types in range(0, number_of_types+1):
                for trailing_vars in range(0, number_of_vars+1):
                    var_count = 0
                    if trailing_types != 0 and trailing_vars == number_of_vars:
                        continue
                    #if trailing_vars != 0 and trailing_types == number_of_types:
                        # two cases covered twice
                    if trailing_types == number_of_types:
                        continue
                    new_set = deepcopy(traces)
                    current_index = 0
                    for _ in range(0, trailing_types):
                        #insert types
                        for trace_num in range(0, len(traces)):
                            for dim in range(0, event_dimensions):
                                new_set[trace_num][current_index][dim] = "a"+str(current_index)
                        current_index += 1
                    for _ in range(0, number_of_vars - trailing_vars):
                        #insert vars
                        current_var_label = _calc_current_var_label(number_of_different_vars, number_of_different_vars_mod, var_count, number_of_vars)
                        var_count += 1
                        for trace_num in range(0, len(traces)):
                            for dim in range(0, event_dimensions):
                                new_set[trace_num][current_index][dim] = "x"+str(trace_num) + str(dim) + str("-") + current_var_label
                        current_index += 1
                    for _ in range(0, number_of_types - trailing_types):
                        #insert types
                        for trace_num in range(0, len(traces)):
                            for dim in range(0, event_dimensions):
                                new_set[trace_num][current_index][dim] = "a"+str(current_index)
                        current_index += 1
                    for _ in range(0, trailing_vars):
                        #insert_vars
                        current_var_label = _calc_current_var_label(number_of_different_vars, number_of_different_vars_mod, var_count, number_of_vars)
                        var_count += 1
                        for trace_num in range(0, len(traces)):
                            for dim in range(0, event_dimensions):
                                new_set[trace_num][current_index][dim] = "x"+str(trace_num) + str(dim) + str("-") + current_var_label
                        current_index += 1
                    new_trace_set = []
                    for trace_num in range(0, len(traces)):
                        new_trace_set.append(' '.join([';'.join(event)+";" for event in new_set[trace_num]]))
                    settings.append(new_trace_set)
            self._settings = settings
            self._setting_count = 0
        if self._setting_count < len(self._settings):
            self._msample = MultidimSample(self._settings[self._setting_count])
            self._setting_count += 1
            return self._msample
        else:
            self._settings = None
            self._setting_count = None
            return None

def _calc_current_var_label(num_of_diff_vars, num_of_diff_mod, current_count, total_num_of_vars):
    correction_term = 0
    if num_of_diff_mod != 0:
        correction_term = 2

    if current_count < total_num_of_vars - correction_term:
        return str(int(current_count/3))
    else:
        return str(num_of_diff_vars)
