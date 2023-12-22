#!/usr/bin/python3
"""Contains a class for generating samples for experiments"""
import logging
from itertools import product
from string import ascii_letters
from random import randrange, sample
import numpy as np
from sample import Sample

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

class SampleGenerator:
    """
        Class for generating samples with different properties.
    """

    def __init__(self):
        self._sample:Sample = Sample()

    def generate_random_sample(self, sample_size:int=100, min_trace_length:int=100, max_trace_length:int=150, type_length:int=-1, type_count:int=-1) -> Sample:
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

        # Build a typeset (list, because sampling from sets is deprecated)
        upper_bound = max_trace_length*sample_size
        typeset = list(self.build_typeset(upper_bound,type_length, type_count))

        self._sample = Sample()

        for _ in range(sample_size):
            random_trace_length = min_trace_length
            if min_trace_length<max_trace_length:
                random_trace_length = randrange(min_trace_length,max_trace_length)
            random_trace = ""
            for type_count in range(random_trace_length):
                random_type = sample(typeset,1)
                random_trace = random_trace + str(random_type[0])
                #Separate random types by whitespace if it is not the last type
                if type_count < random_trace_length-1:
                    random_trace = random_trace + " "
            self._sample._sample.append(random_trace)

        self._sample.set_sample_size()
        self._sample.set_sample_typeset()
        LOGGER.debug('generate_random_sample - Finished')
        return self._sample

    def generate_sample_w_empty_queryset(self, sample_size:int=100, min_trace_length:int=100, max_trace_length:int=150) -> Sample:
        """
            Generates a sample with no repeated types for the given parameters.
            The resulting queryset will be empty. There are no types nor repeated variables that fulfill the support of 1.0.

            Args:
                sample_size: Integer which represents the number of traces
                    which has to be generated.

                min_trace_length: Minimum length of each trace in the generated
                    sample as an integer.

                max_trace_length: Maximum length of each trace in the generated
                    sample as an integer.

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
        # Compute the trace lengths
        if min_trace_length == max_trace_length:
            trace_lengths= [min_trace_length]*sample_size
        else:
            rng = np.random.default_rng()
            trace_lengths = rng.integers(low=min_trace_length, high=max_trace_length, endpoint=True, size= sample_size)

        # Build a typeset (list, because sampling from sets is deprecated)
        type_sum = np.sum(trace_lengths)
        typeset = list(self.build_typeset(type_sum))

        self._sample = Sample()
        typeset_count= 0
        for i in range(sample_size):
            trace_length = trace_lengths[i]
            trace_list = typeset[typeset_count: typeset_count + trace_length]
            trace = ' '.join(trace_list)
            typeset_count += trace_length
            self._sample._sample.append(trace)

        self._sample.set_sample_size()
        self._sample.set_sample_typeset()
        LOGGER.debug('generate_random_sample - Finished')
        return self._sample

    def generate_fragmentation_gauss_sample(self, sample_size:int=100, min_trace_length:int=100, max_trace_length:int=150, type_length:int=-1, type_count:int=-1) -> Sample:
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

            Returns:
                An instance of class Sample.

            Raises:
                AssertionError: min_trace_length has to be greater than zero and
                less equal than max_trace_length.
        """
        LOGGER.debug('generate_fragmentation_gauss_sample - Starting')
        LOGGER.debug('generate_fragmentation_gauss_sample - sample size %s min trace length %s, max trace length %s', str(sample_size), str(min_trace_length), str(max_trace_length))

        assert min_trace_length<=max_trace_length
        assert min_trace_length>0

        # Build a typeset
        upper_bound = max_trace_length*sample_size
        typelist = list(self.build_typeset(upper_bound,type_length, type_count))

        # Build a numpy array to store the indices which corresponds to the types.
        typelist_index=[]
        for counter in range(0,len(typelist)):
            typelist_index.append(counter)
        np_typelist_index = np.array(typelist_index)

        # Build distribution and transform it into a numpy array
        distribution_gauss = self._build_fragmentation_of_1_gauss(typelist)
        np_distribution_gauss = np.array(distribution_gauss)

        self._sample = Sample()

        for _ in range(sample_size):
            random_trace_length = min_trace_length
            if min_trace_length<max_trace_length:
                random_trace_length = randrange(min_trace_length,max_trace_length)
            random_trace = ""
            for type_count in range(random_trace_length):
                random_type_index = int(np.random.choice(np_typelist_index, 1, p=np_distribution_gauss))
                random_type = typelist[random_type_index]
                random_trace = random_trace + str(random_type[0])
                #Separate random types by whitespace if it is not the last type
                if type_count < random_trace_length-1:
                    random_trace = random_trace + " "
            self._sample._sample.append(random_trace)

        self._sample.set_sample_size()
        self._sample.set_sample_typeset()
        LOGGER.debug('generate_fragmentation_gauss_sample - Finished')
        return self._sample

    def generate_fragmentation_quartered_sample(self, sample_size:int=100, min_trace_length:int=100, max_trace_length:int=150, type_length:int=-1, type_count:int=-1) -> Sample:
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

            Returns:
                An instance of class Sample.

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
        typelist = list(self.build_typeset(upper_bound,type_length, type_count))

        # Build a numpy array to store the indices which corresponds to the types.
        typelist_index=[]
        for counter in range(0,len(typelist)):
            typelist_index.append(counter)
        np_typelist_index = np.array(typelist_index)

        # Build distribution and transform it into a numpy array
        distribution_quartered = self._build_fragmentation_of_1_quartered(typelist)
        np_distribution_quartered = np.array(distribution_quartered)

        self._sample = Sample()

        for _ in range(sample_size):
            random_trace_length = min_trace_length
            if min_trace_length<max_trace_length:
                random_trace_length = randrange(min_trace_length,max_trace_length)
            random_trace = ""
            for type_count in range(random_trace_length):
                random_type_index = int(np.random.choice(np_typelist_index, 1, p=np_distribution_quartered))
                random_type = typelist[random_type_index]
                random_trace = random_trace + str(random_type[0])
                #Separate random types by whitespace if it is not the last type
                if type_count < random_trace_length-1:
                    random_trace = random_trace + " "
            self._sample._sample.append(random_trace)

        self._sample.set_sample_size()
        self._sample.set_sample_typeset()
        LOGGER.debug('generate_fragmentation_quartered_sample - Finished')
        return self._sample

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
        if type_length < 0:
            # Calculate upper bound for the size of a typeset: if all positions
            # in all traces were different we would need
            #       max_trace_length*sample_size
            # types. Since build_typeset() can construct all combinations of
            # ascii letters for a given length, it suffices to solve the
            # inequation for i:
            #       52^i >= max_trace_length*sample_size
            tmp = 52
            type_length = 1
            while tmp < upper_bound:
                tmp = tmp*52
                type_length = type_length+1

        assert type_length > 0
        count_to_type_count = 0
        typeset = set()
        for elem in product(ascii_letters, repeat=type_length):
            typeset.add(''.join(elem))
            count_to_type_count = count_to_type_count+1
            if count_to_type_count == type_count:
                break
            if count_to_type_count == upper_bound:
                break
        return typeset

    def _build_fragmentation_of_1_gauss(self, typelist:list) -> list:
        """
            Finds n distinct numbers in the range from 0 to 1 s.t. their sum
            equals 1.

            Args:
                typelist: List of length n which contains types.

            Returns:
                A list of n probabilities (floats).
        """
        prob_list = []
        current_sum = 0
        exponent = int(len(str(len(typelist)))*2) * (-1)

        for _ in range(0,len(typelist)):
            current_sum = current_sum + pow(10, exponent)
            prob_list.append(current_sum)

        # Add rest to 1 to all probabilites
        overall_sum = sum(prob_list)
        rest = 1-overall_sum
        if rest > 0:
            rest_divided = (rest/len(typelist)) - pow(10, exponent-1)
            for elem in range(0,len(typelist)):
                prob_list[elem] = prob_list[elem]+rest_divided

        # Modify last entry s.t. the overall sum is 1
        overall_sum = sum(prob_list)
        prob_list[len(typelist)-1] = prob_list[len(typelist)-1] + (1-overall_sum)

        assert sum(prob_list)<=1
        return prob_list

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
        prob_list = []
        pos = 1
        quarter = (len(typelist)//4)+1

        while pos <= len(typelist):
            if pos%4 == 1:
                prob_list.append(0.6/quarter)
            elif pos%4 == 2:
                prob_list.append(0.25/quarter)
            elif pos%4 == 3:
                prob_list.append(0.1/quarter)
            elif pos%4 == 0:
                prob_list.append(0.05/quarter)
            pos = pos+1

        # Add rest to 1 to all probabilites
        overall_sum = sum(prob_list)
        rest = 1-overall_sum
        if rest > 0:
            rest_divided = (rest/len(typelist))
            for elem in range(0,len(typelist)):
                prob_list[elem] = prob_list[elem]+rest_divided

        # Modify last entry s.t. the overall sum is 1
        overall_sum = sum(prob_list)
        prob_list[len(typelist)-1] = prob_list[len(typelist)-1] + (1-overall_sum)

        assert sum(prob_list)<=1
        return prob_list
