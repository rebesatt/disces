#!/usr/bin/python3
"""Contains different error classes for queries, samples and discovery"""

class EmptySampleError(RuntimeError):
    """
        An Error for when a given Sample is empty.

        :param str message:
            Message for the error.
    """

class InvalidTraceError(RuntimeError):
    """
        An Error for when a given trace has wrong format, e.g. ends with
        whitespace.

        :param str message:
            Message for the error.
    """

class InvalidSampleDimensionError(RuntimeError):
    """
        An Error for when a given dimension is greater than the max. event
        dimension in the sample.

        :param str message:
            Message for the error.
    """

###############################################################################

class ShinoharaInvalidPositionError(RuntimeError):
    """
        An Error for when attribut select_position of function
        shinohara_discovery_icdt is not from SHINOHARA_SELECT_POSITION.

        :param str message:
            Message for the error.
    """

class ShinoharaInvalidOperationError(RuntimeError):
    """
        An Error for when attribut select_operation of function
        shinohara_discovery_icdt is not from SHINOHARA_SELECT_OPERATION.

        :param str message:
            Message for the error.
    """

###############################################################################

class InconsistentQueryError(RuntimeError):
    """
        An Error for when attributes of the current query are inconsistent.

        :param str message:
            Message for the error.
    """

class QueryRegexError(RuntimeError):
    """
        An Error for when the gap constraint or local window size handling went
        wrong during the generation of the regex of the current query string.

        :param str message:
            Message for the error.
    """

class InvalidQueryStringLengthError(RuntimeError):
    """
        An Error if the query string length is incompatible with the query
        string.

        :param str message:
            Message for the error.
    """

class InvalidEventDimensionError(RuntimeError):
    """
        An Error if the query event dimension is less than 1.

        :param str message:
            Message for the error.
    """

class InvalidQuerySupportError(RuntimeError):
    """
        An Error for when a given support is not between 0 and 1.

        :param str message:
            Message for the error.
    """

class InvalidQueryGapConstraintError(RuntimeError):
    """
        An Error for when given gap constraints are incompatible with the query
        string length.

        :param str message:
            Message for the error.
    """

class InvalidQueryGlobalWindowSizeError(RuntimeError):
    """
        An Error for when a given window size is incompatible with the query
        string length.

        :param str message:
            Message for the error.
    """

class InvalidQueryLocalWindowSizeError(RuntimeError):
    """
        An Error for when given local window sizes are incompatible with the
        query string length.

        :param str message:
            Message for the error.
    """

class InvalidQueryClassError(RuntimeError):
    """
        An Error for when a queryclass is not from QUERY_CLASS_LIST.

        :param str message:
            Message for the error.
    """

class InvalidQueryDiscoveryError(RuntimeError):
    """
        An Error for when a discovery algorithm is not from
        DISCOVERY_ALGORITHM_LIST.

        :param str message:
            Message for the error.
    """

class InvalidQueryMatchtestError(RuntimeError):
    """
        An Error for when a match test algorithm is not from MATCH_TEST_LIST.

        :param str message:
            Message for the error.
    """


###############################################################################

class EmptyDataframeError(RuntimeError):
    """
        An Error for when a dataframe is empty.

        :param str message:
            Message for the error.
    """

class InvalidDataframeFilterError(RuntimeError):
    """
        An Error for when a dataframe is filtered by a value in a specific
        column, which does not exist, e.g. 'param support' == 0.5, while 0.5
        never occurs in column 'param support'.

        This would lead to a dataframe with 0 rows and raise an error during
        seabors plotting functions

        :param str message:
            Message for the error.
    """
