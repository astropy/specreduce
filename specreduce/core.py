# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDPROC functions

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import inspect
from dataclasses import dataclass

__all__ = ['SpecreduceOperation']


@dataclass
class SpecreduceOperation:
    """
    An operation to perform as part of a spectroscopic reduction pipeline.

    This class primarily exists to define the basic API for operations:
    parameters for the operation are provided at object creation,
    and then the operation object is called with the data objects required for
    the operation, which then *return* the data objects resulting from the
    operation.
    """
    def __call__(self):
        raise NotImplementedError('__call__ on a SpecreduceOperation needs to '
                                  'be overridden')

    @classmethod
    def as_function(cls, *args, **kwargs):
        """
        Run this operation as a function.  Syntactic sugar for e.g.,
        ``Operation.as_function(arg1, arg2, keyword=value)`` maps to
        ``Operation(arg2, keyword=value)(arg1)`` (if the ``__call__`` of
        ``Operation`` has only one argument)
        """
        argspec = inspect.getargs(SpecreduceOperation.__call__.__code__)
        if argspec.varargs:
            raise NotImplementedError('There is not a way to determine the '
                                      'number of inputs of a *args style '
                                      'operation')
        ninputs = len(argspec.args) - 1

        callargs = args[:ninputs]
        noncallargs = args[ninputs:]
        op = cls(*noncallargs, **kwargs)
        return op(*callargs)
