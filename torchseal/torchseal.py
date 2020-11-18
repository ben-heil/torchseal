"""This file implements a memory leak checker for pytorch"""
import gc
import warnings
from typing import List

import torch


class MemoryLeakError(RuntimeError):
    """ An error to raise when a memory leak is detected """
    pass


class MemoryLeakWarning(RuntimeWarning):
    """ A warning to raise when a memory leak is detected """
    pass


class LeakChecker():
    def __init__(self, error_on_leak: bool=True):
        """
        The initializer for the LeakChecker object

        Arguments
        ---------
        error_on_leak: True if an error should be raised when a leak is
                       detected, False if a warning should be raised
        """

        self.original_tensors = None
        self.error_on_leak = error_on_leak

    def _get_tensors(self) -> List[torch.Tensor]:
        """
        Return the tensors currently being tracked by the garbage collector

        Returns
        -------
        tensors: The current list of tensors
        """
        tensors = []
        # discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
        for object_ in gc.get_objects():
            try:
                if (
                    torch.is_tensor(object_) or
                    hasattr(object_, 'data') and
                    torch.is_tensor(object_)
                ):
                    tensors.append(object_)
            except: #noqa: E722
                # Bare except statements are bad style, but hasattr can raise
                # several weird exceptions
                pass

        return tensors

    def _raise_exception(self, tensors: List[torch.Tensor]):
        # TODO state which tensor(s) are the leaked ones
        message = str(tensors)
        if self.error_on_leak:
            raise MemoryLeakError(message)
        else:
            warnings.warn(message, MemoryLeakWarning)

    def check_leaks(self):
        """
        Keep track of the currently allocated tensors in the calling code
        and raise an exception if memory is leaking

        Raises
        ------
        MemoryLeakError: If there is a memory leak
        MemoryLeakWarning: If there is a memory leak and error_on_leak is False
        """
        tensors = self._get_tensors()
        if self.original_tensors is None:
            tensor_ids = [id(tensor) for tensor in tensors]
            self.original_tensors = set(tensor_ids)
        else:
            leaked_tensors = []
            for tensor in tensors:
                if id(tensor) not in self.original_tensors:
                    leaked_tensors.append(tensor)
                if len(leaked_tensors) > 0:
                    self._raise_exception(leaked_tensors)
