from typing import Generator, Optional, Tuple, Union
from sktime.forecasting.model_selection._split import BaseSplitter, SPLIT_GENERATOR_TYPE, ACCEPTED_Y_TYPES, _check_fh
import pandas as pd
import numpy as np

class CustomSplitter(BaseSplitter):
    '''
    Class that performs and expanding rolling window where the training windows stop at the cutoff points
    '''
    def __init__(
        self,
        cutoffs,
        fh,
    ) -> None:
        self.cutoffs = cutoffs
        self.fh = fh

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        n_timepoints = y.shape[0]
        fh = _check_fh(self.fh)
        max_fh = fh.max()
        max_cutoff = np.max(self.cutoffs)

        for cutoff in self.cutoffs:
            train_start=0 #fix this to be the first not null value
            split_point = y.get_loc(y[y <= cutoff][-1])
            training_window = self._get_train_window(
                y=y
                , train_start=train_start
                , split_point=split_point + 1
            )
            
            test_window = fh.to_numpy() + split_point
            yield training_window, test_window

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        """Return the number of splits.
        For this splitter the number is trivially equal to
        the number of cutoffs given during instance initialization.
        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split
        Returns
        -------
        n_splits : int
            The number of splits.
        """
        return len(self.cutoffs)

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points in .iloc[] context.
        This method trivially returns the cutoffs given during instance initialization,
        in case these cutoffs are integer .iloc[] friendly indices.
        The only change is that the set of cutoffs is sorted from smallest to largest.
        When the given cutoffs are datetime-like,
        then this method returns corresponding integer indices.
        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split
        Returns
        -------
        cutoffs : 1D np.ndarray of int
            iloc location indices, in reference to y, of cutoff indices
        """
        return np.argwhere(y.index.isin(self.cutoffs)).flatten()