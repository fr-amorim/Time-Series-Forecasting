import pandas as pd
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class Dataset:
    '''
    A class to improve interaction with multivariate time series analysis and forecasting
    '''
    name:str
    sales_table: pd.Series
    id_names: str = None
    ids: pd.MultiIndex = None
    periods: pd.MultiIndex = None
    gap: int = 1
    fhorizon: int = 1
    pred_moments: pd.Index = None
    rolling_preds:pd.DataFrame = None
    
    def __post_init__(self):
        self._post_update_sales_table()
    
    def update_ids(self)->None:
        self.ids = self.sales_table.index.droplevel(-1).unique()
    
    def _post_update_sales_table(self)->None:
        self.update_ids()
        self.update_periods()
    
    def setup_backtest(self
                    , pred_start = None
                    , pred_end = None
                    , nlast:int=None
                    
                    , nperiods = None
                    , stride:int = 1
                    
                    , ntest_folds:int = 1
                    , pred_moments = None
                ):
        if pred_moments is not None:
            self.pred_moments = pred_moments
            return None
        pred_start = self.periods[-nlast] if nlast else pred_start
        pred_end = pred_end if pred_end else self.periods[-1]
        self.pred_moments = self.periods[np.where((self.periods>=pred_start) & (self.periods<=pred_end))][::stride]
        if nperiods:
            self.pred_moments = self.pred_moments[:nperiods]
    
    def update_periods(self):
        self.periods =  self.sales_table.unstack(-1).columns.sort_values()
    
    @property
    def levels(self) -> str:
        return list(self.sales_table.index.names)
    
    @property
    def totals(self)->pd.Series:
        return self.sales_table.groupby(self.levels[:-1]).sum()
    
    @property
    def describe(self)->pd.Series:
        return self.totals.describe()
    
    
    @property
    def series(self)->pd.Series:
        for series in self.sales_table.groupby(self.levels[:-1]):
            yield series #this provides the series id and the series itself
    
    @property
    def n_series(self)->int:
        return self.ids.size
    
    @property
    def series_values(self)->np.ndarray:
        for (id_, st_values) in zip(
            self.ids
            , self.sales_table.unstack(-1)[self.periods].values
        ):
            yield id_, st_values
    
    @property
    def dataset_values(self)->np.ndarray:
        return self.sales_table.unstack(-1)[self.periods].values
    
    @property
    def train_folds_dataset(self):
        #this should have train and test
        for pred_moment in self.pred_moments:
            yield self.sales_table.loc[
                lambda x: x.index.get_level_values(-1)<=pred_moment
            ]
    
    def backtest(self, model):
        for sales_table in self.train_folds_dataset:
            print(sales_table.unstack(-1).values)