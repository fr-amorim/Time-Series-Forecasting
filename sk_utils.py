from typing import Callable, Dict
from warnings import simplefilter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import pandas as pd
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_selection._split import BaseSplitter
from joblib import delayed, Parallel
import itertools
import pandas_utils as pdu
from typing import List, Union, Dict

def get_windows(y, cv):
    """Generate windows"""
    train_windows = []
    test_windows = []
    for i, (train, test) in enumerate(cv.split(y)):
        train_windows.append(train)
        test_windows.append(test)
    return train_windows, test_windows

def plot_windows(y, train_windows, test_windows, title=""):
    """Visualize training and test windows"""

    simplefilter("ignore", category=UserWarning)

    def get_y(length, split):
        # Create a constant vector based on the split for y-axis."""
        return np.ones(length) * split

    n_splits = len(train_windows)
    n_timepoints = len(y)
    len_test = len(test_windows[0])

    train_color, test_color = sns.color_palette("colorblind")[:2]

    fig, ax = plt.subplots(figsize=plt.figaspect(0.3))

    for i in range(n_splits):
        train = train_windows[i]
        test = test_windows[i]
        
        ax.plot(
            np.arange(n_timepoints), get_y(n_timepoints, i), marker="o", c="lightgray"
        )
        
        ax.plot(
            train,
            get_y(len(train), i),
            marker="o",
            c=train_color,
            label="Window",
        )
        ax.plot(
            test,
            get_y(len_test, i),
            marker="o",
            c=test_color,
            label="Forecasting horizon",
        )
    
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(np.arange(y.index.size))
    ax.set_xticklabels(y.index)
    #print(labels)
    ax.set(
        title=title,
        ylabel="Window number",
        xlabel="Time",
    )
    # remove duplicate labels/handles
    handles, labels = [(leg[:2]) for leg in ax.get_legend_handles_labels()]
    ax.legend(handles, labels);

def parse_backtests(backtest:pd.DataFrame, forecaster:BaseForecaster)->pd.DataFrame:
    fold_dict = dict(enumerate(list(map(lambda x: x.index[-1], backtest.y_train))))
    return (pd.concat([preds
                .rename('pred')
                .to_frame()
                .assign(fold=fold)
            for fold,preds in enumerate(backtest.y_pred)]
        )
        .assign(
            actual = pd.concat(backtest.y_test.values).values
            , predicted_from = lambda x: x.fold.map(fold_dict)
            , gap = lambda x: x.groupby('fold').cumcount()+1
            , method = str(forecaster)
        )
        .set_index('fold gap method'.split(), append=True)
        .rename_axis('year fold gap method'.split())
        .reorder_levels('fold gap method year'.split())
    )

def parse_backtests_multi(backtest:pd.DataFrame
                        , forecaster:BaseForecaster
                    )->pd.DataFrame:
    fold_dict = dict(enumerate(list(map(lambda x: x.index[-1], backtest.y_train))))
    return (
    pd.concat([preds.assign(fold=fold)
                for fold,preds in enumerate(backtest.y_pred)]
    )
    .assign(
        gap = lambda x: x.groupby('fold').cumcount()+1
        , method = str(forecaster)
    )
    .set_index('fold gap method'.split(), append=True)
    .stack()
    .rename_axis('year fold gap method id'.split())
    .reorder_levels('id fold gap method year'.split())
    .rename('pred')
    #.reorder_levels('id fold gap method year'.split())
    .to_frame()
    .assign(
        predicted_from = lambda x: x.reset_index('fold').fold.map(fold_dict).values
    )
    .join(
        pd.concat(backtest.y_test.values).stack().rename_axis('year id'.split()).rename('actual').drop_duplicates()
    )
    .reorder_levels('id fold gap method year'.split())
    )

def plot_predictions(y:pd.DataFrame
                    , id_:str
                    , cross_val_preds:pd.DataFrame
                    , gap:int
                    )->None:
    ax = cross_val_preds.loc[id_, :, gap].pred.unstack('method').droplevel('fold').plot(style='o--')
    y[id_].plot(style='x-', ax=ax)
    plt.ylim(0)
    plt.show()

def backtest(y:pd.DataFrame
            , cv:BaseSplitter
            , forecaster:BaseForecaster
            , target_var:Union[str, list]
        )->List[pd.DataFrame, Dict[str, BaseForecaster]]:
    backtest = evaluate(
            forecaster=forecaster,
            cv=cv,
            y=y[target_var],
            X=None,
            strategy="refit",
            scoring=None,
            return_data=True,
        )
    if forecaster.get_tags()['scitype:y'] in ['multivariate','both']: #univariate predictions
        return (parse_backtests_multi(backtest, forecaster)
            , ('multivariate', {str(forecaster) : forecaster}) #the fitted forecaster
        )
    else: #multivariate predictions
        return (parse_backtests(backtest, forecaster).assign(id = target_var).set_index('id', append=True).pipe(pdu.pull_to_first_index, 'id')
            , (target_var, {str(forecaster) : forecaster}) #the fitted forecaster
        )

def full_backtest(
        forecasters : List[BaseForecaster]
        , backtest_function: Callable
        , y : pd.DataFrame
        , parallel : Parallel
    )->list(pd.DataFrame, pd.DataFrame):
    multivariate_forecasters = [x for x in forecasters if x.get_tags()['scitype:y'] in ['multivariate','both'] ]
    univariate_forecasters = [x for x in forecasters if x.get_tags()['scitype:y'] not in ['multivariate','both'] ]
    univariate_forecasters_inputs = list(itertools.product(univariate_forecasters, y))
    multivariate_forecasters_inputs = list(itertools.product(multivariate_forecasters, [list(y)] ))
    
    #perform backtests in parallel for each combination of forcaster and series
    out = parallel(
            delayed(backtest_function)(*params) 
            for params in univariate_forecasters_inputs + multivariate_forecasters_inputs
    )
    
    fitted_forecasters = dict.fromkeys(list(y) + ['multivariate'], {})
    
    forecasters = list(map(lambda x: x[1], out))
    preds = list(map(lambda x: x[0], out))

    cross_val_preds = pd.concat(preds)
    
    for k,v in forecasters:
        fitted_forecasters[k].update(v)

    #a pandas dataframe with all the fitted forecasters
    fitted_forecasters = pd.DataFrame(
        fitted_forecasters
    ).T
    return fitted_forecasters, cross_val_preds