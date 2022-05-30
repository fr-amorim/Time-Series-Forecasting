import sktime.performance_metrics.forecasting._functions as eval_functions
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
AVAILABLE_METRICS = [
    #'relative_loss',
    'mean_linex_error',
    'mean_asymmetric_error',
    #  'mean_absolute_scaled_error',
    #  'median_absolute_scaled_error',
    #  'mean_squared_scaled_error',
    #  'median_squared_scaled_error',
    'mean_absolute_error',
    'mean_squared_error',
    'median_absolute_error',
    'median_squared_error',
    'geometric_mean_absolute_error',
    'geometric_mean_squared_error',
    'mean_absolute_percentage_error',
    'median_absolute_percentage_error',
    'mean_squared_percentage_error',
    'median_squared_percentage_error',
    #  'mean_relative_absolute_error',
    #  'median_relative_absolute_error',
    #  'geometric_mean_relative_absolute_error',
    #  'geometric_mean_relative_squared_error'
]


def get_metrics(df):
    return {x : getattr(eval_functions, x)(df.actual.values, df.pred.values) 
            for x in AVAILABLE_METRICS}