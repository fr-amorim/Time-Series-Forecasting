import pandas as pd

def pull_to_first_index(self, indexes:list)->pd.DataFrame:
    '''
    Pulls a given index to the first position while maitaining the relative order of the rest
    '''
    indexes = [indexes] if isinstance(indexes, str) else indexes
    original = self.index.names
    remaining = [x for x in original if x not in indexes]
    return self.reorder_levels(indexes + remaining).sort_index()