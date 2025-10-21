import pandas as pd
from itertools import chain



def append_series(series, value):  # appends value at the end of series
    """Append a value to a pandas series

    Args:
        series (pandas series): series to append value to
        value (same type as other records in the series): value to append

    Returns:
        pandas series: original series with value appended
    """

    return pd.concat([series, pd.Series([value])])


def concatenate_df(df1, df2, ignore_index=False):
    """Concatenate two dataframes despite pandas update

    Args:
        df1 (pandas DataFrame): first dataframe
        df2 (pandas DataFrame): second dataframe
        ignore_index (bool, optional): resets index if True . Defaults to False.

    Returns:
        pandas DataFrame: concatenated dataframe
    """

    if df2.empty:
        output = df1.copy()
    elif df1.empty:
        output = df2.copy()
    elif ignore_index:
        output = pd.concat([df1, df2], ignore_index=True)
    else:
        output = pd.concat([df1, df2])

    return output


def append_df(df, rowdict):  # appends a row described by a dictionnary to df
    """Append a row to a pandas dataframe

    Args:
        df (pandas DataFrame): dataframe to append row to
        rowdict (dictionnary): dictionnary describing the row to append

    Returns:
        pandas DataFrame: dataframe with row appended
    """
    
    return concatenate_df(df, pd.DataFrame.from_records([rowdict]), ignore_index=True)


def dict_union(*args):
    """Union of dictionaries
    Args:
        *args (dictionnaries): dictionnaries to union
    Returns:
        dictionnary: union of dictionnaries
    """
    # Use chain.from_iterable to flatten the list of items from each dictionary
    # and then create a new dictionary from the flattened items
    # This will handle duplicate keys by keeping the last value encountered
    # in the input dictionaries
    # Note: If you want to handle duplicates differently (e.g., keep the first value),
    # you can modify the logic accordingly.
    # This will keep the last value encountered for each key
    # in the input dictionaries
    # If you want to keep the first value, you can use a different approach
    # such as using a defaultdict or checking if the key is already in the result
    # before adding it.
    # This will keep the last value encountered for each key
    # in the input dictionaries
    return dict(chain.from_iterable(d.items() for d in args))


def flatten_list(array_collection):
    """    Flattens a list or dictionnary of lists into a single list. 
    Args:
        array_collection (list of lists): The list of lists to flatten.
    Returns:
        list: A single flattened list containing all elements from the sublists.    

    """
    return [item for sublist in array_collection for item in sublist]