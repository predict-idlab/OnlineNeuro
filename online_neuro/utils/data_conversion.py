# online_neuro/utils/data_conversion.py
import numpy as np


def array_to_list_of_dicts(
    array: np.ndarray, column_names: list | np.ndarray
) -> list[dict]:
    """
    Convert a np.ndarray into a list of dictionaries
    Parameters
    ----------
    array : np.ndarray
        The input array to be converted.
    column_names : list | np.ndarray
        The names of the columns corresponding to the array's second dimension.

    Returns
    -------
    list[dict]
        The list of dictionaries representing the array.
    Raises
    ------
    ValueError
        If ``array`` shape does not match length of colum_names.
    """
    if array.shape[1] != len(column_names):
        raise ValueError(
            "Number of columns in array must match the number of column names."
        )

    return [dict(zip(column_names, row)) for row in array]
