def array_to_list_of_dicts(array, column_names):
    """
    Convert a np.ndarray into a list of dictionaries
    @param array:
    @param column_names:
    @return:
    """
    if array.shape[1] != len(column_names):
        raise ValueError(
            "Number of columns in array must match the number of column names."
        )

    return [dict(zip(column_names, row)) for row in array]
