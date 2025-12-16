# api/backend/utils/string_utils.py
import re
from typing import Any


def convert_strings_to_numbers(data: Any) -> Any:
    """
    Recursively convert string representations of numbers and booleans into
    their corresponding Python types.

    Supports nested dictionaries and lists. The following conversions are applied:
    - ``"true"`` → ``True``
    - ``"false"`` → ``False``
    - Numeric strings without a decimal point → ``int``
    - Numeric strings with a decimal point → ``float``
    If a string cannot be converted, it is returned unchanged.

    Parameters
    ----------
    data : Any
        A nested structure containing dictionaries, lists, strings, or any
        other data types. Conversion is applied recursively to all strings.

    Returns
    -------
    Any
        The input data with all convertible string elements replaced by their
        boolean, integer, or float equivalents.
    """
    if isinstance(data, dict):
        # Recursively apply the function to each value in the dictionary
        return {key: convert_strings_to_numbers(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Recursively apply the function to each element in the list
        return [convert_strings_to_numbers(item) for item in data]
    elif isinstance(data, str):
        # Normalize the string to lowercase for boolean checks
        lower_data = data.lower()

        # Handle boolean strings 'true' and 'false'
        if lower_data == "true":
            return True
        elif lower_data == "false":
            return False

        # Try to convert the string to an int or float
        try:
            if "." in data:  # Convert to float if the string contains a decimal point
                return float(data)
            else:
                return int(data)  # Convert to int otherwise
        except ValueError:
            return data  # Return the original string if conversion fails
    else:
        # Return the value if it is not a string, dict, or list
        return data


def camel_to_snake(name: str) -> str:
    """
    Convert a CamelCase or camelCase string to snake_case.

    This function inserts underscores at the appropriate boundaries between
    lowercase and uppercase character sequences and returns the fully
    lowercased snake_case representation.

    Parameters
    ----------
    name : str
        The input string in CamelCase or camelCase format.

    Returns
    -------
    str
        The converted string in snake_case format.
    """
    # First pass: handle transitions like "CamelCase" → "Camel_Case"
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Second pass: handle transitions like "camelCASE" → "camel_CASE"
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
