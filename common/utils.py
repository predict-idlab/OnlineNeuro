import json
from typing import Union, Dict, Any, List
from pathlib import Path


def load_json(json_file: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
    """
    @param json_file: path to the json file
    @return: loaded json, either dict or list.
    """
    with open(json_file) as f:
        return json.load(f)
