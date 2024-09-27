import json
from typing import Union, Dict, Any, List


def load_json(json_file: str) -> Union[Dict[str, Any], List[Any]]:
    """
    @param json_file: path to the json file
    @return: loaded json, either dict or list.
    """
    with open(json_file) as f:
        return json.load(f)
