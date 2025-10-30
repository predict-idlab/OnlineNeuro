import pandas as pd
import requests


def plot_flask(sample: pd.DataFrame, plot_type: str, post_url: str) -> None:
    """
    Send data to plot to Flask server.
    @param sample: pd.DataFrame with features and observations
    @param plot_type: string specifying the plot type to serve
    @param post_url: the url specifying which plot to update
    @return:
    """
    json_message = dict()
    json_message["data"] = sample.to_json(orient="records")
    json_message["plot_type"] = plot_type

    response = requests.post(post_url + "/update_data", json=json_message)
    # Check the response
    if response.status_code == 200:
        print(response.json())
    else:
        print(
            "Error occured while plotting to flask:",
            response.status_code,
            response.text,
        )
