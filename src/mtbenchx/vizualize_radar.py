import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots


def extract_key_info(key: str):
    fields = key.split("_")
    return {"language": fields[-3], "category": fields[-2], "turn": fields[-1]}


def create_rader_plot(df):
    color_mapping = {
        "EN": ["rgb(238, 130, 238)", "rgb(150, 0, 150)"],
        "ES": ["rgb(69, 173, 42)", "rgb(30, 90, 20)"],
        "IT": ["rgb(255, 176, 0)", "rgb(200, 120, 0)"],
        "FR": ["rgb(254, 97, 0)", "rgb(200, 30, 0)"],
        "DE": ["rgb(212, 48, 31)", "rgb(150, 30, 20)"],
    }
    radar_plot = px.line_polar(
        df,
        r="score",
        theta="category",
        line_close=True,
        category_orders={"category": ["Coding", "Extraction", "Humanities", "Math", "Reasoning", "Roleplay", "Stem", "Writing"]},
        color="Turn",  # Distinguish between turns
        markers=True,
        color_discrete_sequence=color_mapping[lang],
    )

    radar_plot.update_layout(
        height=500,
        width=600,
        legend_font_size=18,
        font=dict(
            size=18,
        ),
        margin=dict(l=0, r=0, t=10, b=60),
        legend=dict(
            x=1.05,
            y=0.965,
            xanchor="center",
            yanchor="middle",
        ),
        polar=dict(
            radialaxis=dict(
                range=[1, 10],
            )
        ),
    )
    for trace in radar_plot.data:
        if trace.name == "2":
            trace.line.dash = "dot"
    return radar_plot


def fix_mathjax():
    import time

    import plotly.express as px

    file_Name = "some_Figure.pdf"
    fig_Throw_Away = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig_Throw_Away.write_image(file_Name, format="pdf")
    time.sleep(0.1)

    # delete the dummy file again
    path_Cwd = Path.cwd()
    file_2_Del = path_Cwd / file_Name
    file_2_Del.unlink(missing_ok=True)

    # ------------------------ end first time ------------------------ #ndError exceptions will be ignored (same behavior as the POSIX rm -f command).
    file_2_Del.unlink(missing_ok=True)

    # ------------------------ end first time ------------------------ #

    # now run your actual code to generate the plot that you like


if __name__ == "__main__":
    import argparse

    fix_mathjax()
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", required=True, type=Path, help="")
    args = parser.parse_args()
    with args.file_path.open() as json_file:
        data = json.load(json_file)

    data = data["results"]
    data = {key: (value | extract_key_info(key)) for key, value in data.items()}
    df = pd.DataFrame.from_dict(list(data.values()))
    df.category = df.category.str.capitalize()
    df.language = df.language.str.upper()
    df.rename(columns={"turn": "Turn"}, inplace=True)

    languages = df["language"].unique()

    for lang in languages:
        df_filtered = df[df["language"] == lang].copy()
        radar_plot = create_rader_plot(df_filtered)
        radar_plot.write_image(f"mt_bench_{lang}_radar_plot.pdf")
