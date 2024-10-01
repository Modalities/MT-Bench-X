import json
import time
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


def fix_mathjs_error():
    file_path = Path("tmp_figure.pdf")
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(str(file_path), format="pdf")
    time.sleep(2)
    file_path.unlink()


def create_rader_plot(df):
    color_mapping = {
        "EN": ["rgb(238, 130, 238)", "rgb(150, 0, 150)"],
        "ES": ["rgb(69, 173, 42)", "rgb(30, 90, 20)"],
        "IT": ["rgb(255, 176, 0)", "rgb(200, 120, 0)"],
        "FR": ["rgb(254, 97, 0)", "rgb(200, 30, 0)"],
        "DE": ["rgb(212, 48, 31)", "rgb(150, 30, 20)"],
        "Avg.": ["rgb(100, 50, 250)", "rgb(0, 0, 200)"],
    }
    radar_plot = px.line_polar(
        df,
        r="score",
        theta="Category",
        line_close=True,
        category_orders={"Category": df["Category"].unique()},
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
        margin=dict(l=25, r=10, t=10, b=30),
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


if __name__ == "__main__":
    import argparse

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
    df.rename(columns={"turn": "Turn", "language": "Language", "category": "Category"}, inplace=True)

    # cross-lingual average
    cross_lingual_avg = df.groupby(["Turn", "Category"]).score.mean().reset_index().copy()
    cross_lingual_avg["Language"] = "Avg."
    df = pd.concat([df, cross_lingual_avg], ignore_index=True)

    languages = df["Language"].unique()

    model_name = args.file_path.stem.rsplit("_", 1)[0]
    out_path = Path(f"visualization/{model_name}/")
    out_path.mkdir(exist_ok=True, parents=True)
    fix_mathjs_error()
    for lang in languages:
        df_filtered = df[df["Language"] == lang].copy()
        radar_plot = create_rader_plot(df_filtered)
        out_file_path = out_path / f"{model_name}_mt_bench_{lang.replace('.', '')}_radar_plot.pdf"
        radar_plot.write_image(str(out_file_path))

    # add category average
    category_avg = df.groupby(["Turn", "Language"]).score.mean().reset_index().copy()
    category_avg["Category"] = "Avg."
    df = pd.concat([df, category_avg], ignore_index=True)

    # create csv
    pivot = df.pivot(index=["Turn", "Language"], columns="Category", values="score")
    pivot = pivot.reset_index()
    pivot.to_csv(out_path / f"{model_name}_mt_bench_x.csv", index=False, float_format="%.2f")
