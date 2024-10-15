import json
import time
from pathlib import Path

import seaborn as sns
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

def get_lang_turn_color_mapping():
    return {
        "EN": ["rgb(238, 130, 238)", "rgb(150, 0, 150)"],
        "ES": ["rgb(69, 173, 42)", "rgb(30, 90, 20)"],
        "IT": ["rgb(255, 176, 0)", "rgb(200, 120, 0)"],
        "FR": ["rgb(254, 97, 0)", "rgb(200, 30, 0)"],
        "DE": ["rgb(212, 48, 31)", "rgb(150, 30, 20)"],
        "Avg.": ["rgb(100, 50, 250)", "rgb(0, 0, 200)"],
    }

def create_single_radar_plot(df, lang):
    color_mapping = get_lang_turn_color_mapping()
    radar_plot = px.line_polar(
        df,
        r="score",
        theta="Category",
        line_close=True,
        category_orders={"Category": df["Category"].unique()},
        color="Turn",  # Distinguish between turns
        markers=True,
        color_discrete_map=color_mapping,
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

def single_model_radar_plot(df, model_name):
    df = df.copy()
    df.drop(columns=["Model"], inplace=True)
    # cross-lingual average
    cross_lingual_avg = df.groupby(["Turn", "Category"]).score.mean().reset_index().copy()
    cross_lingual_avg["Language"] = "Avg."
    df = pd.concat([df, cross_lingual_avg], ignore_index=True)

    languages = df["Language"].unique()

    out_path = Path(f"results/{model_name}/")
    out_path.mkdir(exist_ok=True, parents=True)
    
    fix_mathjs_error()
    for lang in languages:
        df_filtered = df[df["Language"] == lang].copy()
        radar_plot = create_single_radar_plot(df_filtered, lang=lang)
        out_file_path = out_path / f"{model_name}_mt_bench_{lang.replace('.', '')}_radar_plot.pdf"
        radar_plot.write_image(str(out_file_path))
        print(f"Saved radar plot for {model_name} in {lang} to {out_file_path}")
    single_model_create_csv(df, model_name, out_path)
    

def single_model_create_csv(df, model_name, out_path):
    # add category average
    category_avg = df.groupby(["Turn", "Language"]).score.mean().reset_index().copy()
    category_avg["Category"] = "Avg."
    df = pd.concat([df, category_avg], ignore_index=True)

    # create csv
    pivot = df.pivot(index=["Turn", "Language"], columns="Category", values="score")
    pivot = pivot.reset_index()
    pivot.to_csv(out_path / f"{model_name}_mt_bench_x.csv", index=False, float_format="%.2f")
    print(pivot)

def multi_model_radar_plot(df):
    original_model_names = df["Model"].unique()
    df.Model = df.Model.apply(map_model_names)
    # cross-lingual average and average over turns
    cross_lingual_avg = df.groupby(["Model", "Category"]).score.mean().reset_index().copy()
    cross_lingual_avg["Language"] = "Avg."
    df = df.groupby(["Model", "Category", "Language"]).score.mean().reset_index().copy()
    df = pd.concat([df, cross_lingual_avg], ignore_index=True)
    languages = df["Language"].unique()
    model_abbrevs = [model_name.split("-")[0] for model_name in df["Model"].unique()]
    model_str = "_".join(model_abbrevs).replace(" ", "_").replace(".", "")

    out_path = Path(f"results/multi/{model_str}/")
    out_path.mkdir(exist_ok=True, parents=True)
    original_model_names.tofile(out_path / "original_model_names.txt", sep="\n")

    fix_mathjs_error()
    for lang in languages:
        df_filtered = df[df["Language"] == lang].copy()
        radar_plot = create_multi_model_radar_plot(df_filtered)
        out_file_path = out_path / f"mt_bench_{lang.replace('.', '')}_radar_plot.pdf"
        radar_plot.write_image(str(out_file_path))
        print(f"Saved radar plot for {model_str} in {lang} to {out_file_path}")
    multi_model_create_csv(df, model_str, out_path)

def multi_model_create_csv(df, model_str, out_path):
    # add category average
    category_avg = df.groupby(["Model", "Language"]).score.mean().reset_index().copy()
    category_avg["Category"] = "Avg."
    df = pd.concat([df, category_avg], ignore_index=True)

    # create csv
    pivot = df.pivot(index=["Model", "Language"], columns="Category", values="score")
    pivot = pivot.reset_index()
    pivot.to_csv(out_path / f"{model_str}_mt_bench_x.csv", index=False, float_format="%.2f")
    print(pivot)


def create_multi_model_radar_plot(df):
    model_color_mapping = get_model_color_mapping()
    radar_plot = px.line_polar(
        df,
        r="score",
        theta="Category",
        line_close=True,
        category_orders={"Category": df["Category"].unique()},
        color="Model",
        markers=True,
        color_discrete_map=model_color_mapping,
    )
    radar_plot.update_layout(
        height=500,
        width=600,
        legend_font_size=16,
        font=dict(
            size=18,
        ),
        margin=dict(l=0, r=10, t=10, b=30),
        legend=dict(
            x=0.85,
            y=1.1,
            xanchor="center",
            yanchor="middle",
            bgcolor="rgba(0,0,0,0)"  # Transparent background
        ),
        polar=dict(
            radialaxis=dict(
                range=[1, 10],
            )
        ),
    )
    return radar_plot

def map_model_names(model_name):
    model_name_lower = model_name.lower()
    return get_model_name_mapping().get(model_name_lower, model_name)


def get_model_color_mapping():
    model_names = get_model_name_mapping()
    palette = sns.color_palette(None, len(model_names))
    return {model: f"rgb{tuple(int(x * 255) for x in color)}" for model, color in zip(model_names.values(), palette)}

def get_model_name_mapping():
    return {
        'opengptx-7B-24EU-4T-chat'.lower(): "Ours (Instruct)",
        "salamandra-7b-instruct": "Salamandra-7B-Instruct",
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-path", nargs="+", type=Path, help="List of directories containing json result files", required=True)
    parser.add_argument("--mode", choices=["single", "multi", "both"], required=True)
    args = parser.parse_args()

    data = []
    for dir_path in args.dir_path:
        file_path = dir_path / (dir_path.name + "_judgments.json")
        with file_path.open() as json_file:
            results = json.load(json_file)["results"]
            results = {key: (value | extract_key_info(key)) for key, value in results.items()}
            model = dir_path.name
            for key in results.keys():
                results[key]["Model"] = model
                data.append(results[key])

    df = pd.DataFrame.from_records(data)
    df.category = df.category.str.capitalize()
    df.language = df.language.str.upper()
    df.rename(columns={"turn": "Turn", "language": "Language", "category": "Category"}, inplace=True)

    if args.mode == "multi" or args.mode == "both":
        multi_model_radar_plot(df)
    if args.mode == "single" or args.mode == "both":
        for (model_name, model_df) in df.groupby("Model"):
            single_model_radar_plot(model_df, model_name)
