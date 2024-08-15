import json
import warnings
from pathlib import Path

import pandas as pd


def get_category(question_id):
    CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]
    numeric_question_id = int(question_id.split("_")[0]) if isinstance(question_id, str) else question_id
    return CATEGORIES[(numeric_question_id - 81) // 10]


def collect_model_specific_judgments(model_id: str, save_path: str | None = None, judgment_filename: str = "gpt-4_single.jsonl"):
    """
    Collects judgments for a specific model from all evaluation languages and saves them to a JSON file.
    The collection is done by reading the judgment files for each evaluation language and extracting the judgments for the given model.
    """
    results = {}
    non_empty_eval_languages = []
    judge = ""
    categories = []
    for eval_language in ["EN", "DE", "FR", "IT", "ES"]:
        file_path = Path(f"data/mt_bench_{eval_language}/model_judgment/{judgment_filename}")
        if not file_path.exists():
            continue
        with file_path.open() as f:
            entries = [json.loads(line) for line in f]
        data = pd.DataFrame(entries)
        data = data[data.model == model_id]

        data["category"] = data.question_id.apply(get_category)

        # extract evaluation configuration
        if len(data) == 0:
            continue
        assert data.model.unique().size == 1, f"Multiple models found in {file_path}: {data.model.unique()}"
        model = data.model.unique()[0]
        judges = data.judge.apply(lambda x: x[0]).unique()
        assert judges.size == 1
        judge = judges[0]

        # assert validity
        is_within_range = (data.score >= 1) & (data.score <= 10)
        if not is_within_range.all():
            warnings.warn(f"Number of scores out of range for eval language {eval_language}: {len(is_within_range) - is_within_range.sum()}")
        if not len(data) % 160 == 0:
            warnings.warn(f"Number of duplicates found within eval lanauge {eval_language}: {data.question_id.duplicated().sum()}")
        if len(data) > 160:
            warnings.warn("Duplicate judgments found")
            raise NotImplementedError("Handling multiple rounds of judgments not implemented yet. TODO: Standard deviation impl.")

        data.drop(columns=["model", "judge", "user_prompt", "judgment", "tstamp", "question_id"], inplace=True)
        mean_data = data.groupby(["category", "turn"]).mean()
        mean_data.reset_index(inplace=True)
        index = "mt_bench_" + eval_language + "_" + mean_data["category"].str.lower() + "_" + mean_data["turn"].astype(str)
        mean_data.index = pd.Index(index)
        # mean_data.rename(columns={"score": "acc"}, inplace=True)
        categories = mean_data.category.unique().tolist()
        mean_data.drop(columns=["category", "turn"], inplace=True)
        non_empty_eval_languages.append(eval_language)
        results.update(mean_data.to_dict("index"))

    config = dict(model=model_id, judge=judge, range="1 to 10", categories=categories, eval_languages=non_empty_eval_languages)
    judgment_results = dict(results=results, config=config)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving judgment results to {save_path}")
        json.dump(judgment_results, open(save_path, "w"), indent=4)
    return judgment_results


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        help="Model ID to collect judgments for",
        required=True,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save judgment results",
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    collect_model_specific_judgments(args.model_id, save_path=args.save_path)
