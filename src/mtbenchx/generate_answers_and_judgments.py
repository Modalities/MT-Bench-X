import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from mtbenchx.evalset import EvalSet
from mtbenchx.fastchat.llm_judge.common import (
    NEED_REF_CATS,
    check_data,
    load_judge_prompts,
    load_model_answers,
    load_questions,
    play_a_match_single,
)
from mtbenchx.fastchat.llm_judge.gen_judgment import make_judge_single, make_match_single
from mtbenchx.fastchat.llm_judge.gen_model_answer import run_eval
from mtbenchx.fastchat.llm_judge.utils import str_to_torch_dtype


def gen_mulilingual_model_answers(args, bench_names: List[str], model_id: str):
    """Adapted from mtbenchx.fastchat/llm_judge/gen_model_answer.py"""
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    eval_sets = []
    print("Will create model answers for:")
    for bench_name in bench_names:
        answer_file = Path(f"results/{model_id}/{bench_name}/model_answer/{model_id}.jsonl")
        if not answer_file.exists():
            answer_file.parent.mkdir(parents=True, exist_ok=True)
            answer_file.touch()
        eval_set = EvalSet(question_file=f"data/{bench_name}/question.jsonl", answer_file=answer_file)
        eval_sets.append(eval_set)
        print(eval_set)

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        eval_sets=eval_sets,
        question_begin=args.question_begin,
        question_end=args.question_end,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        model_id_postfix=args.model_id_postfix,
        revision=args.revision,
    )
    for eval_set in eval_sets:
        reorg_answer_file(eval_set.answer_file)


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    Path(answer_file).parent.mkdir(parents=True, exist_ok=True)
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w", encoding="utf-8") as fout:
        for qid in qids:
            fout.write(answers[qid])


def run_judgment_jobs(args, bench_names, model_id):
    """Adaped from mtbenchx.fastchat/llm_judge/gen_judgment.py"""
    matches = []
    match_stats = []
    for bench_name in bench_names:
        bench_matches, match_stat = _prepare_judgement_run(args, bench_name, model_id)
        matches += bench_matches
        match_stats.append(match_stat)

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(pd.DataFrame.from_dict(match_stats).to_string())
    if not args.quiet:
        input("Press Enter to confirm...")

    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_single(match)
    else:

        def play_a_match_wrapper(match):
            play_a_match_single(match)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(executor.map(play_a_match_wrapper, matches), total=len(matches)):
                pass
        executor.shutdown(wait=True)

def _prepare_judgement_run(args, bench_name, model_id) -> tuple[list, dict]:
    question_file = f"data/{bench_name}/question.jsonl"
    answer_dir = f"results/{model_id}/{bench_name}/model_answer"
    ref_answer_dir = f"data/{bench_name}/reference_answer"

    # Load questions
    questions = load_questions(question_file, args.question_begin, args.question_end)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # Load judge
    judge_file = f"data/{bench_name}/{args.judge_file_name}"
    judge_prompts = load_judge_prompts(judge_file)

    baseline_model = None
    judges = make_judge_single(args.judge_model, judge_prompts)

    output_file = f"results/{model_id}/{bench_name}/model_judgment/{args.judge_model}_single.jsonl"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    print(f"Will use default result file {Path(output_file).absolute()}")
    models = [model_id]
    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_single(question_default, models, model_answers, judges["default"], output_file, baseline_model)
    matches += make_match_single(
        question_math,
        models,
        model_answers,
        judges["math"],
        output_file,
        baseline_model,
        ref_answers,
    )
    matches += make_match_single(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        output_file,
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_single(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        output_file,
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = bench_name
    match_stat["judge"] = args.judge_model
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file
    return matches, match_stat
