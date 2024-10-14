import argparse
import json
import os
import warnings

from mtbenchx.collect_model_specific_judgments import collect_model_specific_judgments
from mtbenchx.generate_answers_and_judgments import gen_mulilingual_model_answers, run_judgment_jobs


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True, help="A custom name for the model.")
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument("--question-end", type=int, help="A debug option. The end index of questions.")
    parser.add_argument("--save-path", type=str, default=None, help="The final save file to store the model-specific cross-lingual eval results.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument("--num-gpus-total", type=int, default=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")), help="The total number of GPUs.")
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument("--model-id-postfix", type=str, help="Postfix to identify different checkpoints or languages", default="")
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--eval-languages",
        type=str,
        nargs="+",
        default=None,
        help="A list of languages to be evaluated",
    )
    parser.add_argument(
        "--judge-file-name",
        type=str,
        default="judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--parallel", type=int, default=1, help="The number of concurrent API calls.")
    parser.add_argument("--quiet", action="store_true", help="Do not ask to start OpenAI evaluation -  a step which induces costs.")
    args = parser.parse_args()
    print("Arguments:")
    print(json.dumps(vars(args), indent=4))
    return args


def main():
    args = setup_parser()
    if args.num_choices > 1:
        warnings.warn("Warning: num_choices > 1 is not supported yet when judging the generated answers. Only the first is used!")

    print(f"Generating model answers and judgments across {args.num_gpus_total} GPU(s)...")

    model_id = f"{args.model_id}-{args.model_id_postfix}" if args.model_id_postfix != "" else f"{args.model_id}"
    bench_names = [f"mt_bench_{eval_language}" for eval_language in args.eval_languages]
    gen_mulilingual_model_answers(args=args, bench_names=bench_names, model_id=model_id)
    run_judgment_jobs(args, bench_names, model_id)

    if args.save_path is None:
        args.save_path = f"results/{model_id}/{model_id}_judgments.json"
    collect_model_specific_judgments(model_id, save_path=args.save_path)


if __name__ == "__main__":
    main()
