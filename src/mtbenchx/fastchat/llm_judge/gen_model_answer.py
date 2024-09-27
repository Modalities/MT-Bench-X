# This file was modified and originally stemmed from FastChat.
# For more information, visit: https://github.com/lm-sys/FastChat
# Distributed under the Apache License, Version 2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for more details.

# This file was modified and originally stemmed from FastChat.
# For more information, visit: https://github.com/lm-sys/FastChat
# Distributed under the Apache License, Version 2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for more details.

"""
Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""

import argparse
import datetime
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ray
import shortuuid
import torch
from tqdm import tqdm
from transformers import StoppingCriteriaList

from mtbenchx.evalset import EvalSet
from mtbenchx.fastchat.llm_judge.common import load_questions, temperature_config
from mtbenchx.fastchat.model.model_adapter import get_conversation_template, load_model


def run_eval(
    model_path: str,
    model_id: str,
    eval_sets: List[EvalSet],
    question_begin: int,
    question_end: int,
    max_new_token: int,
    num_choices: int,
    num_gpus_per_model: int,
    num_gpus_total: int,
    max_gpu_memory: str,
    dtype: torch.dtype,
    model_id_postfix: str,
    revision: str,
):
    questions = []
    answer_files = []
    for eval_set in eval_sets:
        questions.extend(load_questions(eval_set.question_file, question_begin, question_end))
        answer_files.extend([eval_set.answer_file for _ in range(len(questions))])

    # random shuffle the questions to balance the loading

    random_order = list(range(len(questions)))
    random.shuffle(random_order)
    questions = [questions[i] for i in random_order]
    answer_files = [answer_files[i] for i in random_order]

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path=model_path,
                model_id=model_id,
                model_id_postfix=model_id_postfix,
                questions=questions[i : i + chunk_size],
                answer_files=answer_files[i : i + chunk_size],
                max_new_token=max_new_token,
                num_choices=num_choices,
                num_gpus_per_model=num_gpus_per_model,
                max_gpu_memory=max_gpu_memory,
                dtype=dtype,
                revision=revision,
            )
        )

    if use_ray:
        ray.get(ans_handles)


def get_stop_token_ids_stopping_criteria(stop_token_ids: List[List[int]]):
    stop_tensors = [torch.LongTensor(stop_token_id).cuda() for stop_token_id in stop_token_ids]

    def stop_token_ids_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        do_stop = False
        for stop_token_ids_tensor in stop_tensors:
            do_stop = torch.equal(input_ids[..., -(len(stop_token_ids_tensor)) :].squeeze(0), stop_token_ids_tensor)
            if do_stop:
                return do_stop
        return do_stop

    return stop_token_ids_stopping_criteria


@torch.inference_mode()
def get_model_answers(
    model_path: str,
    model_id: str,
    questions: List[Dict[str, str]],
    answer_files: List[str],
    max_new_token: int,
    model_id_postfix: str,
    num_choices: int = 1,
    num_gpus_per_model: int = 1,
    max_gpu_memory: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    revision: str = "main",
):
    model, tokenizer = load_model(
        model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    print(f"Loaded model as {type(model)} and tokenizer as {type(tokenizer)}")
    print(f"Use conversation template of model ID {model_id}: {get_conversation_template(model_id).name}")

    for question, answer_file in tqdm(zip(questions, answer_files), total=len(questions)):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        language_code = question["question_id"].split("_")[-1]
        assert language_code in "EN DE FR IT ES", f"Language code {language_code} not supported."
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            stop_token_ids = get_stop_token_ids_and_eos_token_id(conv_template=conv, tokenizer=tokenizer)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                if language_code in "EN DE FR IT ES":
                    prompt = conv.get_prompt_by_language_code(language_code)
                else:
                    warnings.warn(f"Language code {language_code} not supported. Use default language code 'EN'.")
                    prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                try:
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                        stopping_criteria=StoppingCriteriaList([get_stop_token_ids_stopping_criteria(stop_token_ids=stop_token_ids)]),
                    )
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]
                    # tqdm.write(
                    #     f"Generated {output_ids.shape} tokens from {torch.as_tensor(input_ids).shape} tokens as prompt"
                    # )

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [i for i, id in enumerate(output_ids) if id in conv.stop_token_ids]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted([output.find(stop_str) for stop_str in conv.stop_str if output.find(stop_str) > 0])
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.replace(conv.sep, "").replace(conv.sep2, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    print(e)
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)
                conv.messages[-1][-1] = output
                # tqdm.write(f"Conversation - Turn #{j}:")
                # tqdm.write(conv.system_template.format(system_message=conv.system_messages[language_code]))
                # for msg in conv.messages:
                #     role, text = msg
                #     tqdm.write(f"{role}: {text[:print_num_chars]} ... {text[-print_num_chars:]}")

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id + "-" + model_id_postfix,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


def get_stop_token_ids_and_eos_token_id(conv_template, tokenizer) -> Tuple[List[List[int]], int]:
    turn_sep_token_ids = tokenize_text(tokenizer, conv_template.sep2)
    stop_token_ids = [turn_sep_token_ids]

    if conv_template.stop_str is not None:
        stop_strings = [conv_template.stop_str] if isinstance(conv_template.stop_str, str) else conv_template.stop_str
        for stop in stop_strings:
            tokens_ids = tokenize_text(tokenizer, stop)
            stop_token_ids.append(tokens_ids)
    unique_stop_token_ids = list(map(list, set(map(tuple, stop_token_ids))))
    return unique_stop_token_ids


def tokenize_text(tokenizer, text: str):
    """Workaround SPTokenizer inconsistencies:
    The SPTokenizer detokenizes e.g. "User:" either as [5468, 539] (["User", ":"])
    or as [1465, 520, 543, 603, 539] (["U", "s", "e", "r", ":"])

    "°°°" gets tokenized by [tokenizer._convert_token_to_id(char) for char in text] to  [1459, 1459, 1459]
    but by tokenizer(text)["input_ids"] to [14341, 1459, 1459]
    """
    try:
        ids = [tokenizer._convert_token_to_id(text)]
    except Exception:
        try:
            ids = [tokenizer._convert_token_to_id(tok) for tok in text]
        except Exception:
            ids = []
    alternative_ids = tokenizer(text)["input_ids"]

    ids = alternative_ids if len(alternative_ids) < len(ids) else ids
    return ids
