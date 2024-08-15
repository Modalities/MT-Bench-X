# MT-Bench-X

MT-Bench-X is a framework to evaluate the multilingual instruction following capabilities of large language models.
Adapting multilingual pre-trained large language models (LLMs) into articulate and effective assistants is crucial for their application across diverse linguistic regions. 
In line with this objective, we release our multilingual evaluation benchmark MT-Bench-X to evaluate multilingual models that have been instruction-tuned on different language compositions. 
We focus on a selection of the most spoken Indo-European languages: English, German, French, Italian, and Spanish.

For more details, see our [Paper](https://arxiv.org/abs/2402.13703).

This evaluation framework allows to 
1. Generate the answers to the MT-Bench-X benchmark across the selected languages
2. Let GPT-4-as-a-judge assess the answers
3. Summarize the results in a single file

## Installation

```bash
cd MT-Bench-X
mkdir venvs
python3 -m virtualenv --prompt mtbenchx --system-site-packages "venvs/mtbenchx"
. venvs/mtbenchx/bin/activate
pip install --upgrade pip
pip install -e . 
```

## Usage


> An OpenAI key must be loaded within your environment to judge the generated model answers by GPT-4!

Example execution:
```bash
OPENAI_API_KEY=xyz mtbenchx \
    # load model from a local checkpoint or by a hugging face hub id
    --model-path "your_model_path" \
    # the model-id is used to get the correct ModelAdapter and Conversation(-Template)
    --model-id "llama-2" \
    # in case there are several checkpoints you want to compare, change --model-id-postfix for their identification
    --model-id-postfix "my-local-model-variation" \
    --question-begin 6 --question-end 10 \
    --eval-languages DE EN --max-new-token 1024 
    # how many OpenAI requests to execute in parallel
    --parallel 6 \
    # allows for data-parallel answer generation
    --num-gpus-per-model 1 --num-gpus-total 8 
```

Type `mtbenchx --help` for more information about the input arguments.



## Citation

```
@misc{
 weber2024investigatingmultilingualinstructiontuningpolyglot,
 title={Investigating Multilingual Instruction-Tuning: Do Polyglot Models Demand for Multilingual Instructions?}, 
 author={Alexander Arno Weber and Klaudia Thellmann and Jan Ebert and Nicolas Flores-Herr and Jens Lehmann and Michael Fromm and Mehdi Ali},
 year={2024},
 eprint={2402.13703},
 archivePrefix={arXiv},
 primaryClass={cs.CL},
 url={https://arxiv.org/abs/2402.13703}, 
}
```
