# MT-Bench-X

MT-Bench-X is a framework to evaluate the multilingual instruction following capabilities of large language models.
The adaptation of multilingual pre-trained large language models (LLMs) into articulate and effective assistants is crucial for their application across diverse linguistic regions. In line with this objective, our research pioneers an extensive examination of the performance of multilingual models that have been instruction-tuned on different language compositions. Utilizing parallel instruction-tuning benchmarks, we focus on a selection of the most spoken Indo-European languages.

For more details see our [Paper](https://arxiv.org/abs/2402.13703).

## Installation

```bash
cd mtbenchx
mkdir venvs
python3 -m virtualenv --prompt mtbenchx --system-site-packages "venvs/mtbenchx"
. venvs/mtbenchx/bin/activate
pip install --upgrade pip
pip install -e . 
```

## Usage
> An OpenAI key needs to be loaded within your environment for judging the generated model answers by GPT-4!


Simply use the `mtbenchx` command installed above.
Type `mtbenchx --help` for more information about input arguments.

Example execution:
```bash
OPENAI_API_KEY=xyz mtbenchx --model-path "your_model_path" --model-id "llama-2" --question-begin 6 --question-end 10 --max-new-token 1024 --model-id-postfix my-local-model-variation --eval-languages DE EN --parallel 6
```

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

