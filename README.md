# InfoBench

- **Paper:** [InFoBench: Evaluating Instruction Following Ability in Large Language Models](https://arxiv.org/pdf/2401.03601.pdf)
- **Dataset:** [InFoBench Dataset](https://huggingface.co/datasets/kqsong/InFoBench)
  
## Citation
```
@article{qin2024infobench,
      title={InFoBench: Evaluating Instruction Following Ability in Large Language Models}, 
      author={Yiwei Qin and Kaiqiang Song and Yebowen Hu and Wenlin Yao and Sangwoo Cho and Xiaoyang Wang and Xuansheng Wu and Fei Liu and Pengfei Liu and Dong Yu},
      year={2024},
      eprint={2401.03601},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Evaluation with InFoBench
### Step1: Dataset Usage
You can directly download it with huggingface datasets.
``` python
from datasets import load_dataset

dataset = load_dataset("kqsong/InFoBench")
```

### Step2: Generating the response
Provide an output file in `model/output.json`.
Each data entry should be a json object with a newline, containing all the fields in the input format.
The generated response should be included in the json object with the new field named `output`.

We suggest using greedy decoding to avoid the randomness of decoding.


### Step3: Evaluation

Evaluate LLM's outputs on decomposed questions. Using GPT-4-0314 by default in this research.
```bash
python evaluation.py \
  --api_key <OPENAI KEY> \
  --eval_model gpt-4-0314 \
  --input model/output.json \
  --output_dir evaluation/ \
  --temperature 0
```

Each data entry will include an "eval" key in the format of ```List[bool]``` which represents "Yes" or "No" answers to each decomposed question.
The final output evaluation file will be saved in JSON format at location ```<output_dir>/<eval_model>/```.
