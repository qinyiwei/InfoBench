# InfoBench
## Dataset
[Instruction Dataset](https://docs.google.com/spreadsheets/d/1aNgUuyLJfCXlzmqbPewETsuY3VbKbEVGYoDPlZmwxQ0/edit#gid=0)\
[Question Lable](https://docs.google.com/spreadsheets/d/1eBqviT84YT7y8pqyrlKLW7_gzgkWg5EUob2WT7R_myI/edit#gid=998404177)\
[LLM Generations](https://docs.google.com/spreadsheets/d/1yw5W6jERRNUbdcYJieOHAos1hCyW7h7daf5x-ya5h0o/edit#gid=0)\
[GPT-4 Evaluation Result](https://docs.google.com/spreadsheets/d/1rRSuIMsglhWKWcqKTkEWWu_XarsnSV6w8aUA6jbmCRQ/edit#gid=0)

## Evaluation
Evaluate LLM's outputs on decomposed questions. Using GPT-4-0314 by default in this research.
```bash
python evaluation.py \
  --input model/output.json
  --output_dir evaluation/
  --eval_model gpt-4-0314
  --temperature 0
```

Each data entry will includes an "eval" key in format of ```List[bool]``` which represent "Yes" or "No" answers to each decomposed questions. The final output evaluation file will be saved in JSON format at location ```<output_dir>/<eval_model>/```.
