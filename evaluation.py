import json
import os
import time
import tiktoken
import argparse

from os.path import join,exists
from openai import OpenAI
from tqdm import tqdm

encoder = tiktoken.get_encoding("cl100k_base")

SYS_MSG ="Based on the provided Input (if any) and Generated Text, answer the ensuing Ouestions with either a YES or NOchoice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. Howevernote that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider aquestion that asks. \"Does each sentence in the generated text use a second person?” If even one sentence doesnot use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated textmust be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no informationthat could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence irthe generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevantinformation to answer the question. Consequently, the answer should be 'NO'.'''"

def load_jsonl(file_path):
    "General function to load jsonl file"
    _data = []
    with open(file_path, 'r') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data

def bool_ratio(fpath):
    "Calculate true false ratio for eval results"
    _data = load_jsonl(fpath)
    count = {"true":0, "false":0}
    for entry in _data:
        if entry.get("eval", None) is None:
            print("Wrong output")
            print(entry['id'])
        if len(entry['decomposed_questions']) != len(entry['eval']):
            print("Wrong length")
            print(entry['id'])
        if None in entry['eval']:
            print("None in eval")
            print(entry['id'])
        
        for eva_value in entry['eval']:
            if eva_value:
                count["true"] += 1
            else:
                count["false"] += 1
    
    print("-------- True False Table --------")
    print(count)
    print(f"Percentage of True: {count['true']/sum(count.values())}")
    return

def run_evaluation(client, in_path, o_dir, eval_model="gpt-4-0314", temperature=0):
    """
    Main function to run decomposed questisons evaluation on models' outputs
        in_path: str, path to the model output file
        o_dir: str, path to the output folder
        eval_model: str, default "gpt-4-0314", model name to be used for evaluation
        temperature: float, default 0, temperature to be used for evaluation
    """
    _data = load_jsonl(in_path)
    _model_name = in_path.split('/')[1].split('_')[0]
    
    # ceate output folder if not exists
    _o_dir = join(o_dir, eval_model)
    if not exists(_o_dir):
        os.mkdir(_o_dir)
                
    _opath = join(_o_dir, f"{_model_name}_DecomposeEval.json")
    
    # load_results if exists
    if os.path.exists(_opath):
        _exist = load_jsonl(_opath)
        _exist_ids = [i['id'] for i in _exist]
        for pos, instance in enumerate(_data):
            if instance['id'] in _exist_ids:
                _data[pos] = _exist[_exist_ids.index(instance['id'])]
    
    result_writer = open(_opath, 'w')
    
    print(f"--------Evaluating output from {in_path}--------")
    print(f"--------Evaluation Using {eval_model}--------")
    for entry in tqdm(_data):
        # ski if eval exists
        if entry.get('eval', None) is not None:
            result_writer.write(json.dumps(entry) + '\n')
            result_writer.flush()
            continue
        
        input_task = entry['input']
        output = entry['output']
        if output is None: # skip if result hasn't been generated
            continue
        
        message = []
        answer = ""
        # print(f"--------Instance {entry['id']}--------")
        for question in entry['decomposed_questions']:
            if len(message) == 0:
                if input_task:
                    content =  f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
                else:
                    content =  f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
            else:
                content = f"{question}\n"
            message.append({"role": "user", "content": content})
            # create a chat completion
            success = False
            early_stop = False
            while not success:
                try:
                    completion = client.chat.completions.create(
                        model=eval_model,
                        messages=message,
                        temperature=temperature,
                    )
                    generation = completion.choices[0].message.content
                    message.append(
                        {"role": "assistant", "content": generation})
                    # check if generation is yes or no
                    if generation.lower().startswith("yes") or generation.lower().startswith("no"):
                        if generation.lower().startswith("yes"):
                            answer += "Yes\n"
                        else:
                            answer += "No\n"
                    else:
                        if "YES" in generation and "NO" not in generation:
                            answer += "Yes\n"
                        elif "YES" not in generation and "NO" in generation:
                            answer += "No\n"
                        else:
                            for msg in message:
                                print(msg['content'])
                            print("NO YES or NO answer!" + generation)
                            answer += "None\n"
                            early_stop = True
                            break
                    success = True
                except Exception as e:
                    print("ERROR!")
                    print(e)
                    print("Retry!")
                    time.sleep(20)

            # when no answer occurs, break the loop and continue to next instance
            if early_stop:
                break

        answer = answer[:-1]
        # save eval results as List[bool]
        bool_results = []
        for i in answer.split('\n'):
            if i == "Yes":
                bool_results.append(True)
            elif i == "No":
                bool_results.append(False)
            else:
                bool_results.append(None)
    
        entry['eval'] = bool_results
        result_writer.write(json.dumps(entry) + '\n')
        result_writer.flush()
        
    result_writer.close()
    
    # run true false ratio calculation
    bool_ratio(_opath)
    
    return _opath

def main_run(args):
    client = OpenAI(api_key=args.api_key)
    results_file = args.input
    output_dir = args.output_dir
    eval_model = args.model
    temperature = args.temperature
    
    if not exists(results_file):
        print(f"results_dir {results_file} not exists")
        return
    
    # run evaluation for each model
    run_evaluation(client, results_file, output_dir, eval_model, temperature) 
    return    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4-0314", help="model name to be used for evaluation")
    
    parser.add_argument("--input", type=str, required=True, help="path to the results file")
    parser.add_argument("--output_dir", type=str, required=True, help="path to the output folder")
    
    parser.add_argument("--temperature", type=float, default=0, help="temperature to be used for evaluation")
    args = parser.parse_args()
    main_run(args)
