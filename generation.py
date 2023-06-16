import openai
import tqdm
import csv
import os
import argparse
import anthropic
import time

def main(args):
  if "claude" in args.model:
    client = anthropic.Client(args.api_key)
  else:
    openai.api_key = args.api_key
    if args.api_base:
      openai.api_base = args.api_base


  start_id = 0
  if os.path.exists(os.path.join(args.generation_file_path, args.generation_file_name)):
    f_read = open(os.path.join(args.generation_file_path, args.generation_file_name),'r')
    csv_reader = csv.reader(f_read, delimiter=',')
    for row in csv_reader:
      start_id += 1
    f_read.close()
  print("start_id:{},end_id:{}".format(start_id,args.end_id))


  f_read = open(os.path.join(args.instruction_file_path, args.instruction_file_name),'r')
  csv_reader = csv.reader(f_read, delimiter=',')
  header = next(csv_reader)

  f_write = open(os.path.join(args.generation_file_path, args.generation_file_name),'a')
  csv_writer = csv.writer(f_write, delimiter=',')

  line_count = 0
  
  instruction_id = header.index("instruction")
  input_id = header.index("input")
  for row in csv_reader:
    if line_count < start_id:
      line_count += 1
      continue
    if args.end_id and line_count >= args.end_id:
      break
    
    success=False
    while (success==False):
      try:
        if "claude" in args.model:
          # model prompt
          content = f"{anthropic.HUMAN_PROMPT}{row[instruction_id]} \
            \n\n{row[input_id]}{anthropic.AI_PROMPT}"
          # create a chat completion
          resp = client.completion(
              prompt=content,
              stop_sequences=[anthropic.HUMAN_PROMPT],
              model=args.model,
              max_tokens_to_sample=args.max_tokens,
              temperature=args.temperature,
              top_p=args.top_p,
          )
          generation = resp['completion']
        else:
          # model prompt
          content = "{}\n\n{}\n".format(row[instruction_id],row[input_id])
          # create a chat completion
          completion = openai.ChatCompletion.create(
            model=args.model,
            messages=[{"role": "user", "content": content}],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
          )
          generation = completion.choices[0].message.content
        # print the completion
        csv_writer.writerow([row[header.index("id")], generation])
        print(line_count)
        print(content)
        print("generation:"+generation)
        print("-------------------------------------------------")
        line_count += 1
        success=True
      except Exception as e: 
        print("ERROR!")
        print(e)
        print("Retry!")
        time.sleep(20)

  f_read.close()
  f_write.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--model", type=str, required=True, default=None)

    parser.add_argument("--instruction_file_path", type=str, required=True, default=None)
    parser.add_argument("--instruction_file_name", type=str, required=True, default=None)
    parser.add_argument("--generation_file_path", type=str, required=True, default=None)
    parser.add_argument("--generation_file_name", type=str, required=True, default=None)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1024)

    parser.add_argument("--end_id", type=int, default=None)

    args = parser.parse_args()

    main(args)



