import json

input_file = '../../data/build/mnli/mm2_split_synchronous.jsonl'
output_file = '../../data/build/mnli/mm2_split_synchronous_orig.jsonl'

with open(output_file, 'w', encoding='utf-8') as outfile:
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            json_obj = json.loads(line)
            if 'modified_conn' not in json_obj:
                json_str = json.dumps(json_obj, ensure_ascii=False)
                outfile.write(json_str + '\n')

print(f"Processing complete. Output saved to {output_file}")