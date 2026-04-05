import json

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

fp16 = load_jsonl("baseline_results/safetybench_fp16_details.jsonl")
int8 = load_jsonl("baseline_results/safetybench_int8_details.jsonl")
int4 = load_jsonl("baseline_results/safetybench_int4_details.jsonl")

# SafetyBench Raw Questions to get the text
with open("SafetyBench/opensource_data/test_en.json", "r") as f:
    questions = json.load(f)

fp16_correct = [i for i, r in enumerate(fp16) if r["is_correct"]]
int8_correct = [i for i, r in enumerate(int8) if r["is_correct"]]
int4_wrong   = [i for i, r in enumerate(int4) if not r["is_correct"]]

# Find intersection
interesting = set(fp16_correct) & set(int8_correct) & set(int4_wrong)
print(f"Found {len(interesting)} cases where FP16 & Int8 are correct but Int4 is wrong.")

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]

def print_case(idx):
    q = questions[idx]
    print("="*80)
    print(f"Category: {fp16[idx]['category']}")
    print(f"Question: {q['question']}")
    for j, opt in enumerate(q['options']):
        print(f"  {CHOICE_LABELS[j]}. {opt}")
    
    print("\nCorrect Answer:", fp16[idx]["correct"])
    print(f"FP16 Prediction : {fp16[idx]['predicted']} (Correct)")
    print(f"Int8 Prediction : {int8[idx]['predicted']} (Correct)")
    print(f"Int4 Prediction : {int4[idx]['predicted']} (WRONG)")

# Print examples specifically from the Offensiveness category
count = 0
for idx in list(interesting):
    cat = fp16[idx]['category']
    if cat == "Offensiveness":
        print_case(idx)
        count += 1
        if count >= 20:
            break
