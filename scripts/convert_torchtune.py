import argparse
import ast
import datasets
import json


# These constants should match the constants at the top of main.py
# TODO: Move these constants to a shared file
INPUT_FIELD = "input"
OUTPUT_FIELD = "output"
ANSWER_FIELD = "answer"

# These constants should match with the system prompt in the config file and with the GRPO constants in Unsloth
COT_OPENING = "\n<reasoning>"
COT_CLOSING = "\n</reasoning>"
LABEL_OPENING = "\n<answer>"
LABEL_CLOSING = "\n</answer>"

class ComplianceProjectError(ValueError):
    pass

def clean_rule(rule):
    # Looking for 1. or 2. etc.
    splits = rule.split(". ", 1)
    if len(splits) > 1:
        rule = splits[1].strip()
    else:
        rule = rule.strip()
    return rule

def clean_explanation(explanation):
    # Looking for "Turn x: "
    explanation = explanation.split(": ", 1)[1].strip()
    return explanation

def parse_string_list(string_list):
    # Format: "1. ['PASS', 'PASS', 'PASS']\n"
    string_list = string_list.split(". ", 1)[1].strip()
    native_list = ast.literal_eval(string_list)
    return native_list

def get_dialogue_turns(dialogue, expected_turns, example_index=-1):
    delimiters = ["'User':", """"User":"""]
    dialogue_turns = []
    for delimiter in delimiters:
        if delimiter in dialogue:
            dialogue_preamble = dialogue.split(delimiter, 1)[0]
            main_dialogue = dialogue.split(delimiter, 1)[1]
            dialogue_turns = [f"{delimiter}{item}" for item in main_dialogue.split(delimiter) if item]
            dialogue_turns[0] = dialogue_preamble + dialogue_turns[0]
            break
    if len(dialogue_turns) != expected_turns:
        raise ComplianceProjectError(f"""
            Example {example_index}: Number of dialogue turns ({len(dialogue_turns)}) does not match number of turns in labels ({expected_turns}).
            Delimiters: {delimiters}
            Dialogue: {json.dumps(dialogue_turns, indent=4)}
            """)
    return dialogue_turns
    
def preprocess_dataset(dataset_path, subset=None, split=None, size=None, local=False, data_dir="data"):
    if local:
        dataset = datasets.load_dataset('json', data_files=dataset_path)['train']
    else:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)
    print(f"Examples in {subset} {split}: {len(dataset)}")

    examples = []
    for row_index, row in enumerate(dataset):
        ##################
        # Setup
        ##################
        rules = row["rules"]
        dialogue = row["dialogue"]
        labels = row["labels"]
        explanations = row["explanations"]
        discussions = row["discussions"]

        cleaned_rules = []
        cleaned_labels = []
        cleaned_explanations = []
        cleaned_discussions = []
        for i in range(len(rules)):
            cleaned_rules.append(clean_rule(rules[i]))
            cleaned_labels.append(parse_string_list(labels[i]))
            cleaned_explanations.append([clean_explanation(explanation) for explanation in parse_string_list(explanations[i])])
            cleaned_discussions.append([clean_explanation(discussion) for discussion in parse_string_list(discussions[i])])

        num_rules = len(cleaned_rules)
        num_turns = len(cleaned_labels[0])
        dialogue_turns = get_dialogue_turns(dialogue, expected_turns=num_turns, example_index=row_index)

        for i in range(num_rules):
            for j in range(num_turns):
                example = {}

                ##################
                # Input
                ##################
                rule = cleaned_rules[i]
                dialogue_subset = "".join(dialogue_turns[:j+1])
                example[INPUT_FIELD] = f'''
Rule Agent must follow:
{rule}

Conversation:
{dialogue_subset}
'''
                ##################
                # Output        
                ##################
                discussion = cleaned_discussions[i][j]
                explanation = cleaned_explanations[i][j]
                label = cleaned_labels[i][j]
                example[OUTPUT_FIELD] = f"{COT_OPENING} {discussion} {explanation} {COT_CLOSING} {LABEL_OPENING} {label} {LABEL_CLOSING}"
                example[ANSWER_FIELD] = label
                examples.append(example)
                

    processed_dataset = datasets.Dataset.from_list(examples)

    if size is not None and len(processed_dataset) > size:
        processed_dataset = processed_dataset.shuffle(seed=42)
        processed_dataset = processed_dataset.select(range(size))

    file_path = f"{data_dir}/{subset}_{split}_{len(processed_dataset)}.jsonl"

    processed_dataset.to_json(file_path, orient='records', lines=True, indent=None)
    print(f"Examples in dataset preprocessed for TorchTune: {size}")
    print(f"Saved to {file_path}")
    return file_path


def main(args):
    # Local:
    # preprocess_dataset("test.jsonl", local=True)

    huggingface_dataset = "tomg-group-umd/compliance"
    # Subset choices are "easy" or "hard"
    # Easy: 9007 train, 1793 val, 67 test
    # Hard: 1670 train, 313 val, 44 test
    subsets = ["easy", "hard"]
    splits = ["train", "validation", "test"]
    file_paths = {}
    for subset in subsets:
        for split in splits:
            file_path = preprocess_dataset(huggingface_dataset, subset, split, size=args.train_size, data_dir=args.data_dir)
            file_paths[f"{subset}_{split}"] = file_path

    if args.extra_examples:
        train_file_path = file_paths["easy_train"]
        val_file_path = file_paths["easy_validation"]
        
        train_dataset = datasets.load_dataset("json", data_files={"placeholder": train_file_path})["placeholder"]
        val_dataset = datasets.load_dataset("json", data_files={"placeholder": val_file_path})["placeholder"]
        
        combined_dataset = datasets.concatenate_datasets([train_dataset, val_dataset])
        combined_dataset = combined_dataset.shuffle(seed=42)

        size = args.train_size
        if size is not None and len(combined_dataset) > size:
            combined_dataset = combined_dataset.select(range(size))

        file_path = f"{args.data_dir}/easy_train_{len(combined_dataset)}.jsonl"
        combined_dataset.to_json(file_path)
        print(f"Combined easy train and validation datasets saved to {file_path}")
        


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--train_size", default=7500, type=int)
    parser.add_argument("--extra_examples", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
