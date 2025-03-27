import argparse
import ast
import json
import random
import datasets
from tqdm import tqdm
import re
import spacy

from constants import (
    INPUT_FIELD,
    OUTPUT_FIELD,
    NUM_RULES_METADATA,
    COT_OPENING,
    COT_CLOSING,
    MULTIRULE_LABEL_OPENING,
    MULTIRULE_LABEL_CLOSING,
    RULES_OPENING,
    RULES_CLOSING,
    RULE_NUMBER_OPENING,
    RULE_NUMBER_CLOSING,
    LINE_OPENING,
    LINE_CLOSING,
    EXPLANATION_OPENING,
    EXPLANATION_CLOSING,
)

"""
Convert a Compliance dataset in the format:
{
    rules: List[str],        # numbered 
    dialogue: str, 
    discussions: List[str],  # numbered
    explanations: List[str], # numbered 
    labels: List[str]}       # numbered
}

to an eval-friendly dataset in the format:
{
    question: str,           # total = num_original_examples
    answer: str,             # XML tagged as below
    num_rules: int,
}
"""

class ComplianceProjectError(ValueError):
    pass

def extract_xml_answer(text: str) -> str:
    answer = text.split(MULTIRULE_LABEL_OPENING.strip())[-1]
    answer = answer.split(MULTIRULE_LABEL_CLOSING.strip())[0]
    return answer.strip()

def print_stats(dataset_path, local=True, obj=False):
    if obj:
        dataset = dataset_path
    elif local:
        dataset = datasets.load_dataset("json", data_files={"_": dataset_path}, split="_")
    else:
        dataset = datasets.load_dataset(dataset_path)
    min_rules = float("inf")
    max_rules = 0
    num_pass = 0
    num_fail = 0
    total_rules = 0
    for i, example in enumerate(dataset):
        label = extract_xml_answer(example[OUTPUT_FIELD])
        if label == "PASS":
            num_pass += 1
        elif label == "FAIL":
            num_fail += 1
        else:
            raise ComplianceProjectError(f"Invalid label for example {i}: {example[OUTPUT_FIELD]}")
        if example[NUM_RULES_METADATA] < min_rules:
            min_rules = example[NUM_RULES_METADATA]
        if example[NUM_RULES_METADATA] > max_rules:
            max_rules = example[NUM_RULES_METADATA]
        total_rules += example[NUM_RULES_METADATA]
    mean_rules = total_rules / len(dataset)
    pass_rate = num_pass / len(dataset)
    print(f"""Number of examples: {len(dataset)}
Number of PASS examples: {num_pass}
Number of FAIL examples: {num_fail}
Pass rate: {pass_rate:.1%}
Min rules: {min_rules}
Max rules: {max_rules}
Mean rules: {mean_rules:.1f}
""")

def clean_rule(rule):
    # Use regex to remove any whitespace followed by a number, a period, and a space at the beginning of the string
    rule = re.sub(r"^\s*\d+\.\s", "", rule).strip()
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

def separate_sentences(paragraph):
    doc = nlp_processor(paragraph)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def fill_short_discussions(short_discussions, processed_discussion):
    # Used to fill up the short discussions list and meant to be called inside a loop.
    # Doing this for speed so we can choose the most efficient loop to use at runtime.
    sentences = [sent.text.strip() for sent in processed_discussion.sents]
    first_two_sentences = sentences[:2]
    short_discussion = ' '.join(first_two_sentences)
    short_discussions.append(short_discussion)


def get_cot(discussions, explanations, nlp_processor):
    # There is a discussion for every rule, and within that a discussion for every turn. Get only the discussion from the last turn for the COT.
    last_turn_discussions = [turn_discussions[-1] for turn_discussions in discussions]
    last_turn_explanations = [explanations[-1] for explanations in explanations]

    short_discussions = []
    # This whole thing is slow so we're trying to speed it up with the pipeline version of Spacy's nlp processor
    nlp_pipeline = nlp_processor.pipe(last_turn_discussions, disable=["ner", "tagger"])
    for processed_discussion in nlp_pipeline:
        sentences = [sent.text.strip() for sent in processed_discussion.sents]
        first_two_sentences = sentences[:2]
        short_discussion = ' '.join(first_two_sentences)
        short_discussions.append(short_discussion)

    # Combine the short discussions with the explanations into a single COT for each rule
    cot_by_rule = [f"{short_discussion} {explanation}" for short_discussion, explanation in zip(short_discussions, last_turn_explanations)]
    enumerated_cot = '\n'.join(f"Rule {i+1}. {cot}" for i, cot in enumerate(cot_by_rule))
    return enumerated_cot

def preprocess_dataset(dataset_path, subset=None, split=None, size=None, local=False, data_dir="data", add_cot=False):
    if local:
        dataset = datasets.load_dataset("json", data_files={"placeholder": dataset_path})["placeholder"]
    else:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)
    print(f"Examples in {subset} {split}: {len(dataset)}")

    if add_cot:
        print(f"Doing sentence processing for the COTs which is slow. Should take {len(dataset)//1000+1} minutes to process {len(dataset)} examples...")
        # Setup nlp processor
        spacy_model = "en_core_web_sm"
        if not spacy.util.is_package(spacy_model):
            spacy.cli.download(spacy_model)
        nlp_processor = spacy.load(spacy_model)

    examples = []
    for row_index, row in tqdm(enumerate(dataset)):
        ##################
        # Setup
        ##################
        example ={}
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
        
        ##################
        # Input
        ##################
        # Shuffle all of the lists (rules, labels, explanations) so there is no bias in the rule order
        zipped = list(zip(cleaned_rules, cleaned_labels, cleaned_explanations, cleaned_discussions))
        random.shuffle(zipped)
        cleaned_rules, cleaned_labels, cleaned_explanations, cleaned_discussions = zip(*zipped)

        enumerated_rules = '\n'.join(f"{i+1}. {rule}" for i, rule in enumerate(cleaned_rules))
        example[INPUT_FIELD] = f'''
Rules agent must follow:
{enumerated_rules}

Transcript:
{dialogue}
'''     
        ##################
        # Output        
        ##################
        allpass_label = "PASS"
        violated_rules = []
        violation_lines = []
        violation_explanations = []

        for i in range(num_rules):
            for j in range(num_turns):
                if cleaned_labels[i][j] == "FAIL":
                    allpass_label = "FAIL"
                    violated_rules.append(i+1)
                    violation_lines.append(dialogue_turns[j])
                    violation_explanations.append(cleaned_explanations[i][j])
                    break # We capture the first violation of a given rule and then move to the next rule
        
        if add_cot:
            cot = get_cot(cleaned_discussions, cleaned_explanations, nlp_processor)

        # Format in xml tags
        cot_block = f"{COT_OPENING}\n{cot}\n{COT_CLOSING}\n" if add_cot else ""
        label_block = f"{MULTIRULE_LABEL_OPENING}\n{allpass_label}\n{MULTIRULE_LABEL_CLOSING}\n"
        rules_block = f"{RULES_OPENING}\n{','.join(map(str, violated_rules))}\n{RULES_CLOSING}\n" if violated_rules else ""
        explanation_blocks = ""
        for i in range(len(violated_rules)):
            rule_number = violated_rules[i]
            line_in_transcript = violation_lines[i]
            explanation = violation_explanations[i]
            explanation_blocks += f"""
{RULE_NUMBER_OPENING}
{rule_number}
{RULE_NUMBER_CLOSING}
{LINE_OPENING}
{line_in_transcript}
{LINE_CLOSING}
{EXPLANATION_OPENING}
{explanation}
{EXPLANATION_CLOSING}
"""
        example[OUTPUT_FIELD] = f"{cot_block}{label_block}{rules_block}{explanation_blocks}"
        example[NUM_RULES_METADATA] = num_rules
        examples.append(example)

    processed_dataset = datasets.Dataset.from_list(examples)

    if size is not None and len(processed_dataset) > size:
        processed_dataset = processed_dataset.shuffle(seed=42)
        processed_dataset = processed_dataset.select(range(size))

    file_path = f"{data_dir}/{subset}_{split}_{len(processed_dataset)}.jsonl"
    if add_cot:
        file_path = file_path.replace(".jsonl", "_cot.jsonl")

    processed_dataset.to_json(file_path)
    print_stats(file_path)
    print(f"Saved to {file_path}")
    return file_path

def combine_datasets(non_cot_filepath, cot_filepath):
    non_cot_dataset = datasets.load_dataset("json", data_files={"_": non_cot_filepath}, split="_")
    cot_dataset = datasets.load_dataset("json", data_files={"_": cot_filepath}, split="_")
    combined_dataset = datasets.concatenate_datasets([non_cot_dataset, cot_dataset])
    combined_dataset = combined_dataset.shuffle(seed=42)
    orig_size = len(cot_dataset)
    new_size = len(combined_dataset)
    new_path = cot_filepath.replace(str(orig_size), str(new_size)).replace("_cot", "_combined")
    combined_dataset.to_json(new_path)
    print(f"Saved combined dataset to {new_path}")

def main(args):
    huggingface_dataset = "tomg-group-umd/compliance"
    subsets = args.subsets
    splits = args.splits
    for subset in subsets:
        for split in splits:
            if args.combine:
                non_cot_filepath = preprocess_dataset(huggingface_dataset, subset, split, size=args.train_size, data_dir=args.data_dir, add_cot=False)
                cot_filepath = preprocess_dataset(huggingface_dataset, subset, split, size=args.train_size, data_dir=args.data_dir, add_cot=True)
                combine_datasets(non_cot_filepath, cot_filepath)
            else:  
                file_path = preprocess_dataset(huggingface_dataset, subset, split, size=args.train_size, data_dir=args.data_dir, add_cot=args.cot)
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/multi_rule", type=str)
    parser.add_argument("--train_size", default=10000, type=int)
    # parser.add_argument("--subsets", type=list, default=["easy", "hard"])
    # parser.add_argument("--splits", type=list, default=["train", "validation", "test"])
    parser.add_argument("--subsets", type=list, default=["multi_rule"])
    parser.add_argument("--splits", type=list, default=["train"])
    parser.add_argument("--combine", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cot", default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)