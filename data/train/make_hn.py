import os
import random
import string
import argparse
import json

def neg_remove(text):
    if not text:
        return text
    valid_indices = [i for i, char in enumerate(text) if char.isalnum()]
    if not valid_indices:
        return text
    idx = random.choice(valid_indices)
    return text[:idx] + text[idx + 1:]

def neg_duplicate(text):
    if not text:
        return text
    valid_indices = [i for i, char in enumerate(text) if char.isalnum()]
    if not valid_indices:
        return text
    idx = random.choice(valid_indices)
    return text[:idx] + text[idx] + text[idx:]

def neg_replace(text):
    if not text:
        return text
    valid_indices = [i for i, char in enumerate(text) if char.isalnum()]
    if not valid_indices:
        return text
    idx = random.choice(valid_indices)
    original_char = text[idx]
    if original_char.isalpha():
        random_char = random.choice([c for c in string.ascii_letters if c != original_char])
    elif original_char.isdigit():
        random_char = random.choice([c for c in string.digits if c != original_char])
    else:
        return text
    return text[:idx] + random_char + text[idx + 1:]

def neg_swap(text):
    if not text or len(text) < 2:
        return text
    valid_indices = [i for i, char in enumerate(text) if char.isalnum()]
    if len(valid_indices) < 2:
        return text
    idx1, idx2 = random.sample(valid_indices, 2)
    text_list = list(text)
    text_list[idx1], text_list[idx2] = text_list[idx2], text_list[idx1]
    return ''.join(text_list)

def neg_add(text):
    if not text:
        return text
    random_char = random.choice(string.ascii_letters + string.digits)
    idx = random.randint(0, len(text))
    return text[:idx] + random_char + text[idx:]

def ocr_neg(gt_text, method_num=0):
    if not gt_text:
        return gt_text
    if type(gt_text) == float:
        gt_text = str(gt_text)
    aug_methods = ['remove', 'duplicate', 'replace', 'swap']
    if len(gt_text) <= 1:
        aug_methods = ['duplicate', 'replace', 'duplicate', 'replace']
    aug_method = aug_methods[method_num]
    if aug_method == 'remove':
        return neg_remove(gt_text)
    elif aug_method == 'duplicate':
        return neg_duplicate(gt_text)
    elif aug_method == 'replace':
        return neg_replace(gt_text)
    elif aug_method == 'swap':
        return neg_swap(gt_text)
    elif aug_method == 'add':
        return neg_add(gt_text)
    return gt_text

def generate_hard_negatives(gt_text, max_retries=10):
    hard_negatives = []
    for num in range(4):
        retries = 0
        modified_text = ocr_neg(gt_text, num)
        while modified_text == gt_text and retries < max_retries:
            modified_text = ocr_neg(gt_text, num)
            retries += 1
        hard_negatives.append(modified_text)
    return "<sep>".join(hard_negatives)

def main(input_json, output_json):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["hard_negatives"] = generate_hard_negatives(item["gt_text"])

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"새로운 JSON 파일이 저장되었습니다: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hard negatives from gt.json")
    parser.add_argument('--input_json', type=str, required=True, help="Input path to gt.json")
    parser.add_argument('--output_json', type=str, required=True, help="Output path for augmented json")
    args = parser.parse_args()

    main(args.input_json, args.output_json)