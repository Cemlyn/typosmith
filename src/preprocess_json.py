import os
import re
import difflib
import json
from typing import Iterator, List, Dict, Any
from collections import defaultdict, Counter

import tqdm
import pandas as pd
from nltk.metrics.distance import edit_distance

FILE_PATH = os.path.join(
    "data", "github-typo-corpus.v1.0.0.jsonl", "github-typo-corpus.v1.0.0.jsonl"
)
OUTPUT_PATH = os.path.join("data", "github-typo-corpus.v1.0.0.csv")

PROBABILITY_TYPO_THRESHOLD = 0.5
TYPO_LANGUAGES = [
    "eng",
    "fra",
    "spa",
    "ita",
    "deu",
    "por",
    "nld",
    "swe",
    "dan",
    "cym",
    "oci",
    "bre",
]


def count_lines(file_path: str) -> int:
    """Count the number of lines in a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def jsonl_chunker(
    file_path: str, chunk_size: int = 1, max_chunks: int = None
) -> Iterator[List[Dict[Any, Any]]]:

    current_chunk = []
    chunk_count = 0

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            json_obj = json.loads(line.strip())
            current_chunk.append(json_obj)

            # When we reach the chunk size, yield the chunk
            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []
                chunk_count += 1

                if max_chunks and chunk_count >= max_chunks:
                    break

        # Yield the last chunk if it's not empty
        if current_chunk:
            yield current_chunk
            chunk_count += 1


def process_repo_typos(repo_typos):
    """
    Process typos from a repository chunk.

    Args:
        repo_typos (List[Dict]): List of repository typo records

    Yields:
        Tuple[str, str]: Pairs of (original_string, corrected_string)
    """
    # Each chunk contains a list of records
    for record in repo_typos:
        # Each record has an 'edits' field containing the typos
        for typo in record["edits"]:
            original_string = typo["src"]["text"].strip()
            corrected_string = typo["tgt"]["text"].strip()
            if (
                "prob_typo" in typo
                and typo["src"]["lang"] in TYPO_LANGUAGES
                and typo["tgt"]["lang"] in TYPO_LANGUAGES
                and typo["prob_typo"] > PROBABILITY_TYPO_THRESHOLD
            ):
                yield remove_diacritics(original_string), remove_diacritics(
                    corrected_string
                )
            else:
                continue


def preprocess_function(examples, tokenizer):
    """Preprocess the examples for training."""
    # Add prefix for T5 - now asking to generate a typo
    inputs = ["make_typo: " + text for text in examples["corrected"]]

    # Tokenize inputs and targets - now correct text is input, typo is target
    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["text"], max_length=128, truncation=True, padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def clean_string(s):
    s = remove_leading_trailing_nonalpha(s)
    s = s.lower()
    return s


def remove_leading_trailing_nonalpha(s):
    return re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", s)


def is_diff_in_word(original: str, corrected: str) -> bool:
    """
    Determines if a change between the original and corrected string
    occurs inside any alphabetic word (e.g., 'cat' -> 'cut' is True).
    """
    matcher = difflib.SequenceMatcher(None, original, corrected)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        # Check the context of the original string
        before = original[max(i1 - 1, 0) : i1]
        after = original[i2 : i2 + 1]

        # If the change is surrounded by alphabetic characters, it's in a word
        in_word = (before.isalpha() if before else False) and (
            after.isalpha() if after else False
        )

        # Or if the changed characters themselves are alphabetic
        changed_text = original[i1:i2] + corrected[j1:j2]
        if any(c.isalpha() for c in changed_text) and in_word:
            return True

    return False


def find_word_pairs(sentence1, sentence2):

    words1 = sentence1.split()
    words2 = sentence2.split()

    matcher = difflib.SequenceMatcher(None, words1, words2)
    pairs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend(zip(words1[i1:i2], words2[j1:j2]))
        elif tag == "replace":
            pairs.extend(zip(words1[i1:i2], words2[j1:j2]))
        elif tag == "insert":
            pairs.extend([(None, word) for word in words2[j1:j2]])
        elif tag == "delete":
            pairs.extend([(word, None) for word in words1[i1:i2]])

    return pairs


def is_only_non_alpha(s):
    """Returns True if the string contains only non-alphabetic characters."""
    return bool(s) and not re.search(r"[a-zA-Z]", s)


import unicodedata


def remove_diacritics(text):
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def extract_typo_info(correct: str, typo: str):
    for i, (c1, c2) in enumerate(zip(correct, typo)):
        if c1 != c2:
            return {"position": i, "correct_char": c1, "typo_char": c2}
    # Handle case where lengths differ (insertion/deletion)
    if len(correct) != len(typo):
        return {
            "position": min(len(correct), len(typo)),
            "correct_char": correct[len(typo) :] if len(correct) > len(typo) else "",
            "typo_char": typo[len(correct) :] if len(typo) > len(correct) else "",
        }
    return None


def has_consecutive_diffs(a: str, b: str) -> bool:
    counter = 0
    for diff in difflib.ndiff(a, b):
        if diff.startswith("+") or diff.startswith("-"):
            counter += 1
        elif counter > 0:
            counter = 0
        if counter > 2:
            return True
    return False


def diff_count(a: str, b: str) -> int:
    diff = list(difflib.ndiff(a, b))
    return sum(1 for i in diff if i.startswith("+") or i.startswith("-"))


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # First, create the CSV file if it doesn't exist
    if not os.path.exists(OUTPUT_PATH):
        # Collect all pairs first
        all_pairs = []
        chunker = jsonl_chunker(FILE_PATH, 1)
        for chunk in tqdm.tqdm(chunker):
            for original_string, corrected_string in process_repo_typos(chunk):
                all_pairs.append(
                    {"text": original_string, "corrected": corrected_string}
                )

        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_pairs)
        df.to_csv(
            OUTPUT_PATH, index=False, quoting=1
        )  # quoting=1 means quote all fields

    # extracing word pairs from the file
    df = pd.read_csv(OUTPUT_PATH)
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].apply(remove_diacritics)

    df["corrected"] = df["corrected"].str.lower()
    df["corrected"] = df["corrected"].apply(remove_diacritics)
    df["diff_in_word"] = df.apply(
        lambda row: is_diff_in_word(row["text"], row["corrected"]), axis=1
    )

    print(df.shape)
    # Constraining typos to those that occur in a word for now
    df = df[df["diff_in_word"]]
    print(df.shape)

    # Extract word pairs
    all_word_pairs = []
    for _, row in tqdm.tqdm(df.iterrows()):
        word_pairs = find_word_pairs(row["text"], row["corrected"])
        # Add original and corrected sentences to each word pair

        for pair in word_pairs:
            if pair[0] is not None and pair[1] is not None and pair[0] != pair[1]:
                # Clean both words
                clean_typo = clean_string(pair[0])
                clean_correct = clean_string(pair[1])

                # Running checks
                both_not_empty = clean_typo and clean_correct
                only_non_alpha = is_only_non_alpha(clean_typo) or is_only_non_alpha(clean_correct)
                

                if both_not_empty and (not only_non_alpha):
                    all_word_pairs.append(
                        {
                            "correct": clean_correct,
                            "typo": clean_typo,
                            #"original_sentence": row["text"],
                            #"corrected_sentence": row["corrected"],
                        }
                    )

    # Convert to DataFrame for easier viewing and saving
    word_pairs_df = pd.DataFrame(all_word_pairs)

    # Remove pairs where correct and typo are the same due to the string cleaning performed earlier
    word_pairs_df = word_pairs_df[word_pairs_df["correct"] != word_pairs_df["typo"]]

    # Remove rows where most chars are non alpha, i.e. greater than 20%
    word_pairs_df["non_alpha_percentage"] = word_pairs_df.apply(
        lambda row: sum(1 for c in row["correct"] if not c.isalpha())
        / len(row["correct"]),
        axis=1,
    )
    word_pairs_df = word_pairs_df[word_pairs_df["non_alpha_percentage"] < 0.2]

    word_pairs_df["edit_distance"] = word_pairs_df.apply(
        lambda row: edit_distance(row["correct"], row["typo"]), axis=1
    )

    # Remove rows where every letter is different, i.e. word substitution
    word_pairs_df["len_typo"] = word_pairs_df["typo"].apply(len)
    word_pairs_df = word_pairs_df[
        word_pairs_df["edit_distance"] <= word_pairs_df["len_typo"]
    ]

    word_pairs_df.to_csv(os.path.join("data", "word_pairs.csv"), index=False)
