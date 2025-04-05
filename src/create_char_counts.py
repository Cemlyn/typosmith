import re
import os
import pickle
import difflib
from collections import defaultdict, Counter

import tqdm
import pandas as pd
from nltk.metrics.distance import edit_distance


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

    df = pd.read_csv(
        os.path.join("data", "github-typo-corpus.v1.0.0.csv"), encoding="utf-8"
    )
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].apply(remove_diacritics)

    df["corrected"] = df["corrected"].str.lower()
    df["corrected"] = df["corrected"].apply(remove_diacritics)
    df["diff_in_word"] = df.apply(
        lambda row: is_diff_in_word(row["text"], row["corrected"]), axis=1
    )
    df = df[df["diff_in_word"]]

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

                # Only add if both cleaned words are non-empty
                if clean_typo and clean_correct:
                    all_word_pairs.append(
                        {
                            "correct": clean_correct,
                            "typo": clean_typo,
                            "original_sentence": row["text"],
                            "corrected_sentence": row["corrected"],
                        }
                    )

    # Convert to DataFrame for easier viewing and saving
    word_pairs_df = pd.DataFrame(all_word_pairs)

    # Remove cases where both pairs are the same
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

    word_pairs_df = word_pairs_df[
        ~word_pairs_df["typo"].apply(lambda x: is_only_non_alpha(x))
    ]
    word_pairs_df = word_pairs_df[
        ~word_pairs_df["correct"].apply(lambda x: is_only_non_alpha(x))
    ]

    word_pairs_df = word_pairs_df.dropna()
    word_pairs_df = word_pairs_df[~word_pairs_df["correct"].isin(["nan", "NaN", "Nan"])]
    word_pairs_df = word_pairs_df[~word_pairs_df["typo"].isin(["nan", "NaN", "Nan"])]
    word_pairs_df = word_pairs_df[~word_pairs_df["correct"].isna()]
    word_pairs_df = word_pairs_df[~word_pairs_df["typo"].isna()]

    word_pairs_df = word_pairs_df[
        ~word_pairs_df["correct"].apply(lambda x: pd.isnull(x))
    ]
    word_pairs_df = word_pairs_df[~word_pairs_df["typo"].apply(lambda x: pd.isnull(x))]

    word_pairs_df["has_consecutive_diffs"] = word_pairs_df.apply(
        lambda row: has_consecutive_diffs(row["correct"], row["typo"]), axis=1
    )
    word_pairs_df["correct_len"] = word_pairs_df["correct"].apply(len)
    word_pairs_df["typo_len"] = word_pairs_df["typo"].apply(len)
    word_pairs_df["diff_count"] = word_pairs_df.apply(
        lambda row: diff_count(row["correct"], row["typo"]), axis=1
    )

    single_char_typos = word_pairs_df[
        (~word_pairs_df["has_consecutive_diffs"])
        & (word_pairs_df["correct_len"] == word_pairs_df["typo_len"])
        & (word_pairs_df["edit_distance"] == 1)
        & (word_pairs_df["correct_len"] > 1)
    ]

    char_counts = defaultdict(Counter)
    for row_id, row_df in tqdm.tqdm(single_char_typos.iterrows()):
        info = extract_typo_info(row_df["correct"], row_df["typo"])
        if info:
            char_counts[info["correct_char"]][info["typo_char"]] += 1

    new_dict = {}
    letters = "abcdefghijklmnopqrstuvwxyz"
    for c in letters:
        norm_dict = {}
        for k, v in char_counts[c].items():
            if k in letters:
                norm_dict[k] = v / sum(char_counts[c].values())
        new_dict[c] = norm_dict

    with open(os.path.join("data", "char_counts.pkl"), "wb") as f:
        pickle.dump(new_dict, f)
