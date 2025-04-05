import os
import json
from typing import Iterator, List, Dict, Any

import tqdm
import pandas as pd

FILE_PATH = os.path.join(
    "data", "github-typo-corpus.v1.0.0.jsonl", "github-typo-corpus.v1.0.0.jsonl"
)
OUTPUT_PATH = os.path.join("data", "github-typo-corpus.v1.0.0.csv")


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
                typo["src"]["lang"] == "eng"
                and typo["tgt"]["lang"] == "eng"
                and typo["prob_typo"] > 0.8
            ):
                yield original_string, corrected_string
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
