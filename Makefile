# Makefile for Typosmith project

# Dataset URL
DATASET_URL := https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz
DATASET_FILE := data/github-typo-corpus.v1.0.0.jsonl.gz
DECOMPRESSED_FILE := data/github-typo-corpus.v1.0.0.jsonl

.PHONY: all download clean

all: download

# Create data directory if it doesn't exist
data:
	mkdir -p data

# Download and decompress the dataset
download: data
	@echo "Downloading GitHub Typo Corpus dataset..."
	curl -L $(DATASET_URL) -o $(DATASET_FILE)
	@echo "Decompressing dataset..."
	$(DECOMPRESS_CMD)

# Format Python code using black
fmt: 
	@echo "Formatting Python code..."
	black .

# Clean up downloaded files
clean:
	$(RM_CMD) $(DATASET_FILE) $(DECOMPRESSED_FILE)
	$(RMDIR_CMD) data 2>/dev/null || true

create-char-counts:
	python src/create_char_counts.py

create-plots:
	python src/plot_adjacent_typos.py
	python src/plot_non_adjacent_typos.py

create-char-counts-and-plots:
	make create-char-counts
	make create-plots
