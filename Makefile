# Makefile for Typosmith project

# Dataset URL
DATASET_URL := https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz
DATASET_FILE := data/github-typo-corpus.v1.0.0.jsonl.gz
DECOMPRESSED_FILE := data/github-typo-corpus.v1.0.0.jsonl

# Detect OS
ifeq ($(OS),Windows_NT)
    # Use full path to 7zip executable
    SEVENZIP := "C:\Program Files\7-Zip\7z.exe"
    DECOMPRESS_CMD = $(SEVENZIP) x $(DATASET_FILE) -o$(dir $(DATASET_FILE))
    RM_CMD = del /Q
    RMDIR_CMD = rmdir /Q
else
    DECOMPRESS_CMD = gzip -d $(DATASET_FILE)
    RM_CMD = rm -f
    RMDIR_CMD = rmdir
endif

.PHONY: all download clean check-7zip

all: download

# Check if 7zip is installed (Windows only)
check-7zip:
ifeq ($(OS),Windows_NT)
	@if not exist "$(SEVENZIP)" ( \
		echo "Error: 7-Zip not found at $(SEVENZIP)" && \
		echo "Please install 7-Zip from https://www.7-zip.org/" && \
		exit 1 \
	)
endif

# Create data directory if it doesn't exist
data:
	mkdir -p data

# Download and decompress the dataset
download: check-7zip data
	@echo "Downloading GitHub Typo Corpus dataset..."
	curl -L $(DATASET_URL) -o $(DATASET_FILE)
	@echo "Decompressing dataset..."
	$(DECOMPRESS_CMD)

# Clean up downloaded files
clean:
	$(RM_CMD) $(DATASET_FILE) $(DECOMPRESSED_FILE)
	$(RMDIR_CMD) data 2>/dev/null || true 