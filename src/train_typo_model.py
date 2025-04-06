import os
import math
from typing import List, Dict, Tuple
import logging
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 100):
        super(PositionalEncoder, self).__init__()

        # Create positional encodings
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        Returns:
            Tensor, shape [batch_size, seq_len, embedding_dim * 2]
        """
        # Get positional encodings for the current sequence length
        pos_enc = self.pe[: x.size(1)]  # [seq_len, embedding_dim]
        # Expand to match batch dimension
        pos_enc = pos_enc.unsqueeze(0).expand(
            x.size(0), -1, -1
        )  # [batch_size, seq_len, embedding_dim]
        # Concatenate input embeddings with positional encodings
        x = torch.cat([x, pos_enc], dim=-1)
        return x


class CharTokenizer:
    def __init__(self):
        self.unk_token = "UNK"
        self.pad_token = "PAD"
        self.eos_token = "EOS"  # End of string token
        self.sos_token = "SOS"  # Start of string token
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size = 0

    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        # Get all unique alphabetic characters
        chars = sorted(list(set(c for text in texts for c in str(text) if c.isalpha())))

        # Add special tokens
        chars = [self.pad_token, self.unk_token, self.sos_token, self.eos_token] + chars

        # Create mappings
        self.char_to_idx = {char: i for i, char in enumerate(chars)}
        self.idx_to_char = {i: char for i, char in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str, max_length: int = None) -> List[int]:
        """Convert text to indices."""
        # Add SOS token at the beginning
        indices = [self.char_to_idx[self.sos_token]]

        # Convert each character to its index, using UNK for non-alphabetic
        indices.extend(
            [
                (
                    self.char_to_idx[self.unk_token]
                    if (not c.isalpha()) | (c not in self.char_to_idx)
                    else self.char_to_idx[c]
                )
                for c in str(text)
            ]
        )

        # Add EOS token
        indices.append(self.char_to_idx[self.eos_token])

        # Pad if max_length is specified
        if max_length is not None:
            if len(indices) < max_length:
                indices.extend(
                    [self.char_to_idx[self.pad_token]] * (max_length - len(indices))
                )
            else:
                indices = indices[:max_length]

        return indices

    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text."""
        # Convert indices to characters, stopping at EOS token
        chars = []
        for idx in indices:
            char = self.idx_to_char[idx]
            if char == self.eos_token:
                break
            if char not in [self.pad_token, self.sos_token]:
                chars.append(char)
        return "".join(chars)

    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return self.vocab_size


class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=48,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model // 2)
        self.pos_encoder = PositionalEncoder(d_model=d_model // 2)

        # Create decoder layers manually
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_seq, tgt_seq):
        """
        Args:
            input_seq: (B, S) input sequence
            tgt_seq: (B, T) target sequence so far (for teacher forcing or generation)
        """
        # Embed and encode positions
        input_emb = self.pos_encoder(self.embedding(input_seq))
        tgt_emb = self.pos_encoder(self.embedding(tgt_seq))

        # Generate attention mask for auto-regressive decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(1)).to(
            tgt_seq.device
        )

        # Decode
        out = self.decoder(tgt=tgt_emb, memory=input_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)  # (B, T, vocab_size)

    def generate(self, input_seq, tokenizer, max_len=30):
        """Generate output sequence given input."""
        self.eval()
        input_ids = tokenizer.encode(input_seq)
        input_tensor = (
            torch.tensor(input_ids).unsqueeze(0).to(next(self.parameters()).device)
        )  # (1, S)

        # Start with SOS
        generated = [tokenizer.char_to_idx[tokenizer.sos_token]]
        for _ in range(max_len):
            tgt_tensor = (
                torch.tensor(generated).unsqueeze(0).to(next(self.parameters()).device)
            )  # (1, T)
            with torch.no_grad():
                logits = self.forward(input_tensor, tgt_tensor)
                next_token_logits = logits[0, -1]  # (vocab_size,)
                next_token = torch.argmax(next_token_logits).item()
                generated.append(next_token)
                if next_token == tokenizer.char_to_idx[tokenizer.eos_token]:
                    break
        return tokenizer.decode(generated)


class TypoDataset(Dataset):
    def __init__(self, texts: List[Tuple[str, str]], tokenizer, max_length: int = 30):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        original_text, typo_text = self.texts[idx]
        # Encode both texts
        original_ids = self.tokenizer.encode(original_text, max_length=self.max_length)
        typo_ids = self.tokenizer.encode(typo_text, max_length=self.max_length)
        # Convert to tensors
        original_tensor = torch.tensor(original_ids, dtype=torch.long)
        typo_tensor = torch.tensor(typo_ids, dtype=torch.long)
        return original_tensor, typo_tensor


def contains_non_alpha(s: str) -> bool:
    """Return True if string contains any non-alphabetic character."""
    return any(not c.isalpha() for c in str(s))


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/typo_generator_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info(f"Logging started. Log file: {log_file}")
    
    # Log GPU information
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "None"
        logging.info(f"GPU available: Yes")
        logging.info(f"GPU count: {gpu_count}")
        logging.info(f"GPU name: {gpu_name}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    else:
        logging.info("GPU available: No, using CPU")
    
    return log_file


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train the model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.char_to_idx[tokenizer.pad_token], reduction="sum"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    logging.info(f"Starting training with {num_epochs} epochs")
    logging.info(f"Using device: {device}")
    logging.info(f"Model architecture: {model}")
    logging.info(f"Training samples: {len(train_loader.dataset)}")
    logging.info(f"Validation samples: {len(val_loader.dataset)}")

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        total_train_tokens = 0

        for original, typo in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            original, typo = original.to(device), typo.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(original, typo[:, :-1])
            target = typo[:, 1:].reshape(-1)

            # Calculate loss and number of non-padding tokens
            loss = criterion(output.view(-1, output.size(-1)), target)
            num_tokens = (
                (target != tokenizer.char_to_idx[tokenizer.pad_token]).sum().item()
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_tokens += num_tokens

        avg_train_loss = total_train_loss / total_train_tokens

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_tokens = 0

        with torch.no_grad():
            for original, typo in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                original, typo = original.to(device), typo.to(device)
                output = model(original, typo[:, :-1])
                target = typo[:, 1:].reshape(-1)

                loss = criterion(output.view(-1, output.size(-1)), target)
                num_tokens = (
                    (target != tokenizer.char_to_idx[tokenizer.pad_token]).sum().item()
                )

                total_val_loss += loss.item()
                total_val_tokens += num_tokens

        avg_val_loss = total_val_loss / total_val_tokens

        # Log epoch results
        logging.info(f"Epoch {epoch+1}/{num_epochs}:")
        logging.info(f"Training Loss (per token): {avg_train_loss:.4f}")
        logging.info(f"Validation Loss (per token): {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            logging.info("Saved best model!")

        # Example generation
        test_word = "necessary"
        generated = model.generate(test_word, tokenizer)
        logging.info(f"Test word: {test_word}")
        logging.info(f"Generated: {generated}")


def generate_examples(model, tokenizer, test_words: List[str]):
    """Generate example typos for a list of test words."""
    logging.info("\nExample Typo Generation:")
    logging.info("-" * 50)
    for word in test_words:
        generated = model.generate(word, tokenizer)
        logging.info(f"Original: {word}")
        logging.info(f"Generated: {generated}")
        logging.info("-" * 50)


if __name__ == "__main__":
    # Set up logging
    log_file = setup_logging()
    logging.info("Starting typo generator training")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # loading the data
    logging.info("Loading training data...")
    typo_df = pd.read_csv(os.path.join("data", "word_pairs.csv"))
    logging.info(f"Loaded {len(typo_df)} examples from dataset")

    word_pairs = []
    for row_id, row_df in typo_df.iterrows():
        word_correct = row_df["correct"]
        word_typoed = row_df["typo"]
        word_pairs.append((word_correct, word_typoed))

    logging.info(f"Filtered to {len(word_pairs)} valid word pairs")

    # Initialize tokenizer and fit on all words
    logging.info("Initializing tokenizer...")
    tokenizer = CharTokenizer()
    all_words = [word for pair in word_pairs for word in pair]
    tokenizer.fit(all_words)
    logging.info(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")

    # Create dataset
    logging.info("Creating datasets...")
    dataset = TypoDataset(word_pairs, tokenizer)

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    logging.info(f"Train set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize and train model
    logging.info("Initializing model...")
    model = CharTransformer(vocab_size=tokenizer.get_vocab_size())
    
    # Move model to device and log memory usage if GPU
    model = model.to(device)
    if device.type == "cuda":
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        logging.info(f"GPU memory cached: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")
    
    logging.info("Starting training...")
    train_model(model, train_loader, val_loader, tokenizer, num_epochs=15, device=device)

    # Load best model
    logging.info("Loading best model for example generation...")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)  # Ensure model is on the correct device

    # Generate examples
    test_words = [
        "necessary",
        "accommodation",
        "definitely",
        "separate",
        "occurrence",
        "embarrass",
        "privilege",
        "conscience",
        "maintenance",
        "recommend",
        "variable",
        "very",
        "cemlyn",
        "cymru",
        "souvent",
        "mangeais",
        "aerated",
        "anemone",
        "sunny",
        "theory",
        "curiosity",
    ]
    logging.info("Generating example typos...")
    generate_examples(model, tokenizer, test_words)

    logging.info("Training and example generation complete!")
    logging.info(f"Log file saved at: {log_file}")
