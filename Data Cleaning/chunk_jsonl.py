import os
import json
import re
import hashlib
import logging
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
from transformers import AutoTokenizer

# ====== CONFIGURATION ======
class ChunkingConfig:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_MAX_TOKENS = 512
    SAFETY_MARGIN = 12
    ALLOWED_TOKENS = MODEL_MAX_TOKENS - SAFETY_MARGIN
    CHUNK_TOKENS = min(1200, ALLOWED_TOKENS)
    OVERLAP = 60
    MIN_TOKENS_PER_CHUNK = 50
    MIN_CHAR_LENGTH = 100
    
    BOILERPLATE_PATTERNS = [
        r"national building code.*",
        r"government of india.*",
        r"all rights reserved.*",
        r"bureau of indian standards.*",
        r"www\.wbdg\.org.*",
        r"^\s*table of contents\s*$",
        r"^\s*copyright.*$",
        r"^\s*page \d+ of \d+\s*$",
        r"^\s*confidential\s*$",
        r"^\s*proprietary\s*$",
    ]
    
    MEANINGLESS_PATTERNS = [
        r"^[0-9\.\s]+$",      # Just numbers and dots
        r"^[A-Z\s]+$",        # All caps (often headings without context)
        r"^\W+$",             # Only punctuation
        r"^.{1,3}$",          # Very short strings
    ]

# ====== SETUP ======
config = ChunkingConfig()
BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR.parent / "cleaned"
OUT_DIR = BASE_DIR.parent / "chunks"
OUT_DIR.mkdir(exist_ok=True)

# ====== LOGGING ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'chunking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====== TOKENIZER ======
try:
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    logger.info(f"Loaded tokenizer: {config.MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise

# ====== UTILITY FUNCTIONS ======
def clean_boilerplate(text: str) -> str:
    """Remove common repeated boilerplate/footer/header lines."""
    if not text or not isinstance(text, str):
        return ""
    
    t = text
    for pat in config.BOILERPLATE_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE | re.MULTILINE)
    
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_boilerplate(text: str) -> bool:
    """Check if text appears to be boilerplate or meaningless content."""
    if not text or len(text.strip()) < 10:
        return True
    
    # Check for meaningless patterns
    for pattern in config.MEANINGLESS_PATTERNS:
        if re.match(pattern, text, flags=re.IGNORECASE):
            return True
    
    # Check if it's mostly non-alphanumeric
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / max(1, len(text)) < 0.3:  # Less than 30% alphabetic
        return True
    
    return False

@lru_cache(maxsize=10000)
def hash_text(text: str) -> str:
    """Cache frequent hash computations."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def safe_tokenize(text: str):
    """Handle tokenization errors gracefully."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        logger.warning(f"Tokenization error: {e}")
        return []

@lru_cache(maxsize=5000)
def cached_token_len(text: str) -> int:
    """Cache token length computations."""
    return len(safe_tokenize(text))

def sliding_windows(token_ids, window=config.ALLOWED_TOKENS, overlap=config.OVERLAP):
    """
    Yield token slices of length <= window with overlap.
    Ensures no slice exceeds ALLOWED_TOKENS.
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    
    i = 0
    n = len(token_ids)
    while i < n:
        j = min(i + window, n)
        yield token_ids[i:j]
        if j >= n:
            break
        i = max(0, j - overlap)

def decode(ids):
    """Safe token decoding."""
    try:
        return tokenizer.decode(ids).strip()
    except Exception as e:
        logger.warning(f"Decoding error: {e}")
        return ""

def is_quality_chunk(chunk: str) -> bool:
    """Check if chunk meets quality standards."""
    if not chunk or not isinstance(chunk, str):
        return False
    
    # Token length check
    if cached_token_len(chunk) < config.MIN_TOKENS_PER_CHUNK:
        return False
    
    # Character length check
    if len(chunk.strip()) < config.MIN_CHAR_LENGTH:
        return False
    
    # Boilerplate check
    if is_boilerplate(chunk):
        return False
    
    # Check for meaningful content (not just numbers/symbols)
    if re.match(r"^[\d\W\s]+$", chunk):  # Only digits, punctuation, whitespace
        return False
    
    # Check sentence structure (rough heuristic)
    sentences = re.split(r'[.!?]+', chunk)
    if len(sentences) < 2 and len(chunk) > 200:  # Long but no sentence breaks
        return False
    
    return True

def chunk_text_no_truncation(text: str):
    """
    Split arbitrarily long text into multiple <= ALLOWED_TOKENS chunks.
    """
    if not text or is_boilerplate(text):
        return
    
    ids = safe_tokenize(text)
    if not ids:
        return
    
    for slice_ids in sliding_windows(ids, window=config.ALLOWED_TOKENS, overlap=config.OVERLAP):
        chunk = decode(slice_ids)
        if chunk and is_quality_chunk(chunk):
            yield chunk

# ====== PROCESSING FUNCTIONS ======
def process_record(rec, seen_hashes):
    """Process a single record and yield valid chunks."""
    if not rec or not isinstance(rec, dict):
        return
    
    text = clean_boilerplate(rec.get("text", ""))
    if not text or is_boilerplate(text):
        return
    
    chunks_generated = 0
    for chunk in chunk_text_no_truncation(text):
        if not chunk:
            continue
        
        # Deduplication
        h = hash_text(chunk)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        # Final safety check
        if cached_token_len(chunk) > config.ALLOWED_TOKENS:
            # Recursively split oversized chunks
            re_ids = safe_tokenize(chunk)
            for sub_ids in sliding_windows(re_ids, window=config.ALLOWED_TOKENS, overlap=config.OVERLAP):
                sub_chunk = decode(sub_ids)
                if is_quality_chunk(sub_chunk):
                    yield {
                        "doc_id": rec.get("doc_id", ""),
                        "file": rec.get("file", ""),
                        "category": rec.get("category", ""),
                        "page_span": [rec.get("page_number", 0), rec.get("page_number", 0)],
                        "text": sub_chunk,
                        "chunk_hash": hash_text(sub_chunk)
                    }
                    chunks_generated += 1
            continue
        
        # Yield valid chunk
        yield {
            "doc_id": rec.get("doc_id", ""),
            "file": rec.get("file", ""),
            "category": rec.get("category", ""),
            "page_span": [rec.get("page_number", 0), rec.get("page_number", 0)],
            "text": chunk,
            "chunk_hash": h
        }
        chunks_generated += 1
    
    return chunks_generated

def process_single_file(jsonl_file, checkpoint_file=None):
    """Process a single JSONL file with checkpoint support."""
    category = jsonl_file.stem
    out_path = OUT_DIR / f"{category}_chunks.jsonl"
    
    # Checkpoint handling
    processed_hashes = set()
    if checkpoint_file and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                processed_hashes.update(line.strip() for line in f)
            logger.info(f"Loaded {len(processed_hashes)} hashes from checkpoint")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    
    seen_hashes = set(processed_hashes)
    kept = 0
    skipped = 0
    
    try:
        # Count total lines for progress bar
        total_lines = sum(1 for _ in open(jsonl_file, 'r', encoding='utf-8'))
        
        with open(out_path, "a" if checkpoint_file else "w", encoding="utf-8") as out_f:
            with open(jsonl_file, "r", encoding="utf-8") as in_f:
                for line_num, line in enumerate(tqdm(in_f, total=total_lines, desc=f"Processing {category}"), 1):
                    try:
                        rec = json.loads(line)
                        for chunk_data in process_record(rec, seen_hashes):
                            out_f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                            kept += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in {jsonl_file} line {line_num}: {e}")
                        skipped += 1
                    except Exception as e:
                        logger.error(f"Unexpected error processing line {line_num}: {e}")
                        skipped += 1
                    
                    # Save checkpoint every 1000 lines
                    if line_num % 1000 == 0 and checkpoint_file:
                        with open(checkpoint_file, 'w') as f:
                            f.writelines(h + '\n' for h in seen_hashes)
                
    except Exception as e:
        logger.error(f"Error processing file {jsonl_file}: {e}")
        return kept, skipped, False
    
    # Final checkpoint save
    if checkpoint_file:
        with open(checkpoint_file, 'w') as f:
            f.writelines(h + '\n' for h in seen_hashes)
    
    return kept, skipped, True

def process_file_wrapper(args):
    """Wrapper for parallel processing."""
    jsonl_file, checkpoint_file = args
    return process_single_file(jsonl_file, checkpoint_file)

# ====== MAIN PROCESS ======
def main():
    logger.info("Starting chunking process")
    
    jsonl_files = list(CLEANED_DIR.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {CLEANED_DIR}")
        return
    
    logger.info(f"Found {len(jsonl_files)} files to process")
    
    # Prepare checkpoint files
    checkpoint_files = []
    for jsonl_file in jsonl_files:
        checkpoint_file = OUT_DIR / f"{jsonl_file.stem}_checkpoint.txt"
        checkpoint_files.append(checkpoint_file)
    
    # Process files (sequential or parallel)
    total_kept = 0
    total_skipped = 0
    
    if len(jsonl_files) > 1 and multiprocessing.cpu_count() > 1:
        # Parallel processing for multiple files
        logger.info("Using parallel processing")
        with ProcessPoolExecutor(max_workers=min(len(jsonl_files), multiprocessing.cpu_count())) as executor:
            args = zip(jsonl_files, checkpoint_files)
            results = list(executor.map(process_file_wrapper, args))
            
            for (kept, skipped, success), jsonl_file in zip(results, jsonl_files):
                if success:
                    logger.info(f"Completed {jsonl_file.name}: kept={kept}, skipped={skipped}")
                    total_kept += kept
                    total_skipped += skipped
                else:
                    logger.error(f"Failed to process {jsonl_file.name}")
    else:
        # Sequential processing
        logger.info("Using sequential processing")
        for jsonl_file, checkpoint_file in zip(jsonl_files, checkpoint_files):
            kept, skipped, success = process_single_file(jsonl_file, checkpoint_file)
            if success:
                logger.info(f"Completed {jsonl_file.name}: kept={kept}, skipped={skipped}")
                total_kept += kept
                total_skipped += skipped
            else:
                logger.error(f"Failed to process {jsonl_file.name}")
    
    logger.info(f"Chunking complete! Total kept: {total_kept}, Total skipped: {total_skipped}")
    logger.info(f"Output directory: {OUT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        raise