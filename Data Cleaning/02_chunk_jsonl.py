import os
import json
from pathlib import Path
from transformers import AutoTokenizer

# Path setup
BASE_DIR = Path(__file__).resolve().parent  # Data Cleaning folder
CLEANED_DIR = BASE_DIR.parent / "cleaned"   # cleaned folder at root level
OUT_DIR = BASE_DIR.parent / "chunks"
OUT_DIR.mkdir(exist_ok=True)

# Tokenizer (free model for embedding prep)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, chunk_tokens=500, overlap=60):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    i = 0
    while i < len(tokens):
        j = min(i + chunk_tokens, len(tokens))
        chunk = tokenizer.decode(tokens[i:j])
        yield chunk
        i = j - overlap

# Process all cleaned JSONL files
for jsonl_file in CLEANED_DIR.glob("*.jsonl"):
    category = jsonl_file.stem
    out_path = OUT_DIR / f"{category}_chunks.jsonl"

    with open(out_path, "w", encoding="utf-8") as out_f:
        with open(jsonl_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                rec = json.loads(line)
                for chunk in chunk_text(rec["text"]):
                    chunk_rec = {
                        "doc_id": rec["doc_id"],
                        "file": rec["file"],
                        "category": rec["category"],
                        "page_span": [rec["page_number"], rec["page_number"]],
                        "text": chunk
                    }
                    out_f.write(json.dumps(chunk_rec, ensure_ascii=False) + "\n")

    print(f"✅ Chunked {jsonl_file.name} → {out_path}")

print("🎯 All JSONL files have been chunked.")
