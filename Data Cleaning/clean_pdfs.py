import json
import re
import hashlib
from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# ========= CONFIG =========
# Automatically resolve paths relative to script location
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../Dataset_PDFs").resolve()
OUT = (BASE_DIR / "../cleaned").resolve()

# OCR settings
OCR_LANG = "eng"
# ==========================

def clean_text(text):
    """Clean extracted PDF text."""
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces/newlines
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)  # remove "Page X of Y"
    text = re.sub(r'^\s*\d+\s*$', '', text)  # remove lines with only numbers
    return text.strip()

def hash_text(text):
    """Create a hash to detect duplicate pages."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def extract_text_with_ocr(page):
    """Fallback OCR if no extractable text is found."""
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(img, lang=OCR_LANG)

def process_pdf(pdf_path, category, out_f):
    """Process one PDF into JSONL records."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Cannot open {pdf_path.name}: {e}")
        return 0

    seen_hashes = set()
    page_count = 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")

        if not text.strip():
            text = extract_text_with_ocr(page)

        text = clean_text(text)

        if not text:
            continue

        text_hash = hash_text(text)
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        rec = {
            "doc_id": pdf_path.stem,
            "file": pdf_path.name,
            "category": category,
            "page_number": page_num,
            "text": text
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        page_count += 1

    return page_count

def main():
    print(f"[INFO] Looking for PDFs in: {ROOT}")
    OUT.mkdir(exist_ok=True)

    if not ROOT.exists():
        print(f"[ERROR] Data folder not found at: {ROOT}")
        return

    for category_dir in ROOT.iterdir():
        if category_dir.is_dir():
            category = category_dir.name
            pdfs = list(category_dir.glob("*.pdf")) + list(category_dir.glob("*.PDF"))
            print(f"[INFO] Category '{category}' -> {len(pdfs)} PDFs found")

            if not pdfs:
                continue

            output_path = OUT / f"{category}.jsonl"

            total_pages = 0
            with open(output_path, "w", encoding="utf-8") as out_f:
                for pdf in pdfs:
                    pages_written = process_pdf(pdf, category, out_f)
                    print(f"   [OK] {pdf.name} -> {pages_written} pages cleaned")
                    total_pages += pages_written

            print(f"[DONE] Wrote {total_pages} pages for '{category}' -> {output_path}")

    print("[COMPLETE] All PDFs processed.")

if __name__ == "__main__":
    main()
