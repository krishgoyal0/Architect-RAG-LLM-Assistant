# Architect-RAG-LLM-Assistant
A project showcasing AI algorithms and models with a focus on Machine Learning, NLP, and Deep Learning. Includes dataset analysis, debugging, and research-driven enhancements using Python and Data Structures.

🏛 Architect-RAG-LLM-Assistant

An AI-powered Retrieval-Augmented Generation (RAG) assistant for
architecture, urban planning, and building codes.
This project collects, cleans, and processes architecture-related PDFs
(case studies, codes, material guides, and references), and prepares
them for use in LLM pipelines.

------------------------------------------------------------------------

🚀 Features

-   📂 Organized dataset → PDFs categorized into:
    -   building_codes
    -   case_studies
    -   material_guide
    -   misc
-   🧹 Automated PDF Cleaning
    -   Extracts clean text from PDFs (including OCR fallback with
        Tesseract).
    -   Handles corrupted/empty PDFs gracefully.
-   📑 Chunking Pipeline
    -   Splits large documents into manageable JSONL chunks for LLM
        ingestion.
-   ⚡ RAG-ready dataset → Creates clean JSONL outputs for vector
    database ingestion.

