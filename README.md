# CODEBOOKS - Digital Humanities Metadata Pipeline

A simplified Python application for extracting and managing metadata from archival documents using OCR and AI-powered prompts.

## Features

- **File Ingestion**: Add individual files or entire directories with variable structure
- **OCR Processing**: Extract text using Tesseract (extensible for additional OCR models)
- **AI Metadata Extraction**: Generate Dublin Core metadata using customizable prompts
- **Interactive Approval**: Review and approve/reject AI-generated metadata suggestions
- **Ledger Management**: Track all operations and store metadata in a CSV ledger
- **Error Handling**: Clear and reimport problematic entries with confirmation

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

## Usage

1. Run the application:
```bash
python main.py
```

2. **Add Files**: Use "Add Files/Directory" to import documents
3. **Run OCR**: Process documents with "Run Tesseract OCR"
4. **Setup AI**: Enter your OpenAI API key with "Setup AI Prompts"
5. **Generate Metadata**: Extract Dublin Core metadata with "Generate Metadata"

## File Structure

```
CODEBOOKS/
├── main.py              # Main GUI application
├── src/
│   ├── ledger_manager.py    # Metadata ledger management
│   ├── ocr_processor.py     # OCR processing (extensible)
│   └── prompt_processor.py  # AI prompt processing
├── prompts/             # Dublin Core prompt templates
│   ├── title.txt
│   ├── creator.txt
│   ├── subject.txt
│   ├── description.txt
│   ├── date.txt
│   └── type.txt
├── data/               # Generated ledger files
└── requirements.txt    # Python dependencies
```

## Dublin Core Fields Supported

- Title
- Creator
- Subject
- Description
- Date
- Type
- (Additional fields can be added by creating prompt files)

## Extending the System

### Adding New OCR Models

1. Create a new class inheriting from `BaseOCRModel` in `ocr_processor.py`
2. Implement the `process_image()` method and `name` property
3. Add the model to the OCRProcessor using `add_model()`

### Adding New Dublin Core Fields

1. Create a new `.txt` file in the `prompts/` directory
2. Write a prompt that instructs the AI how to extract that metadata field
3. The field will automatically appear in the metadata generation interface

## Data Management

- All metadata is stored in `metadata_ledger.csv`
- Each file gets a unique ID for tracking
- Operations can be run on individual files or in batch
- Failed operations can be cleared and re-run
- Confirmation required for data deletion

## Requirements

- Python 3.7+
- OpenAI API key (for AI metadata extraction)
- Tesseract OCR installed on system
- Supported file formats: JPG, PNG, TIFF, PDF