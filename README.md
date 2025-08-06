# CODEBOOKS - Digital Humanities Metadata Pipeline

A comprehensive Python application for extracting and managing metadata from archival documents using multiple OCR engines and AI-powered analysis.

## ✨ Features

### 🔍 **Multi-Engine OCR Processing**
- **EasyOCR**: GPU-accelerated AI-powered text extraction
- **Tesseract**: Traditional OCR with advanced preprocessing
- **PyPDF2**: Direct PDF text extraction
- **OpenAI OCR**: GPT-4o vision-based transcription
- **Ollama OCR**: Local LLM image transcription (privacy-focused)

### 🤖 **Dual AI Metadata Generation**
- **OpenAI GPT**: Cloud-based metadata extraction using GPT-4o-mini
- **Ollama Local**: Privacy-focused local LLM processing
- **Smart Prompting**: Optimized prompts for clean metadata extraction
- **Interactive Approval**: Review and approve AI-generated suggestions

### 📊 **Advanced Analysis & Evaluation**
- **OCR Quality Assessment**: Compare engine performance with standard metrics
- **Ground Truth Comparison**: Evaluate accuracy using reference text
- **Visual Analytics**: Charts and graphs for performance analysis
- **Cross-Engine Evaluation**: Comprehensive quality scoring

### 🎨 **Modern Interface**
- **Streamlined Workflow**: Intuitive step-by-step process
- **Resizable Panels**: Customizable workspace layout
- **Tabbed Interface**: Separate processing and analysis views
- **Progress Indicators**: Real-time feedback on operations
- **Keyboard Shortcuts**: Power user efficiency (Ctrl+O, Ctrl+R, F5)

## 🚀 Installation

### 1. Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. OCR Engines Setup

**Tesseract OCR:**
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- macOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

**Ollama (Optional - for local AI):**
```bash
# Install Ollama from https://ollama.ai
# Then pull a vision model:
ollama pull gemma3
```

### 3. API Keys (Optional)
- **OpenAI**: Get API key from https://platform.openai.com/api-keys
- Configure through the application's "Configure AI" button

## 📖 Usage

### Quick Start
```bash
python main.py
```

### Workflow
1. **📁 Add Files**: Import individual files or entire directories
2. **🔍 OCR Processing**: Choose engine (EasyOCR, Tesseract, PyPDF2, OpenAI, Ollama) and run
3. **🤖 AI Setup**: Configure OpenAI API or launch Ollama for local processing
4. **📝 Generate Metadata**: Select AI model and extract Dublin Core metadata
5. **🛠️ Analyze**: Evaluate OCR quality and compare engine performance

### Keyboard Shortcuts
- `Ctrl+O`: Add files
- `Ctrl+R`: Run selected OCR engine
- `F5`: Refresh display

## File Structure

```
CODEBOOKS/
├── main.py                  # Modern GUI application
├── src/
│   ├── ledger_manager.py    # CSV-based metadata management
│   ├── ocr_processor.py     # Multi-engine OCR processing
│   ├── prompt_processor.py  # Dual AI metadata extraction
│   └── ocr_evaluator.py     # Quality assessment & analytics
├── prompts/                 # Dublin Core prompt templates
│   ├── title.txt           # Title extraction prompts
│   ├── creator.txt         # Creator/author prompts
│   ├── subject.txt         # Subject classification
│   ├── description.txt     # Content description
│   ├── date.txt           # Date extraction
│   └── type.txt           # Document type classification
├── metadata_ledger.csv     # Generated metadata database
└── requirements.txt        # Python dependencies

## 🤝 Contributing

Contributions welcome! Areas for development:
- Additional OCR engines
- New AI model integrations  
- Enhanced evaluation metrics
- UI/UX improvements
- Documentation and tutorials

## 📄 License

Open source - see LICENSE file for details.
```

## 📋 Dublin Core Metadata Fields

**Currently Supported:**
- **Title**: Document titles and headings
- **Creator**: Authors, organizations, publishers
- **Subject**: Topics, keywords, classifications
- **Description**: Content summaries and abstracts
- **Date**: Creation, publication, or modification dates
- **Type**: Document format and genre classification

**Extensible**: Add new fields by creating prompt files in `prompts/` directory

## 🔧 Extending the System

### Adding New OCR Engines
```python
class CustomOCRModel(BaseOCRModel):
    def process_image(self, image: Image.Image) -> str:
        # Your OCR implementation
        return extracted_text
    
    @property
    def name(self) -> str:
        return "custom_ocr"

# Add to processor
ocr_processor.add_model("custom", CustomOCRModel())
```

### Adding Dublin Core Fields
1. Create `prompts/new_field.txt` with extraction instructions
2. Field automatically appears in metadata generation interface
3. Results stored in ledger with status tracking

### Custom AI Models
- Extend `PromptProcessor` for new AI providers
- Implement `_generate_custom()` method
- Add to model selection dropdown

## 💾 Data Management

### Ledger System
- **CSV Storage**: All metadata in `metadata_ledger.csv`
- **Unique IDs**: Each file tracked with UUID
- **Status Tracking**: Monitor processing state for all engines
- **Batch Operations**: Process multiple files simultaneously
- **Error Recovery**: Clear and retry failed operations
- **Data Safety**: Confirmation required for deletions

### OCR Results Storage
- **Multi-Engine Support**: Separate columns for each OCR engine
- **Text Preservation**: Full OCR output stored and displayable
- **Quality Metrics**: Performance scores and comparisons
- **Export Ready**: CSV format for analysis in other tools

## 📋 Requirements

### System Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **Memory**: 4GB RAM minimum (8GB+ for GPU acceleration)
- **Storage**: 1GB free space for models and data

### Optional Components
- **OpenAI API Key**: For cloud-based AI metadata generation
- **CUDA GPU**: For EasyOCR acceleration (optional)
- **Ollama**: For local AI processing (privacy-focused)

### Supported File Formats
- **Images**: JPG, JPEG, PNG, TIF, TIFF
- **Documents**: PDF (both text and image-based)
- **Batch Processing**: Entire directories with mixed formats

## 🎯 Use Cases

- **Digital Archives**: Batch metadata extraction for historical documents
- **Library Sciences**: Cataloging and classification workflows
- **Research Projects**: OCR quality assessment and comparison
- **Document Digitization**: Automated metadata generation pipelines
- **Privacy-Sensitive Work**: Local processing with Ollama integration