# ARCHIVE ANALYZER
## AI-Powered Archival Document Analysis

Archive Analyzer is a comprehensive desktop application for analyzing historical documents and archival collections using artificial intelligence. It combines multiple OCR engines, natural language processing, and machine learning to extract insights from digitized archival materials.

## 🎯 Key Features

### 📄 Multi-Engine OCR Processing
- **EasyOCR** - AI-powered text recognition
- **Tesseract** - Traditional OCR engine
- **PyPDF2** - Direct PDF text extraction
- **OpenAI Vision** - Advanced AI transcription
- **Ollama OCR** - Local LLM transcription
- **Ground Truth Evaluation** - Compare and select best OCR results

### 🏷️ Named Entity Recognition
- Extract people, organizations, and places from documents
- Merge similar entities automatically
- AI-powered entity analysis with document citations
- Browse entities by type with document references

### 📊 Advanced Analysis
- **Topic Modeling** - Discover themes using LDA, NMF, or BERT
- **Timeline Analysis** - Extract and visualize temporal patterns
- **Geographic Mapping** - Map locations mentioned in documents
- **Document Relationships** - Analyze connections between files

### 🔍 Interactive Exploration
- **Document Viewer** - View OCR results alongside original images
- **File Selection** - Choose subsets for targeted analysis
- **Date Range Filtering** - Focus on specific time periods
- **Archival Structure** - Organize by collection/box/folder hierarchy

## 🚀 Quick Start

### Prerequisites
```bash
pip install tkinter pandas numpy scikit-learn spacy matplotlib pillow
pip install easyocr pytesseract PyPDF2 openai ollama
python -m spacy download en_core_web_sm
```

### Installation
1. Clone the repository
2. Install dependencies
3. Run the application:
```bash
python main.py
```

## 📋 Workflow

### 1️⃣ File Management
- Add files or directories to analyze
- Supports PDF, TIFF, JPG, PNG formats
- Automatic file discovery and organization

### 2️⃣ AI Configuration
- Configure OpenAI API keys
- Set up Ollama local models
- Choose OCR engines and parameters

### 3️⃣ Processing
- Batch process multiple files
- Document type classification
- Multi-engine OCR extraction

### 4️⃣ Ground Truth
- Evaluate OCR quality across engines
- Set optimal OCR source for analysis
- Quality metrics and similarity analysis

### 5️⃣ Content Analysis
- Generate metadata fields
- Extract named entities
- Analyze document relationships

### 6️⃣ Advanced Analysis
- Topic modeling and theme discovery
- Timeline extraction and visualization
- Geographic analysis and mapping

## 🛠️ Technical Architecture

### Core Components
- **LedgerManager** - Document tracking and metadata storage
- **OCRProcessor** - Multi-engine text extraction
- **NERProcessor** - Named entity recognition
- **TopicModeler** - Theme discovery algorithms
- **TimelineExtractor** - Temporal pattern analysis
- **GeoMapper** - Geographic location mapping

### AI Integration
- **OpenAI GPT** - Advanced text analysis and entity explanation
- **Ollama** - Local LLM processing for privacy
- **spaCy** - Natural language processing pipeline
- **scikit-learn** - Machine learning algorithms

## 📈 Use Cases

### Historical Research
- Analyze correspondence collections
- Extract biographical information
- Map historical networks and relationships

### Digital Humanities
- Large-scale text analysis
- Temporal pattern discovery
- Geographic visualization of historical events

### Archival Processing
- Automated metadata generation
- Quality assessment of digitization
- Content discovery and organization

## 🔧 Configuration

### AI Models
- **OpenAI**: Requires API key for GPT models
- **Ollama**: Local installation for privacy-focused processing
- **OCR Engines**: Multiple options for different document types

### File Organization
- Supports archival hierarchy (Collection/Box/Folder)
- Automatic path parsing and organization
- Flexible file naming conventions

## 📊 Output Formats

### Data Export
- **CSV** - Tabular data for spreadsheet analysis
- **JSON** - Structured data for further processing
- **TimelineJS** - Interactive timeline visualization
- **Geographic Data** - Location coordinates and frequencies

### Visualizations
- Interactive timeline charts
- Document frequency histograms
- Topic modeling results
- Geographic heat maps

## 🤝 Contributing

Archive Analyzer is designed for extensibility:
- Modular architecture for easy component addition
- Plugin system for new analysis methods
- Configurable AI model integration
- Open source libraries and standards

## 📄 License

This project is developed for academic and research use. Please cite appropriately when using in scholarly work.

## 🆘 Support

For issues, feature requests, or questions:
- Check the documentation in the `/docs` folder
- Review the example workflows
- Examine the configuration files for customization options

---

**Archive Analyzer** - Transforming archival research through AI-powered document analysis.