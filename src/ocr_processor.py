#!/usr/bin/env python3
"""
OCR Processor for extracting text from images and PDFs
Extensible design for multiple OCR models
"""

import os
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
from pdf2image import convert_from_path
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from openai import OpenAI
    import base64
    import io
except ImportError:
    OpenAI = None

class BaseOCRModel(ABC):
    """Base class for OCR models"""
    
    @abstractmethod
    def process_image(self, image: Image.Image) -> str:
        """Process a PIL Image and return text"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the OCR model"""
        pass

class EasyOCRModel(BaseOCRModel):
    """EasyOCR implementation"""
    
    def __init__(self):
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
            print("✅ EasyOCR initialized with GPU support")
        except Exception as e:
            print(f"⚠️ GPU initialization failed, falling back to CPU: {e}")
            self.reader = easyocr.Reader(['en'], gpu=False)
    
    def process_image(self, image: Image.Image) -> str:
        img_array = np.array(image)
        results = self.reader.readtext(img_array)
        return '\n'.join([result[1] for result in results])
    
    @property
    def name(self) -> str:
        return "easyocr"

class TesseractOCRModel(BaseOCRModel):
    """Tesseract OCR implementation"""
    
    def __init__(self):
        if pytesseract is None:
            raise ImportError("pytesseract not available")
        
        # Try to find Tesseract executable on Windows
        import platform
        if platform.system() == 'Windows':
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
    
    def process_image(self, image: Image.Image) -> str:
        # Convert to grayscale for better Tesseract performance
        if image.mode != 'L':
            image = image.convert('L')
        
        # Use Tesseract configuration for better text recognition
        #custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}"\'-/\n '
        
        try:
            # Try with custom config first
            text = pytesseract.image_to_string(image)
            if text.strip():  # If we got meaningful text
                return text
        except:
            pass
        
        # Fallback to default settings
        return pytesseract.image_to_string(image)
    
    @property
    def name(self) -> str:
        return "tesseract"

class PyPDF2OCRModel(BaseOCRModel):
    """PyPDF2 text extraction implementation"""
    
    def __init__(self):
        if PyPDF2 is None:
            raise ImportError("PyPDF2 not available")
    
    def process_image(self, image: Image.Image) -> str:
        """Convert image to temporary PDF and extract text"""
        import tempfile
        import os
        
        try:
            # Create temporary PDF from image
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_pdf_path = temp_pdf.name
                
            # Convert image to PDF
            image.save(temp_pdf_path, "PDF", resolution=100.0)
            
            # Extract text from temporary PDF
            result = self.process_pdf(temp_pdf_path)
            
            # Clean up temporary file
            os.unlink(temp_pdf_path)
            
            return result if result != "No text found in PDF" else "No embedded text found in image (image-based content)"
            
        except Exception as e:
            return f"Error converting image to PDF: {str(e)}"
    
    def process_pdf(self, pdf_path: str) -> str:
        """Extract text directly from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"[Page {page_num}]\n{page_text}")
                
                return "\n\n".join(text_parts) if text_parts else "No text found in PDF"
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"
    
    @property
    def name(self) -> str:
        return "pypdf2"

class OpenAIOCRModel(BaseOCRModel):
    """OpenAI GPT-4o image transcription implementation"""
    
    def __init__(self, api_key: str = None):
        if OpenAI is None:
            raise ImportError("OpenAI not available")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        self.client = OpenAI()
    
    def process_image(self, image: Image.Image) -> str:
        """Process image with OpenAI GPT-4o"""
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Send to GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe this document accurately, preserving all text content."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    @property
    def name(self) -> str:
        return "openai_ocr"

class OCRProcessor:
    """Handles OCR processing with multiple model support"""
    
    def __init__(self):
        self.max_pixels = 80_000_000  # Image size threshold
        self.models = {}
        self.default_model = None
        
        # Try to initialize OCR models
        try:
            self.models['easyocr'] = EasyOCRModel()
            self.default_model = 'easyocr'
        except Exception:
            pass
        
        try:
            self.models['tesseract'] = TesseractOCRModel()
            if not self.default_model:
                self.default_model = 'tesseract'
        except Exception:
            pass
        
        try:
            self.models['pypdf2'] = PyPDF2OCRModel()
            if not self.default_model:
                self.default_model = 'pypdf2'
        except Exception:
            pass
    
    def add_openai_ocr(self, api_key: str):
        """Add OpenAI OCR model with API key"""
        try:
            self.models['openai_ocr'] = OpenAIOCRModel(api_key)
            print("✅ OpenAI OCR initialized")
        except Exception as e:
            print(f"❌ OpenAI OCR initialization failed: {e}")
    
    def preprocess_image(self, img: Image.Image, model_name: str = None) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large
        if img.width * img.height > self.max_pixels:
            print(f"🔧 Resizing large image: {img.width}x{img.height}")
            scale_factor = (self.max_pixels / (img.width * img.height)) ** 0.5
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"✅ Resized to: {img.width}x{img.height}")
        
        # Different preprocessing for different OCR engines
        if model_name == 'tesseract':
            # Tesseract-specific preprocessing
            # Convert to grayscale
            img = img.convert('L')
            
            # Enhance contrast more aggressively
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            
            # Apply threshold to create binary image
            import numpy as np
            img_array = np.array(img)
            threshold = np.mean(img_array)
            binary_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
            img = Image.fromarray(binary_array, mode='L')
        else:
            # EasyOCR preprocessing
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            img = img.filter(ImageFilter.MedianFilter(size=3))
        
        return img
    
    def add_model(self, name: str, model: BaseOCRModel):
        """Add a new OCR model"""
        self.models[name] = model
    
    def get_available_models(self) -> list:
        """Get list of available OCR models"""
        return list(self.models.keys())
    
    def process_image(self, image_path: str, model_name: str = None) -> Tuple[str, str]:
        """Process a single image file with specified OCR model"""
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            return f"Error: Unknown model {model_name}", "error"
        
        try:
            img = Image.open(image_path)
            img = self.preprocess_image(img, model_name)
            text = self.models[model_name].process_image(img)
            return text, "completed"
        except Exception as e:
            error_msg = str(e)
            if "Tesseract not found" in error_msg:
                error_msg = "Tesseract OCR not installed. Install from: https://github.com/UB-Mannheim/tesseract/wiki"
            return f"Error: {error_msg}", "error"
    
    def process_pdf(self, pdf_path: str, model_name: str = None) -> Tuple[str, str]:
        """Process PDF by converting to images and running OCR on each page"""
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            return f"Error: Unknown model {model_name}", "error"
        
        try:
            # PyPDF2 extracts text directly without OCR
            if model_name == 'pypdf2':
                text = self.models[model_name].process_pdf(pdf_path)
                return text, "completed"
            
            # Other models need image conversion
            images = convert_from_path(pdf_path, dpi=200)
            all_text = []
            
            for i, image in enumerate(images):
                print(f"Processing page {i+1}/{len(images)}")
                image = self.preprocess_image(image, model_name)
                page_text = self.models[model_name].process_image(image)
                all_text.append(f"[Page {i+1}]\n{page_text}")
            
            combined_text = "\n\n".join(all_text)
            return combined_text, "completed"
        except Exception as e:
            return f"Error: {str(e)}", "error"
    
    def process_file(self, file_path: str, model_name: str = None) -> Tuple[str, str]:
        """Process a file based on its type"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            # PyPDF2 can now handle images by converting to PDF first
            if model_name == 'pypdf2':
                try:
                    img = Image.open(file_path)
                    text = self.models[model_name].process_image(img)
                    return text, "completed"
                except Exception as e:
                    return f"Error: {str(e)}", "error"
            else:
                return self.process_image(file_path, model_name)
        elif file_ext == '.pdf':
            return self.process_pdf(file_path, model_name)
        else:
            return f"Unsupported file type: {file_ext}", "error"