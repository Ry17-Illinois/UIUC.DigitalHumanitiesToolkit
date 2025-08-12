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
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from openai import OpenAI
    import base64
    import io
except ImportError:
    OpenAI = None

try:
    import ollama
except ImportError:
    ollama = None

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

class PyMuPDFOCRModel(BaseOCRModel):
    """PyMuPDF text extraction and OCR implementation"""
    
    def __init__(self):
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not available")
    
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
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if page_text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
                else:
                    # If no text found, try OCR on page image using PyMuPDF's built-in capabilities
                    try:
                        # Get page as image
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                        img_data = pix.tobytes("png")
                        
                        # Convert to PIL Image
                        from io import BytesIO
                        img = Image.open(BytesIO(img_data))
                        
                        # Try EasyOCR if available
                        try:
                            import easyocr
                            import numpy as np
                            reader = easyocr.Reader(['en'], gpu=False)  # Use CPU to avoid GPU issues
                            img_array = np.array(img)
                            ocr_results = reader.readtext(img_array)
                            ocr_text = '\n'.join([result[1] for result in ocr_results if result[2] > 0.5])  # Confidence threshold
                            if ocr_text.strip():
                                text_parts.append(f"[Page {page_num + 1} - OCR]\n{ocr_text}")
                            else:
                                text_parts.append(f"[Page {page_num + 1}]\n[Image-based content - low confidence OCR]")
                        except ImportError:
                            text_parts.append(f"[Page {page_num + 1}]\n[Image-based content - EasyOCR not available]")
                        except Exception as ocr_error:
                            text_parts.append(f"[Page {page_num + 1}]\n[Image-based content - OCR failed: {str(ocr_error)}]")
                    except Exception as img_error:
                        text_parts.append(f"[Page {page_num + 1}]\n[Image extraction failed: {str(img_error)}]")
            
            doc.close()
            return "\n\n".join(text_parts) if text_parts else "No text found in PDF"
            
        except Exception as e:
            return f"Error processing PDF with PyMuPDF: {str(e)}"
    
    @property
    def name(self) -> str:
        return "pymupdf"

class OllamaOCRModel(BaseOCRModel):
    """Ollama local LLM image transcription implementation"""
    
    def __init__(self, model_name: str = "gemma3"):
        if ollama is None:
            raise ImportError("ollama not available")
        
        self.model_name = model_name
        
        # Test if Ollama is running and model is available
        try:
            models = ollama.list()
            available_models = [m['name'] for m in models['models']]
            if not any(model_name in m for m in available_models):
                print(f"⚠️ Model {model_name} not found. Available models: {available_models}")
        except Exception as e:
            print(f"⚠️ Could not connect to Ollama: {e}")
    
    def process_image(self, image: Image.Image) -> str:
        """Process image with Ollama local LLM"""
        import time
        import threading
        
        try:
            print(f"🔄 Starting Ollama OCR with model {self.model_name}...")
            
            # Convert image to bytes
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            print(f"📷 Image size: {len(img_bytes)} bytes")
            
            start_time = time.time()
            print(f"⏱️ Sending request to Ollama at {time.strftime('%H:%M:%S')}")
            
            # Check if this is a timeout situation from PDF processing
            if hasattr(threading.current_thread(), '_timeout_applied'):
                print("⚠️ Already in timeout context, using shorter timeout")
                timeout_duration = 30
            else:
                timeout_duration = 90
            
            # Use threading with timeout
            result = [None]
            error = [None]
            
            def ollama_request():
                try:
                    response = ollama.generate(
                        model=self.model_name,
                        prompt="Transcribe all text from this image. Only return the text content, no additional commentary.",
                        images=[img_bytes],
                        options={
                            'temperature': 0.1,
                            'top_p': 0.9,
                            'num_predict': 1000
                        }
                    )
                    result[0] = response['response']
                except Exception as e:
                    error[0] = str(e)
            
            thread = threading.Thread(target=ollama_request)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_duration)  # Dynamic timeout
            
            if thread.is_alive():
                print("⏰ Ollama request timed out after 90 seconds")
                return "Error: Ollama request timed out - model may be overloaded or stuck"
            
            if error[0]:
                print(f"❌ Ollama error: {error[0]}")
                return f"Error: {error[0]}"
            
            elapsed = time.time() - start_time
            print(f"✅ Ollama response received in {elapsed:.2f}s")
            
            return result[0] or "No response from Ollama"
            
        except Exception as e:
            print(f"❌ Ollama error: {str(e)}")
            return f"Error: {str(e)}"
    
    @property
    def name(self) -> str:
        return "ollama_ocr"

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
        
        try:
            self.models['pymupdf'] = PyMuPDFOCRModel()
            if not self.default_model:
                self.default_model = 'pymupdf'
        except Exception:
            pass
    
    def add_openai_ocr(self, api_key: str):
        """Add OpenAI OCR model with API key"""
        try:
            self.models['openai_ocr'] = OpenAIOCRModel(api_key)
            print("✅ OpenAI OCR initialized")
        except Exception as e:
            print(f"❌ OpenAI OCR initialization failed: {e}")
    
    def add_ollama_ocr(self, model_name: str = "gemma3"):
        """Add Ollama OCR model"""
        try:
            self.models['ollama_ocr'] = OllamaOCRModel(model_name)
            print(f"✅ Ollama OCR initialized with model: {model_name}")
        except Exception as e:
            print(f"❌ Ollama OCR initialization failed: {e}")
    
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
        """Process PDF by converting pages to images and running OCR"""
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            return f"Error: Unknown model {model_name}", "error"
        
        try:
            # PyPDF2 and PyMuPDF handle PDFs directly
            if model_name in ['pypdf2', 'pymupdf']:
                if model_name == 'pypdf2':
                    text = self.models[model_name].process_pdf(pdf_path)
                else:
                    text = self.models[model_name].process_pdf(pdf_path)
                return text, "completed"
            
            # For other OCR models, convert PDF pages to images using PyMuPDF
            if not PYMUPDF_AVAILABLE:
                return "Error: PyMuPDF required for PDF to image conversion. Install with: pip install PyMuPDF", "error"
            
            import fitz
            doc = fitz.open(pdf_path)
            all_text = []
            
            for page_num in range(len(doc)):
                print(f"Processing page {page_num + 1}/{len(doc)}")
                
                # Convert page to image
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                
                # Convert to PIL Image
                from io import BytesIO
                img_data = pix.tobytes("png")
                image = Image.open(BytesIO(img_data))
                
                # Preprocess and run OCR with timeout for Ollama
                image = self.preprocess_image(image, model_name)
                
                if model_name == 'ollama_ocr':
                    # Apply timeout for Ollama OCR on PDF pages
                    import threading
                    import time
                    
                    result = [None]
                    error = [None]
                    
                    def ocr_with_timeout():
                        try:
                            result[0] = self.models[model_name].process_image(image)
                        except Exception as e:
                            error[0] = str(e)
                    
                    thread = threading.Thread(target=ocr_with_timeout)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=90)  # 90 second timeout
                    
                    if thread.is_alive():
                        page_text = f"Error: Ollama OCR timed out on page {page_num + 1}"
                    elif error[0]:
                        page_text = f"Error: {error[0]}"
                    else:
                        page_text = result[0] or "No response from Ollama"
                else:
                    page_text = self.models[model_name].process_image(image)
                
                all_text.append(f"[Page {page_num + 1}]\n{page_text}")
            
            doc.close()
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
            # Use PyMuPDF for PDF processing if available
            if model_name == 'pymupdf' and 'pymupdf' in self.models:
                return self.models['pymupdf'].process_pdf(file_path), "completed"
            else:
                return self.process_pdf(file_path, model_name)
        else:
            return f"Unsupported file type: {file_ext}", "error"