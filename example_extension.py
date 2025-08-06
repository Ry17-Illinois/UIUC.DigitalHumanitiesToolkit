#!/usr/bin/env python3
"""
Example of how to extend CODEBOOKS with additional OCR models
This demonstrates the extensible architecture
"""

from PIL import Image
from src.ocr_processor import BaseOCRModel, OCRProcessor

class ExampleOCRModel(BaseOCRModel):
    """
    Example OCR model implementation
    In practice, this would integrate with another OCR service/library
    """
    
    def process_image(self, image: Image.Image) -> str:
        # This is just a placeholder - replace with actual OCR implementation
        # For example, you might integrate with:
        # - Google Cloud Vision API
        # - Amazon Textract
        # - Azure Computer Vision
        # - EasyOCR
        # - PaddleOCR
        
        # Placeholder implementation
        return f"[Example OCR Model] Processed image of size {image.size}"
    
    @property
    def name(self) -> str:
        return "example_ocr"

def demonstrate_extension():
    """Demonstrate how to add a new OCR model"""
    
    # Create OCR processor
    processor = OCRProcessor()
    
    # Add the new model
    example_model = ExampleOCRModel()
    processor.add_model("example", example_model)
    
    # Show available models
    print("Available OCR models:", processor.get_available_models())
    
    # You could now use this model in the main application
    # by modifying the GUI to allow model selection

if __name__ == "__main__":
    demonstrate_extension()