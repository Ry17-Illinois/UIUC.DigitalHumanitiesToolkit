#!/usr/bin/env python3
"""
CODEBOOKS Tutorial - Understanding the System Components
This script demonstrates how each component works independently
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ledger_manager import LedgerManager
from ocr_processor import OCRProcessor
from prompt_processor import PromptProcessor

def tutorial_ledger_management():
    """Tutorial: Understanding the Ledger Manager"""
    print("=== LEDGER MANAGER TUTORIAL ===")
    
    # Create a ledger manager
    ledger = LedgerManager("tutorial_ledger.csv")
    
    # Show initial state
    print(f"Initial ledger has {len(ledger.df)} files")
    
    # Simulate adding files (you would use real file paths)
    example_files = [
        "example1.jpg",
        "example2.pdf", 
        "example3.png"
    ]
    
    print(f"Adding {len(example_files)} example files...")
    # Note: This will fail because files don't exist, but shows the concept
    
    # Show Dublin Core fields
    print("Dublin Core fields supported:")
    for field in ledger.DUBLIN_CORE_FIELDS:
        print(f"  - {field}")
    
    # Show summary
    summary = ledger.get_summary()
    print(f"Summary: {summary}")
    
    print()

def tutorial_ocr_processing():
    """Tutorial: Understanding OCR Processing"""
    print("=== OCR PROCESSOR TUTORIAL ===")
    
    # Create OCR processor
    ocr = OCRProcessor()
    
    # Show available models
    print("Available OCR models:")
    for model in ocr.get_available_models():
        print(f"  - {model}")
    
    # Show how to add new models (conceptually)
    print("To add new OCR models:")
    print("1. Create a class inheriting from BaseOCRModel")
    print("2. Implement process_image() method")
    print("3. Add to processor with add_model()")
    
    print()

def tutorial_prompt_processing():
    """Tutorial: Understanding Prompt Processing"""
    print("=== PROMPT PROCESSOR TUTORIAL ===")
    
    # Show available prompts
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    
    if os.path.exists(prompts_dir):
        print("Available Dublin Core prompts:")
        for filename in os.listdir(prompts_dir):
            if filename.endswith('.txt'):
                field_name = filename.replace('.txt', '')
                print(f"  - {field_name}")
                
                # Show prompt content
                with open(os.path.join(prompts_dir, filename), 'r') as f:
                    prompt_content = f.read().strip()
                    print(f"    Prompt: {prompt_content[:100]}...")
    
    print()
    print("To add new Dublin Core fields:")
    print("1. Create a new .txt file in prompts/ directory")
    print("2. Write a clear prompt for AI extraction")
    print("3. Field will automatically appear in the GUI")
    
    print()

def tutorial_workflow():
    """Tutorial: Understanding the Complete Workflow"""
    print("=== COMPLETE WORKFLOW TUTORIAL ===")
    
    print("CODEBOOKS Workflow:")
    print("1. ADD FILES: Import documents into the ledger")
    print("   - Individual files or entire directories")
    print("   - Supports JPG, PNG, TIFF, PDF formats")
    print("   - Each file gets unique ID and tracking status")
    
    print()
    print("2. RUN OCR: Extract text from documents")
    print("   - Uses Tesseract by default")
    print("   - Extensible to support multiple OCR models")
    print("   - Results stored directly in ledger")
    
    print()
    print("3. SETUP AI: Configure OpenAI API for metadata extraction")
    print("   - Requires OpenAI API key")
    print("   - Uses GPT models for intelligent extraction")
    
    print()
    print("4. GENERATE METADATA: Extract Dublin Core metadata")
    print("   - Select which Dublin Core field to extract")
    print("   - AI generates 1-3 examples per document")
    print("   - User approves/rejects each suggestion")
    print("   - Results added directly to ledger")
    
    print()
    print("5. MANAGE DATA: Clean up and maintain the ledger")
    print("   - Clear problematic entries")
    print("   - Re-run operations on specific files")
    print("   - Export/backup ledger data")
    
    print()

def main():
    """Run all tutorials"""
    print("CODEBOOKS SYSTEM TUTORIAL")
    print("=" * 50)
    print()
    
    tutorial_ledger_management()
    tutorial_ocr_processing()
    tutorial_prompt_processing()
    tutorial_workflow()
    
    print("=== NEXT STEPS ===")
    print("1. Run 'python main.py' to start the GUI application")
    print("2. Try adding some test files to see the system in action")
    print("3. Experiment with creating new prompt files")
    print("4. Consider extending with additional OCR models")
    print()
    print("For more information, see README.md")

if __name__ == "__main__":
    main()