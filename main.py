#!/usr/bin/env python3
"""
CODEBOOKS - Simplified Digital Humanities Metadata Pipeline
Main application with GUI interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.ledger_manager import LedgerManager
from src.ocr_processor import OCRProcessor
from src.prompt_processor import PromptProcessor
from src.ocr_evaluator import OCREvaluator

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class CodebooksApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CODEBOOKS - Digital Humanities Metadata Pipeline")
        self.root.geometry("1000x700")
        
        # Initialize components
        self.ledger = LedgerManager()
        self.ocr = OCRProcessor()
        self.prompt_processor = None
        
        self.setup_ui()
        self.refresh_display()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="CODEBOOKS", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Instructions
        inst_label = ttk.Label(control_frame, text="📋 WORKFLOW STEPS:", font=("Arial", 9, "bold"))
        inst_label.grid(row=0, column=0, pady=(0,5), sticky=tk.W)
        
        # File operations with step numbers and descriptions
        ttk.Button(control_frame, text="1️⃣ Add Files/Directory", 
                  command=self.add_files).grid(row=1, column=0, pady=2, sticky=tk.W)
        ttk.Label(control_frame, text="   Import documents to process", 
                 font=("Arial", 8), foreground="gray").grid(row=2, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="2️⃣ Run EasyOCR (🤖 AI)", 
                  command=self.run_ocr).grid(row=3, column=0, pady=2, sticky=tk.W)
        ttk.Label(control_frame, text="   GPU-powered text extraction", 
                 font=("Arial", 8), foreground="gray").grid(row=4, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="2️⃣ Run Tesseract OCR (📄)", 
                  command=self.run_tesseract_ocr).grid(row=5, column=0, pady=2, sticky=tk.W)
        ttk.Label(control_frame, text="   Traditional OCR engine", 
                 font=("Arial", 8), foreground="gray").grid(row=6, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="2️⃣ Run PyPDF2 Extract (📋)", 
                  command=self.run_pypdf2_ocr).grid(row=7, column=0, pady=2, sticky=tk.W)
        ttk.Label(control_frame, text="   Direct PDF text extraction", 
                 font=("Arial", 8), foreground="gray").grid(row=8, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="2️⃣ Run OpenAI OCR (🤖📄)", 
                  command=self.run_openai_ocr).grid(row=9, column=0, pady=2, sticky=tk.W)
        ttk.Label(control_frame, text="   AI-powered image transcription", 
                 font=("Arial", 8), foreground="gray").grid(row=10, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="3️⃣ Setup AI Prompts (🤖)", 
                  command=self.setup_ai).grid(row=11, column=0, pady=2, sticky=tk.W)
        ttk.Label(control_frame, text="   Configure OpenAI API key", 
                 font=("Arial", 8), foreground="gray").grid(row=12, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="4️⃣ Generate Metadata (📝)", 
                  command=self.generate_metadata).grid(row=13, column=0, pady=2, sticky=tk.W)
        ttk.Label(control_frame, text="   AI-powered Dublin Core extraction", 
                 font=("Arial", 8), foreground="gray").grid(row=14, column=0, sticky=tk.W)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=15, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(control_frame, text="🛠️ UTILITIES:", font=("Arial", 9, "bold")).grid(row=16, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="🔍 Evaluate OCR Quality", 
                  command=self.evaluate_ocr).grid(row=17, column=0, pady=2, sticky=tk.W)
        ttk.Label(control_frame, text="   Compare OCR engine performance", 
                 font=("Arial", 8), foreground="gray").grid(row=18, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="🗑️ Clear Selected Rows", 
                  command=self.clear_rows).grid(row=19, column=0, pady=2, sticky=tk.W)
        
        ttk.Button(control_frame, text="🔄 Refresh Display", 
                  command=self.refresh_display).grid(row=20, column=0, pady=2, sticky=tk.W)
        
        # Status display
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.grid(row=21, column=0, pady=(10, 0), sticky=tk.W)
        
        # Data display frame
        data_frame = ttk.LabelFrame(main_frame, text="Metadata Ledger", padding="10")
        data_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)
        
        # Treeview for data display
        columns = ['filename', 'easyocr_status', 'tesseract_status', 'pypdf2_status', 'openai_ocr_status', 'easyocr_text', 'tesseract_text', 'pypdf2_text', 'openai_ocr_text', 'title', 'creator', 'subject']
        self.tree = ttk.Treeview(data_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.tree.heading(col, text=col.replace('_', ' ').title())
            if col in ['easyocr_text', 'tesseract_text', 'pypdf2_text', 'openai_ocr_text']:
                self.tree.column(col, width=100)
            else:
                self.tree.column(col, width=80)
        
        # Bind double-click to view full text
        self.tree.bind('<Double-1>', self.on_tree_double_click)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(data_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Summary frame
        summary_frame = ttk.LabelFrame(main_frame, text="Summary", padding="10")
        summary_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.summary_text = tk.Text(summary_frame, height=10, width=30)
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        summary_frame.rowconfigure(0, weight=1)
        summary_frame.columnconfigure(0, weight=1)
    
    def add_files(self):
        """Add files or directory to the ledger"""
        choice = messagebox.askyesnocancel("Add Files", 
                                          "Yes: Add individual files\nNo: Add entire directory\nCancel: Cancel")
        
        if choice is None:  # Cancel
            return
        elif choice:  # Yes - individual files
            files = filedialog.askopenfilenames(
                title="Select files",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff"), 
                          ("PDFs", "*.pdf"), ("All files", "*.*")]
            )
            file_paths = list(files)
        else:  # No - directory
            directory = filedialog.askdirectory(title="Select directory")
            if not directory:
                return
            
            file_paths = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if Path(file).suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pdf']:
                        file_paths.append(os.path.join(root, file))
        
        if file_paths:
            added_count = self.ledger.add_files(file_paths)
            messagebox.showinfo("Files Added", f"Added {added_count} new files to the ledger")
            self.refresh_display()
    
    def run_ocr(self):
        """Run OCR on pending files"""
        # Check if OCR models are available
        if not self.ocr.models:
            messagebox.showerror("OCR Not Available", 
                               "EasyOCR not available. Install with: pip install easyocr")
            return
        
        pending_files = self.ledger.get_files_by_status('easyocr', 'pending')
        
        if pending_files.empty:
            messagebox.showinfo("OCR", "No files pending OCR processing")
            return
        
        def ocr_worker():
            self.status_var.set("Running OCR...")
            processed = 0
            
            for _, row in pending_files.iterrows():
                self.status_var.set(f"OCR: {processed + 1}/{len(pending_files)} - {row['filename']}")
                
                text, status = self.ocr.process_file(row['filepath'], 'easyocr')
                self.ledger.update_ocr_result(row['file_id'], text, status, 'easyocr')
                processed += 1
            
            self.status_var.set("OCR completed")
            self.root.after(0, self.refresh_display)
        
        threading.Thread(target=ocr_worker, daemon=True).start()
    
    def run_tesseract_ocr(self):
        """Run Tesseract OCR on pending files"""
        # Check if Tesseract model is available
        if 'tesseract' not in self.ocr.models:
            # Try to get more specific error info
            try:
                import pytesseract
                from PIL import Image
                # Test with a small image
                test_img = Image.new('RGB', (100, 50), color='white')
                pytesseract.image_to_string(test_img)
                error_msg = "Tesseract model failed to initialize for unknown reasons."
            except Exception as e:
                error_msg = f"Tesseract Error: {str(e)}\n\nPlease install Tesseract OCR from:\nhttps://github.com/UB-Mannheim/tesseract/wiki"
            
            messagebox.showerror("Tesseract Not Available", error_msg)
            return
        
        pending_files = self.ledger.get_files_by_status('tesseract', 'pending')
        
        if pending_files.empty:
            messagebox.showinfo("Tesseract OCR", "No files pending Tesseract OCR processing")
            return
        
        def tesseract_worker():
            self.status_var.set("Running Tesseract OCR...")
            processed = 0
            
            for _, row in pending_files.iterrows():
                self.status_var.set(f"Tesseract OCR: {processed + 1}/{len(pending_files)} - {row['filename']}")
                
                try:
                    text, status = self.ocr.process_file(row['filepath'], 'tesseract')
                    self.ledger.update_ocr_result(row['file_id'], text, status, 'tesseract')
                except Exception as e:
                    error_text = f"Error processing {row['filename']}: {str(e)}"
                    self.ledger.update_ocr_result(row['file_id'], error_text, 'error', 'tesseract')
                    print(f"Tesseract error: {e}")
                processed += 1
            
            self.status_var.set("Tesseract OCR completed")
            self.root.after(0, self.refresh_display)
        
        threading.Thread(target=tesseract_worker, daemon=True).start()
    
    def run_pypdf2_ocr(self):
        """Run PyPDF2 text extraction on pending PDF files"""
        # Check if PyPDF2 model is available
        if 'pypdf2' not in self.ocr.models:
            messagebox.showerror("PyPDF2 Not Available", 
                               "PyPDF2 not available. Install with: pip install PyPDF2")
            return
        
        pending_files = self.ledger.get_files_by_status('pypdf2', 'pending')
        
        if pending_files.empty:
            messagebox.showinfo("PyPDF2 Extract", "No files pending PyPDF2 processing")
            return
        
        def pypdf2_worker():
            self.status_var.set("Running PyPDF2 extraction...")
            processed = 0
            
            for _, row in pending_files.iterrows():
                self.status_var.set(f"PyPDF2: {processed + 1}/{len(pending_files)} - {row['filename']}")
                
                try:
                    text, status = self.ocr.process_file(row['filepath'], 'pypdf2')
                    self.ledger.update_ocr_result(row['file_id'], text, status, 'pypdf2')
                except Exception as e:
                    error_text = f"Error processing {row['filename']}: {str(e)}"
                    self.ledger.update_ocr_result(row['file_id'], error_text, 'error', 'pypdf2')
                    print(f"PyPDF2 error: {e}")
                processed += 1
            
            self.status_var.set("PyPDF2 extraction completed")
            self.root.after(0, self.refresh_display)
        
        threading.Thread(target=pypdf2_worker, daemon=True).start()
    
    def run_openai_ocr(self):
        """Run OpenAI OCR on pending files"""
        # Check if we have OpenAI API key and add the model
        if not self.prompt_processor:
            messagebox.showwarning("OpenAI Not Setup", "Please setup AI first to use OpenAI OCR")
            return
        
        # Add OpenAI OCR model using the same API key
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.ocr.add_openai_ocr(api_key)
        
        if 'openai_ocr' not in self.ocr.models:
            messagebox.showerror("OpenAI OCR Not Available", 
                               "OpenAI OCR not available. Setup AI first.")
            return
        
        pending_files = self.ledger.get_files_by_status('openai_ocr', 'pending')
        
        if pending_files.empty:
            messagebox.showinfo("OpenAI OCR", "No files pending OpenAI OCR processing")
            return
        
        def openai_worker():
            self.status_var.set("Running OpenAI OCR...")
            processed = 0
            
            for _, row in pending_files.iterrows():
                self.status_var.set(f"OpenAI OCR: {processed + 1}/{len(pending_files)} - {row['filename']}")
                
                try:
                    text, status = self.ocr.process_file(row['filepath'], 'openai_ocr')
                    self.ledger.update_ocr_result(row['file_id'], text, status, 'openai_ocr')
                except Exception as e:
                    error_text = f"Error processing {row['filename']}: {str(e)}"
                    self.ledger.update_ocr_result(row['file_id'], error_text, 'error', 'openai_ocr')
                    print(f"OpenAI OCR error: {e}")
                processed += 1
            
            self.status_var.set("OpenAI OCR completed")
            self.root.after(0, self.refresh_display)
        
        threading.Thread(target=openai_worker, daemon=True).start()
    
    def setup_ai(self):
        """Setup AI API key for prompt processing"""
        api_key = simpledialog.askstring("OpenAI API Key", 
                                        "Enter your OpenAI API key:", show='*')
        if api_key:
            try:
                self.prompt_processor = PromptProcessor(api_key)
                messagebox.showinfo("AI Setup", "AI processor initialized successfully")
            except Exception as e:
                messagebox.showerror("AI Setup Error", f"Failed to initialize AI: {str(e)}")
    
    def generate_metadata(self):
        """Generate metadata using AI prompts"""
        if not self.prompt_processor:
            messagebox.showwarning("AI Not Setup", "Please setup AI first")
            return
        
        # Show metadata generation setup dialog
        config = self.show_metadata_config_dialog()
        if not config:
            return
        
        # Process files with selected configuration
        def metadata_worker():
            self.status_var.set("Generating metadata...")
            processed = 0
            
            for _, row in config['files'].iterrows():
                # Skip if already processed
                if row[f"{config['field']}_status"] == 'completed':
                    continue
                
                self.status_var.set(f"Metadata: {processed + 1}/{len(config['files'])} - {row['filename']}")
                
                # Get OCR text from selected source
                ocr_text = row.get(f"{config['ocr_source']}_ocr", '')
                if not ocr_text or str(ocr_text).strip() == '' or str(ocr_text) == 'nan':
                    self.status_var.set(f"Skipping {row['filename']} - no {config['ocr_source']} text")
                    continue
                
                examples = self.prompt_processor.generate_examples(str(ocr_text), config['field'])
                
                # Show examples to user for approval
                approved_value = self.show_examples_dialog(row['filename'], config['field'], examples, config['ocr_source'])
                
                if approved_value:
                    self.ledger.update_dublin_core_field(row['file_id'], config['field'], approved_value)
                
                processed += 1
            
            self.status_var.set("Metadata generation completed")
            self.root.after(0, self.refresh_display)
        
        threading.Thread(target=metadata_worker, daemon=True).start()
    
    def select_dublin_core_field(self, available_fields: List[str]) -> str:
        """Dialog to select Dublin Core field"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Dublin Core Field")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        selected_field = tk.StringVar()
        
        ttk.Label(dialog, text="Select a Dublin Core field:").pack(pady=10)
        
        for field in available_fields:
            ttk.Radiobutton(dialog, text=field.title(), variable=selected_field, 
                           value=field).pack(anchor=tk.W, padx=20)
        
        def on_ok():
            dialog.destroy()
        
        ttk.Button(dialog, text="OK", command=on_ok).pack(pady=10)
        
        dialog.wait_window()
        return selected_field.get()
    
    def show_metadata_config_dialog(self) -> dict:
        """Dialog to configure metadata generation settings"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure Metadata Generation")
        dialog.geometry("900x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {}
        
        # Main container with left and right panels
        main_container = ttk.Frame(dialog)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel for configuration
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right panel for OCR preview
        right_panel = ttk.LabelFrame(main_container, text="📄 OCR Text Preview")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = tk.Text(left_panel, height=6, wrap=tk.WORD, bg='#f0f0f0')
        instructions.pack(fill=tk.X, pady=5)
        instructions.insert(1.0, 
            "📋 METADATA GENERATION SETUP\n\n"
            "1️⃣ Choose OCR Source: Select which OCR text to use for AI analysis\n"
            "2️⃣ Choose Dublin Core Field: Select metadata field to generate\n"
            "3️⃣ AI will analyze OCR text and suggest metadata values\n"
            "4️⃣ You'll review and approve each suggestion")
        instructions.config(state=tk.DISABLED)
        
        # OCR Source Selection
        ttk.Label(left_panel, text="1️⃣ Select OCR Source:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10,5))
        
        ocr_var = tk.StringVar(value="easyocr")
        ocr_frame = ttk.Frame(left_panel)
        ocr_frame.pack(fill=tk.X, padx=10)
        
        ttk.Radiobutton(ocr_frame, text="🤖 EasyOCR (AI-powered, best quality)", 
                       variable=ocr_var, value="easyocr").pack(anchor=tk.W)
        ttk.Radiobutton(ocr_frame, text="📄 Tesseract (Traditional OCR)", 
                       variable=ocr_var, value="tesseract").pack(anchor=tk.W)
        ttk.Radiobutton(ocr_frame, text="📋 PyPDF2 (Direct PDF text)", 
                       variable=ocr_var, value="pypdf2").pack(anchor=tk.W)
        ttk.Radiobutton(ocr_frame, text="🤖📄 OpenAI OCR (AI transcription)", 
                       variable=ocr_var, value="openai_ocr").pack(anchor=tk.W)
        
        # Dublin Core Field Selection
        ttk.Label(left_panel, text="2️⃣ Select Dublin Core Field:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(15,5))
        
        available_fields = self.prompt_processor.get_available_fields() if self.prompt_processor else []
        if not available_fields:
            ttk.Label(left_panel, text="❌ No prompt files found in prompts directory", 
                     foreground="red").pack(anchor=tk.W, padx=10)
            return None
        
        field_var = tk.StringVar()
        field_frame = ttk.Frame(left_panel)
        field_frame.pack(fill=tk.X, padx=10)
        
        for field in available_fields:
            ttk.Radiobutton(field_frame, text=f"📝 {field.title()}", 
                           variable=field_var, value=field).pack(anchor=tk.W)
        
        def on_ok():
            ocr_source = ocr_var.get()
            field = field_var.get()
            
            if not field:
                messagebox.showwarning("Selection Required", "Please select a Dublin Core field")
                return
            
            # Get files with completed OCR from selected source
            files = self.ledger.get_files_by_status(ocr_source, 'completed')
            
            if files.empty:
                messagebox.showinfo("No Files", f"No files with completed {ocr_source.upper()} found")
                return
            
            result['ocr_source'] = ocr_source
            result['field'] = field
            result['files'] = files
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # OCR Preview Text Widget
        preview_text = tk.Text(right_panel, wrap=tk.WORD, height=15, width=40)
        preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        preview_text.insert(1.0, "Select an OCR source to preview text...")
        preview_text.config(state=tk.DISABLED)
        
        def update_preview():
            """Update OCR text preview when source changes"""
            ocr_source = ocr_var.get()
            files = self.ledger.get_files_by_status(ocr_source, 'completed')
            
            preview_text.config(state=tk.NORMAL)
            preview_text.delete(1.0, tk.END)
            
            if files.empty:
                preview_text.insert(1.0, f"❌ No files with completed {ocr_source.upper()} found.\n\nRun {ocr_source.upper()} first to see preview.")
            else:
                preview_text.insert(1.0, f"📊 Found {len(files)} files with {ocr_source.upper()} text\n\n")
                preview_text.insert(tk.END, "📄 Sample from first file:\n")
                preview_text.insert(tk.END, "-" * 40 + "\n")
                
                # Show sample from first file
                sample_text = str(files.iloc[0].get(f'{ocr_source}_ocr', '') or '')
                if sample_text and sample_text != 'nan':
                    preview_sample = sample_text[:500] + "..." if len(sample_text) > 500 else sample_text
                    preview_text.insert(tk.END, preview_sample)
                else:
                    preview_text.insert(tk.END, "No text available")
            
            preview_text.config(state=tk.DISABLED)
        
        # Bind OCR selection change to preview update
        for child in ocr_frame.winfo_children():
            if isinstance(child, ttk.Radiobutton):
                child.configure(command=update_preview)
        
        # Update preview initially
        dialog.after(100, update_preview)
        
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="▶️ Start Generation", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()
        return result
    
    def show_examples_dialog(self, filename: str, field: str, examples: List[str], ocr_source: str = "easyocr") -> str:
        """Show examples dialog for user approval"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Approve {field.title()} for {filename}")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = tk.StringVar()
        
        # Header with instructions
        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(header_frame, text=f"📝 {field.title()} Suggestions for: {filename}", 
                 font=("Arial", 12, "bold")).pack()
        ttk.Label(header_frame, text=f"📊 Source: {ocr_source.upper()} OCR", 
                 font=("Arial", 9), foreground="gray").pack()
        
        # Instructions
        inst_text = tk.Text(dialog, height=3, wrap=tk.WORD, bg='#f8f8f8')
        inst_text.pack(fill=tk.X, padx=10, pady=5)
        inst_text.insert(1.0, 
            "🤖 AI analyzed the OCR text and generated these suggestions.\n"
            "✅ Select a suggestion OR enter your own value below.\n"
            "❌ Click Reject to skip this file.")
        inst_text.config(state=tk.DISABLED)
        
        # AI Suggestions section
        ttk.Label(dialog, text="🤖 AI Suggestions:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10,5))
        
        suggestions_frame = ttk.Frame(dialog)
        suggestions_frame.pack(fill=tk.X, padx=20)
        
        for i, example in enumerate(examples, 1):
            ttk.Radiobutton(suggestions_frame, text=f"{i}. {example}", variable=result, 
                           value=example).pack(anchor=tk.W, pady=2)
        
        # Custom entry section
        ttk.Label(dialog, text="✏️ Or enter custom value:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(15, 5))
        custom_entry = ttk.Entry(dialog, width=60)
        custom_entry.pack(padx=20, pady=5)
        
        def on_approve():
            if custom_entry.get().strip():
                result.set(custom_entry.get().strip())
            dialog.destroy()
        
        def on_reject():
            result.set("")
            dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="✅ Approve", command=on_approve).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Reject", command=on_reject).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()
        return result.get()
    
    def clear_rows(self):
        """Clear selected rows from the ledger"""
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select rows to clear")
            return
        
        # Confirmation dialog
        confirmation_text = simpledialog.askstring(
            "Confirm Deletion", 
            f"Type 'DELETE {len(selected_items)} ROWS' to confirm deletion of selected rows:"
        )
        
        expected_text = f"DELETE {len(selected_items)} ROWS"
        if confirmation_text != expected_text:
            messagebox.showinfo("Cancelled", "Deletion cancelled - text did not match")
            return
        
        # Get file IDs from selected items
        file_ids = []
        for item in selected_items:
            values = self.tree.item(item)['values']
            filename = values[0]
            matching_rows = self.ledger.df[self.ledger.df['filename'] == filename]
            if not matching_rows.empty:
                file_ids.extend(matching_rows['file_id'].tolist())
        
        self.ledger.clear_rows(file_ids)
        messagebox.showinfo("Deleted", f"Deleted {len(file_ids)} rows")
        self.refresh_display()
    
    def refresh_display(self):
        """Refresh the data display"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add current data
        for _, row in self.ledger.df.iterrows():
            # Truncate OCR text for display - handle NaN values
            easyocr_text = str(row.get('easyocr_ocr', '') or '')
            tesseract_text = str(row.get('tesseract_ocr', '') or '')
            pypdf2_text = str(row.get('pypdf2_ocr', '') or '')
            openai_text = str(row.get('openai_ocr_ocr', '') or '')
            easyocr_display = easyocr_text[:20] + '...' if len(easyocr_text) > 20 else easyocr_text
            tesseract_display = tesseract_text[:20] + '...' if len(tesseract_text) > 20 else tesseract_text
            pypdf2_display = pypdf2_text[:20] + '...' if len(pypdf2_text) > 20 else pypdf2_text
            openai_display = openai_text[:20] + '...' if len(openai_text) > 20 else openai_text
            
            values = [
                row['filename'],
                row.get('easyocr_status', 'pending'),
                row.get('tesseract_status', 'pending'),
                row.get('pypdf2_status', 'pending'),
                row.get('openai_ocr_status', 'pending'),
                easyocr_display,
                tesseract_display,
                pypdf2_display,
                openai_display,
                row.get('title', ''),
                row.get('creator', ''),
                row.get('subject', '')
            ]
            self.tree.insert('', 'end', values=values)
        
        # Update summary with workflow diagram
        summary = self.ledger.get_summary()
        summary_text = f"""📋 CODEBOOKS WORKFLOW:

📁 Files → 📄 OCR → 🤖 AI → 📝 Metadata

📊 PROCESSING STATUS:

Total Files: {summary['total_files']}

🤖 EasyOCR (AI-powered):
  ✅ {summary['easyocr_completed']} | ⏳ {summary['easyocr_pending']} | ❌ {summary['easyocr_error']}

📄 Tesseract (Traditional):
  ✅ {summary['tesseract_completed']} | ⏳ {summary['tesseract_pending']} | ❌ {summary['tesseract_error']}

📋 PyPDF2 (Direct):
  ✅ {summary['pypdf2_completed']} | ⏳ {summary['pypdf2_pending']} | ❌ {summary['pypdf2_error']}

🤖📄 OpenAI OCR:
  ✅ {summary['openai_ocr_completed']} | ⏳ {summary['openai_ocr_pending']} | ❌ {summary['openai_ocr_error']}

📝 DUBLIN CORE METADATA:"""
        
        for field, stats in summary['dublin_core_fields'].items():
            if stats['completed'] > 0 or stats['pending'] > 0 or stats['error'] > 0:
                summary_text += f"\n  {field.title()}: {stats['completed']}/{stats['pending']}/{stats['error']}"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary_text)
    
    def on_tree_double_click(self, event):
        """Handle double-click on tree item to show full OCR text"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if not item:
            return
        
        values = self.tree.item(item)['values']
        filename = values[0]
        
        # Find the full OCR text
        matching_rows = self.ledger.df[self.ledger.df['filename'] == filename]
        if matching_rows.empty:
            return
        
        row = matching_rows.iloc[0]
        easyocr_text = str(row.get('easyocr_ocr', '') or '')
        tesseract_text = str(row.get('tesseract_ocr', '') or '')
        pypdf2_text = str(row.get('pypdf2_ocr', '') or '')
        openai_text = str(row.get('openai_ocr_ocr', '') or '')
        
        # Show full text in dialog with tabs
        dialog = tk.Toplevel(self.root)
        dialog.title(f"OCR Text - {filename}")
        dialog.geometry("700x500")
        dialog.transient(self.root)
        
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # EasyOCR tab
        if easyocr_text:
            easy_frame = ttk.Frame(notebook)
            notebook.add(easy_frame, text="EasyOCR")
            easy_text = tk.Text(easy_frame, wrap=tk.WORD)
            easy_text.pack(fill=tk.BOTH, expand=True)
            easy_text.insert(1.0, easyocr_text)
            easy_text.config(state=tk.DISABLED)
        
        # Tesseract tab
        if tesseract_text:
            tess_frame = ttk.Frame(notebook)
            notebook.add(tess_frame, text="Tesseract")
            tess_text = tk.Text(tess_frame, wrap=tk.WORD)
            tess_text.pack(fill=tk.BOTH, expand=True)
            tess_text.insert(1.0, tesseract_text)
            tess_text.config(state=tk.DISABLED)
        
        # PyPDF2 tab
        if pypdf2_text:
            pdf_frame = ttk.Frame(notebook)
            notebook.add(pdf_frame, text="PyPDF2")
            pdf_text = tk.Text(pdf_frame, wrap=tk.WORD)
            pdf_text.pack(fill=tk.BOTH, expand=True)
            pdf_text.insert(1.0, pypdf2_text)
            pdf_text.config(state=tk.DISABLED)
        
        # OpenAI OCR tab
        if openai_text:
            ai_frame = ttk.Frame(notebook)
            notebook.add(ai_frame, text="OpenAI OCR")
            ai_text = tk.Text(ai_frame, wrap=tk.WORD)
            ai_text.pack(fill=tk.BOTH, expand=True)
            ai_text.insert(1.0, openai_text)
            ai_text.config(state=tk.DISABLED)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
    
    def evaluate_ocr(self):
        """Evaluate OCR quality for selected files"""
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select files to evaluate")
            return
        
        # Get selected files data
        files_data = []
        for item in selected_items:
            values = self.tree.item(item)['values']
            filename = values[0]
            matching_rows = self.ledger.df[self.ledger.df['filename'] == filename]
            if not matching_rows.empty:
                files_data.append(matching_rows.iloc[0])
        
        if not files_data:
            messagebox.showwarning("No Data", "No OCR data found for selected files")
            return
        
        self.show_ocr_evaluation_dialog(files_data)
    
    def show_ocr_evaluation_dialog(self, files_data: List):
        """Show OCR evaluation results dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("OCR Quality Evaluation")
        dialog.geometry("900x600")
        dialog.transient(self.root)
        
        # Main container
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(header_frame, text="🔍 OCR Quality Evaluation", 
                 font=("Arial", 14, "bold")).pack()
        ttk.Label(header_frame, text=f"Evaluating {len(files_data)} files", 
                 font=("Arial", 10), foreground="gray").pack()
        
        # Notebook for different views
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Summary tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📊 Summary")
        
        summary_text = tk.Text(summary_frame, wrap=tk.WORD, font=("Consolas", 9))
        summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Calculate overall metrics
        all_results = []
        for file_data in files_data:
            ocr_results = {
                'easyocr': file_data.get('easyocr_ocr', ''),
                'tesseract': file_data.get('tesseract_ocr', ''),
                'pypdf2': file_data.get('pypdf2_ocr', ''),
                'openai_ocr': file_data.get('openai_ocr_ocr', '')
            }
            evaluation = OCREvaluator.evaluate_ocr_engines(ocr_results)
            all_results.append((file_data['filename'], evaluation))
        
        # Generate summary report
        summary_report = self.generate_evaluation_summary(all_results)
        summary_text.insert(1.0, summary_report)
        summary_text.config(state=tk.DISABLED)
        
        # Detailed results tab
        details_frame = ttk.Frame(notebook)
        notebook.add(details_frame, text="📋 Detailed Results")
        
        # Treeview for detailed results
        detail_columns = ['filename', 'engine', 'quality_score', 'text_length', 'similarity']
        detail_tree = ttk.Treeview(details_frame, columns=detail_columns, show='headings')
        
        for col in detail_columns:
            detail_tree.heading(col, text=col.replace('_', ' ').title())
            detail_tree.column(col, width=120)
        
        # Populate detailed results
        for filename, evaluation in all_results:
            for engine, metrics in evaluation.items():
                similarity = metrics.get('avg_similarity_to_others', 0.0)
                detail_tree.insert('', 'end', values=[
                    filename[:20] + '...' if len(filename) > 20 else filename,
                    engine,
                    f"{metrics['quality_score']:.3f}",
                    metrics['text_length'],
                    f"{similarity:.3f}" if similarity else "N/A"
                ])
        
        detail_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Ground truth comparison tab
        gt_frame = ttk.Frame(notebook)
        notebook.add(gt_frame, text="✅ Ground Truth")
        
        # Instructions
        inst_label = ttk.Label(gt_frame, text="Compare OCR engines using ground truth or OCR-to-OCR comparison:", 
                              font=("Arial", 10, "bold"))
        inst_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # File selector for ground truth
        gt_file_frame = ttk.Frame(gt_frame)
        gt_file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(gt_file_frame, text="File:").pack(side=tk.LEFT)
        gt_file_var = tk.StringVar(value=files_data[0]['filename'] if files_data else "")
        gt_file_combo = ttk.Combobox(gt_file_frame, textvariable=gt_file_var, 
                                    values=[f['filename'] for f in files_data])
        gt_file_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Ground truth source selection
        gt_source_frame = ttk.LabelFrame(gt_frame, text="Ground Truth Source")
        gt_source_frame.pack(fill=tk.X, padx=5, pady=5)
        
        gt_source_var = tk.StringVar(value="manual")
        ttk.Radiobutton(gt_source_frame, text="📝 Manual Entry", variable=gt_source_var, 
                       value="manual").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(gt_source_frame, text="🤖 EasyOCR as Reference", variable=gt_source_var, 
                       value="easyocr").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(gt_source_frame, text="📄 Tesseract as Reference", variable=gt_source_var, 
                       value="tesseract").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(gt_source_frame, text="📋 PyPDF2 as Reference", variable=gt_source_var, 
                       value="pypdf2").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(gt_source_frame, text="🤖📄 OpenAI OCR as Reference", variable=gt_source_var, 
                       value="openai_ocr").pack(anchor=tk.W, padx=5)
        
        # Ground truth text input
        gt_text_label = ttk.Label(gt_frame, text="Manual Ground Truth Text:")
        gt_text_label.pack(anchor=tk.W, padx=5, pady=(10,0))
        gt_text = tk.Text(gt_frame, height=6, wrap=tk.WORD)
        gt_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Ground truth results
        gt_results_text = tk.Text(gt_frame, height=12, wrap=tk.WORD, font=("Consolas", 9))
        gt_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        def update_gt_text():
            """Update ground truth text when source changes"""
            source = gt_source_var.get()
            filename = gt_file_var.get()
            
            if source == "manual":
                gt_text.config(state=tk.NORMAL)
                return
            
            # Find selected file data
            selected_file = None
            for file_data in files_data:
                if file_data['filename'] == filename:
                    selected_file = file_data
                    break
            
            if selected_file is not None:
                ocr_text = str(selected_file.get(f'{source}_ocr', '') or '')
                if source == 'openai_ocr':
                    ocr_text = str(selected_file.get('openai_ocr_ocr', '') or '')
                
                gt_text.config(state=tk.NORMAL)
                gt_text.delete(1.0, tk.END)
                if ocr_text and ocr_text != 'nan':
                    gt_text.insert(1.0, ocr_text)
                gt_text.config(state=tk.DISABLED)
        
        # Bind events to update ground truth text
        for child in gt_source_frame.winfo_children():
            if isinstance(child, ttk.Radiobutton):
                child.configure(command=update_gt_text)
        gt_file_combo.bind('<<ComboboxSelected>>', lambda e: update_gt_text())
        
        def evaluate_with_ground_truth():
            selected_filename = gt_file_var.get()
            gt_source = gt_source_var.get()
            
            # Get ground truth text
            if gt_source == "manual":
                ground_truth = gt_text.get(1.0, tk.END).strip()
                if not ground_truth:
                    messagebox.showwarning("No Ground Truth", "Please enter ground truth text")
                    return
                reference_label = "Manual Entry"
            else:
                # Find selected file data
                selected_file = None
                for file_data in files_data:
                    if file_data['filename'] == selected_filename:
                        selected_file = file_data
                        break
                
                if selected_file is None:
                    return
                
                if gt_source == 'openai_ocr':
                    ground_truth = str(selected_file.get('openai_ocr_ocr', '') or '')
                else:
                    ground_truth = str(selected_file.get(f'{gt_source}_ocr', '') or '')
                
                if not ground_truth or ground_truth == 'nan':
                    messagebox.showwarning("No Reference Text", f"No {gt_source.upper()} text found for this file")
                    return
                
                reference_label = f"{gt_source.upper()} OCR"
            
            # Find selected file data
            selected_file = None
            for file_data in files_data:
                if file_data['filename'] == selected_filename:
                    selected_file = file_data
                    break
            
            if selected_file is None:
                return
            
            ocr_results = {
                'easyocr': selected_file.get('easyocr_ocr', ''),
                'tesseract': selected_file.get('tesseract_ocr', ''),
                'pypdf2': selected_file.get('pypdf2_ocr', ''),
                'openai_ocr': selected_file.get('openai_ocr_ocr', '')
            }
            
            # Remove the reference engine from comparison if using OCR as ground truth
            if gt_source != "manual":
                comparison_results = {k: v for k, v in ocr_results.items() if k != gt_source}
            else:
                comparison_results = ocr_results
            
            evaluation = OCREvaluator.evaluate_ocr_engines(comparison_results, ground_truth)
            
            # Display results
            gt_results_text.delete(1.0, tk.END)
            report = f"📊 COMPARATIVE EVALUATION: {selected_filename}\n"
            report += f"📋 Reference: {reference_label}\n"
            report += "=" * 60 + "\n\n"
            
            # Sort by similarity to ground truth
            sorted_engines = sorted(evaluation.items(), 
                                  key=lambda x: x[1].get('similarity_to_ground_truth', 0), reverse=True)
            
            for engine, metrics in sorted_engines:
                if metrics['text_length'] > 0:
                    report += f"🔧 {engine.upper()}:\n"
                    report += f"   Character Error Rate: {metrics['cer']:.3f}\n"
                    report += f"   Word Error Rate: {metrics['wer']:.3f}\n"
                    report += f"   Similarity to Reference: {metrics['similarity_to_ground_truth']:.3f}\n"
                    report += f"   Quality Score: {metrics['quality_score']:.3f}\n"
                    report += f"   Text Length: {metrics['text_length']} chars\n\n"
            
            gt_results_text.insert(1.0, report)
        
        ttk.Button(gt_frame, text="🔍 Run Comparison", 
                  command=evaluate_with_ground_truth).pack(pady=5)
        
        # Visualization tab
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="📈 Visualizations")
        
        # Debug label to show matplotlib status
        ttk.Label(viz_frame, text=f"Matplotlib Available: {MATPLOTLIB_AVAILABLE}", 
                 font=("Arial", 10)).pack(pady=5)
        
        if MATPLOTLIB_AVAILABLE:
            # Visualization controls
            viz_controls = ttk.Frame(viz_frame)
            viz_controls.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(viz_controls, text="📊 Quality Comparison", 
                      command=lambda: self.plot_quality_comparison(viz_frame, all_results)).pack(side=tk.LEFT, padx=5)
            ttk.Button(viz_controls, text="🔗 Similarity Matrix", 
                      command=lambda: self.plot_similarity_matrix(viz_frame, all_results)).pack(side=tk.LEFT, padx=5)
            ttk.Button(viz_controls, text="📏 Length vs Quality", 
                      command=lambda: self.plot_length_vs_quality(viz_frame, all_results)).pack(side=tk.LEFT, padx=5)
            
            ttk.Label(viz_frame, text="Click buttons above to generate visualizations", 
                     font=("Arial", 10), foreground="gray").pack(pady=10)
        else:
            ttk.Label(viz_frame, text="📊 Matplotlib not available\n\nInstall with: pip install matplotlib", 
                     font=("Arial", 12), justify=tk.CENTER).pack(expand=True)
        
        # Initialize ground truth text on dialog open
        dialog.after(100, update_gt_text)
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
    
    def generate_evaluation_summary(self, all_results: List[Tuple]) -> str:
        """Generate evaluation summary report"""
        report = "📊 OCR QUALITY EVALUATION SUMMARY\n"
        report += "=" * 50 + "\n\n"
        
        # Engine performance summary
        engine_stats = {'easyocr': [], 'tesseract': [], 'pypdf2': [], 'openai_ocr': []}
        
        for filename, evaluation in all_results:
            for engine, metrics in evaluation.items():
                if metrics['text_length'] > 0:
                    engine_stats[engine].append(metrics['quality_score'])
        
        report += "🏆 ENGINE PERFORMANCE (Average Quality Score):\n"
        for engine, scores in engine_stats.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                report += f"   {engine.upper()}: {avg_score:.3f} ({len(scores)} files)\n"
            else:
                report += f"   {engine.upper()}: No data\n"
        
        report += "\n" + "=" * 50 + "\n\n"
        
        # Per-file summary
        report += "📋 PER-FILE RESULTS:\n\n"
        for filename, evaluation in all_results:
            report += f"📄 {filename}:\n"
            
            # Sort engines by quality score
            sorted_engines = sorted(evaluation.items(), 
                                  key=lambda x: x[1]['quality_score'], reverse=True)
            
            for engine, metrics in sorted_engines:
                if metrics['text_length'] > 0:
                    similarity = metrics.get('avg_similarity_to_others', 0)
                    report += f"   {engine:12} | Quality: {metrics['quality_score']:.3f} | "
                    report += f"Length: {metrics['text_length']:4d} | Similarity: {similarity:.3f}\n"
                else:
                    report += f"   {engine:12} | No text extracted\n"
            report += "\n"
        
        return report
    
    def plot_quality_comparison(self, parent_frame, all_results):
        """Plot quality comparison bar chart"""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning("Matplotlib Required", "Install matplotlib: pip install matplotlib")
            return
        
        # Clear existing plots
        for widget in parent_frame.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()
        
        # Calculate average quality scores
        engine_scores = {'easyocr': [], 'tesseract': [], 'pypdf2': [], 'openai_ocr': []}
        
        for filename, evaluation in all_results:
            for engine, metrics in evaluation.items():
                if metrics['text_length'] > 0:
                    engine_scores[engine].append(metrics['quality_score'])
        
        engines = []
        scores = []
        for engine, score_list in engine_scores.items():
            if score_list:
                engines.append(engine.upper())
                scores.append(sum(score_list) / len(score_list))
        
        if not engines:
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(engines, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_ylabel('Average Quality Score')
        ax.set_title('OCR Engine Quality Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def plot_similarity_matrix(self, parent_frame, all_results):
        """Plot similarity matrix heatmap"""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning("Matplotlib Required", "Install matplotlib: pip install matplotlib")
            return
        
        # Clear existing plots
        for widget in parent_frame.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()
        
        engines = ['easyocr', 'tesseract', 'pypdf2', 'openai_ocr']
        similarity_matrix = np.zeros((4, 4))
        
        # Calculate pairwise similarities
        for i, engine1 in enumerate(engines):
            for j, engine2 in enumerate(engines):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarities = []
                    for filename, evaluation in all_results:
                        if engine1 in evaluation and engine2 in evaluation:
                            text1 = str(evaluation[engine1].get('text_length', 0))
                            text2 = str(evaluation[engine2].get('text_length', 0))
                            if evaluation[engine1]['text_length'] > 0 and evaluation[engine2]['text_length'] > 0:
                                # Use quality score similarity as proxy
                                score1 = evaluation[engine1]['quality_score']
                                score2 = evaluation[engine2]['quality_score']
                                similarities.append(1 - abs(score1 - score2))
                    
                    if similarities:
                        similarity_matrix[i][j] = sum(similarities) / len(similarities)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(similarity_matrix, cmap='RdYlBu', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([e.upper() for e in engines])
        ax.set_yticklabels([e.upper() for e in engines])
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        ax.set_title('OCR Engine Similarity Matrix')
        plt.colorbar(im, ax=ax, label='Similarity Score')
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def plot_length_vs_quality(self, parent_frame, all_results):
        """Plot text length vs quality scatter plot"""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning("Matplotlib Required", "Install matplotlib: pip install matplotlib")
            return
        
        # Clear existing plots
        for widget in parent_frame.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = {'easyocr': '#FF6B6B', 'tesseract': '#4ECDC4', 'pypdf2': '#45B7D1', 'openai_ocr': '#96CEB4'}
        
        for engine, color in colors.items():
            lengths = []
            qualities = []
            
            for filename, evaluation in all_results:
                if engine in evaluation and evaluation[engine]['text_length'] > 0:
                    lengths.append(evaluation[engine]['text_length'])
                    qualities.append(evaluation[engine]['quality_score'])
            
            if lengths:
                ax.scatter(lengths, qualities, c=color, label=engine.upper(), alpha=0.7, s=50)
        
        ax.set_xlabel('Text Length (characters)')
        ax.set_ylabel('Quality Score')
        ax.set_title('Text Length vs Quality Score by OCR Engine')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

def main():
    root = tk.Tk()
    app = CodebooksApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()