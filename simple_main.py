#!/usr/bin/env python3
"""
CODEBOOKS - Simple version without pandas
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import csv
import uuid
from datetime import datetime
from pathlib import Path
import easyocr
from PIL import Image
from pdf2image import convert_from_path

class SimpleLedger:
    def __init__(self, csv_path="simple_ledger.csv"):
        self.csv_path = csv_path
        self.headers = ['file_id', 'filename', 'filepath', 'ocr_text', 'ocr_status', 'title', 'creator', 'subject']
        self.data = self.load_data()
    
    def load_data(self):
        if os.path.exists(self.csv_path):
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
                return list(csv.DictReader(f))
        return []
    
    def save_data(self):
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.data)
    
    def add_files(self, file_paths):
        added = 0
        existing_paths = {row['filepath'] for row in self.data}
        
        for path in file_paths:
            if path not in existing_paths:
                self.data.append({
                    'file_id': str(uuid.uuid4()),
                    'filename': os.path.basename(path),
                    'filepath': path,
                    'ocr_text': '',
                    'ocr_status': 'pending',
                    'title': '',
                    'creator': '',
                    'subject': ''
                })
                added += 1
        
        self.save_data()
        return added
    
    def update_ocr(self, file_id, text, status):
        for row in self.data:
            if row['file_id'] == file_id:
                row['ocr_text'] = text
                row['ocr_status'] = status
                break
        self.save_data()
    
    def get_pending_ocr(self):
        return [row for row in self.data if row['ocr_status'] == 'pending']

class SimpleOCR:
    def __init__(self):
        try:
            self.reader = easyocr.Reader(['en'])
            self.available = True
        except:
            self.available = False
    
    def process_file(self, file_path):
        if not self.available:
            return "EasyOCR not available", "error"
        
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.pdf':
                images = convert_from_path(file_path, dpi=150)
                all_text = []
                for i, img in enumerate(images):
                    results = self.reader.readtext(img)
                    page_text = '\n'.join([result[1] for result in results])
                    all_text.append(f"[Page {i+1}]\n{page_text}")
                return '\n\n'.join(all_text), "completed"
            
            elif ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                results = self.reader.readtext(file_path)
                text = '\n'.join([result[1] for result in results])
                return text, "completed"
            
            else:
                return f"Unsupported file type: {ext}", "error"
                
        except Exception as e:
            return f"Error: {str(e)}", "error"

class SimpleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CODEBOOKS - Simple Version")
        self.root.geometry("800x600")
        
        self.ledger = SimpleLedger()
        self.ocr = SimpleOCR()
        
        self.setup_ui()
        self.refresh_display()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Run OCR", command=self.run_ocr).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh", command=self.refresh_display).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.RIGHT)
        
        # Data display
        data_frame = ttk.LabelFrame(main_frame, text="Files", padding="10")
        data_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ['filename', 'ocr_status', 'title']
        self.tree = ttk.Treeview(data_frame, columns=columns, show='headings')
        
        for col in columns:
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=200)
        
        scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def add_files(self):
        files = filedialog.askopenfilenames(
            title="Select files",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff"), ("PDFs", "*.pdf")]
        )
        
        if files:
            added = self.ledger.add_files(list(files))
            messagebox.showinfo("Success", f"Added {added} files")
            self.refresh_display()
    
    def run_ocr(self):
        if not self.ocr.available:
            messagebox.showerror("Error", "EasyOCR not available. Install with: pip install easyocr")
            return
        
        pending = self.ledger.get_pending_ocr()
        if not pending:
            messagebox.showinfo("Info", "No files pending OCR")
            return
        
        for i, row in enumerate(pending):
            self.status_var.set(f"Processing {i+1}/{len(pending)}: {row['filename']}")
            self.root.update()
            
            text, status = self.ocr.process_file(row['filepath'])
            self.ledger.update_ocr(row['file_id'], text, status)
        
        self.status_var.set("OCR completed")
        self.refresh_display()
    
    def refresh_display(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for row in self.ledger.data:
            self.tree.insert('', 'end', values=[
                row['filename'],
                row['ocr_status'],
                row['title']
            ])

def main():
    root = tk.Tk()
    app = SimpleApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()