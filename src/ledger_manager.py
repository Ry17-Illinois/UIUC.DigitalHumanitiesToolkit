#!/usr/bin/env python3
"""
Ledger Manager for Dublin Core Metadata
Handles the creation and management of the metadata ledger
"""

import pandas as pd
import os
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

class LedgerManager:
    """Manages the metadata ledger with Dublin Core fields"""
    
    # Dublin Core metadata fields
    DUBLIN_CORE_FIELDS = [
        'title', 'creator', 'subject', 'description', 'publisher', 
        'contributor', 'date', 'type', 'format', 'identifier', 
        'source', 'language', 'relation', 'coverage', 'rights'
    ]
    
    def __init__(self, ledger_path: str = "metadata_ledger.csv"):
        self.ledger_path = ledger_path
        self._lock = threading.Lock()
        self.df = self.load_or_create_ledger()
    
    def load_or_create_ledger(self) -> pd.DataFrame:
        """Load existing ledger or create new one"""
        if os.path.exists(self.ledger_path):
            df = pd.read_csv(self.ledger_path)
            
            # Add missing columns if they don't exist
            if 'tesseract_ocr' not in df.columns:
                df['tesseract_ocr'] = ''
            if 'tesseract_status' not in df.columns:
                df['tesseract_status'] = 'pending'
            if 'pypdf2_ocr' not in df.columns:
                df['pypdf2_ocr'] = ''
            if 'pypdf2_status' not in df.columns:
                df['pypdf2_status'] = 'pending'
            if 'openai_ocr_ocr' not in df.columns:
                df['openai_ocr_ocr'] = ''
            if 'openai_ocr_status' not in df.columns:
                df['openai_ocr_status'] = 'pending'
            if 'ollama_ocr_ocr' not in df.columns:
                df['ollama_ocr_ocr'] = ''
            if 'ollama_ocr_status' not in df.columns:
                df['ollama_ocr_status'] = 'pending'
            if 'document_type' not in df.columns:
                df['document_type'] = ''
            if 'document_type_status' not in df.columns:
                df['document_type_status'] = 'pending'
            if 'named_entities' not in df.columns:
                df['named_entities'] = ''
        else:
            # Create base columns
            columns = [
                'file_id', 'filename', 'filepath', 'file_type', 'file_size',
                'date_added', 'easyocr_ocr', 'easyocr_status', 'tesseract_ocr', 'tesseract_status', 'pypdf2_ocr', 'pypdf2_status', 'openai_ocr_ocr', 'openai_ocr_status', 'ollama_ocr_ocr', 'ollama_ocr_status',
                'document_type', 'document_type_status', 'named_entities'
            ] + self.DUBLIN_CORE_FIELDS + [f"{field}_status" for field in self.DUBLIN_CORE_FIELDS]
            
            df = pd.DataFrame(columns=columns)
        
        return df
    
    def save_ledger(self):
        """Save the ledger to CSV with thread safety"""
        with self._lock:
            print(f"DEBUG: Saving ledger with {len(self.df)} rows to {self.ledger_path}")
            self.df.to_csv(self.ledger_path, index=False)
            print(f"DEBUG: Ledger saved successfully")
    
    def add_files(self, file_paths: List[str]) -> int:
        """Add new files to the ledger"""
        print(f"DEBUG: add_files called with {len(file_paths)} paths")
        added_count = 0
        new_files = []
        
        for i, file_path in enumerate(file_paths):
            print(f"DEBUG: Processing file {i+1}/{len(file_paths)}: {file_path}")
            if not os.path.exists(file_path):
                print(f"DEBUG: File does not exist, skipping")
                continue
                
            # Check if file already exists
            if not self.df[self.df['filepath'] == file_path].empty:
                print(f"DEBUG: File already in ledger, skipping")
                continue
            
            file_type = Path(file_path).suffix.lower()
            
            file_info = {
                'file_id': str(uuid.uuid4()),
                'filename': os.path.basename(file_path),
                'filepath': file_path,
                'file_type': file_type,
                'file_size': os.path.getsize(file_path),
                'date_added': pd.Timestamp.now(),
                'easyocr_ocr': '',
                'easyocr_status': 'pending',
                'tesseract_ocr': '',
                'tesseract_status': 'pending',
                'pypdf2_ocr': '',
                'pypdf2_status': 'pending',
                'openai_ocr_ocr': '',
                'openai_ocr_status': 'pending',
                'ollama_ocr_ocr': '',
                'ollama_ocr_status': 'pending',
                'document_type': 'document' if file_type == '.pdf' else '',
                'document_type_status': 'completed' if file_type == '.pdf' else 'pending',
                'named_entities': ''
            }
            
            # Initialize Dublin Core fields
            for field in self.DUBLIN_CORE_FIELDS:
                file_info[field] = ''
                file_info[f"{field}_status"] = 'pending'
            
            new_files.append(file_info)
            added_count += 1
            print(f"DEBUG: Added file to batch, total so far: {added_count}")
        
        # Add all files at once
        if new_files:
            print(f"DEBUG: Adding {len(new_files)} files to dataframe")
            self.df = pd.concat([self.df, pd.DataFrame(new_files)], ignore_index=True)
            print(f"DEBUG: Dataframe now has {len(self.df)} rows")
            self.save_ledger()
        else:
            print(f"DEBUG: No new files to add")
        
        return added_count
    
    def update_ocr_result(self, file_id: str, ocr_text: str, status: str = 'completed', model: str = 'easyocr'):
        """Update OCR results"""
        mask = self.df['file_id'] == file_id
        self.df.loc[mask, f'{model}_ocr'] = ocr_text
        self.df.loc[mask, f'{model}_status'] = status
        self.save_ledger()
    
    def update_dublin_core_field(self, file_id: str, field: str, value: str, status: str = 'completed'):
        """Update a Dublin Core metadata field"""
        if field not in self.DUBLIN_CORE_FIELDS:
            raise ValueError(f"Invalid Dublin Core field: {field}")
        
        mask = self.df['file_id'] == file_id
        self.df.loc[mask, field] = value
        self.df.loc[mask, f"{field}_status"] = status
        self.save_ledger()
    
    def update_document_type(self, file_id: str, doc_type: str, status: str = 'completed'):
        """Update document type classification"""
        mask = self.df['file_id'] == file_id
        self.df.loc[mask, 'document_type'] = doc_type
        self.df.loc[mask, 'document_type_status'] = status
        self.save_ledger()
    
    def update_named_entities(self, file_id: str, entities: str):
        """Update named entities for a file"""
        mask = self.df['file_id'] == file_id
        self.df.loc[mask, 'named_entities'] = entities
        self.save_ledger()
    
    def get_files_by_status(self, operation: str, status: str = 'pending') -> pd.DataFrame:
        """Get files by operation status"""
        if operation in ['easyocr', 'tesseract', 'pypdf2', 'openai_ocr', 'ollama_ocr', 'document_type']:
            return self.df[self.df[f'{operation}_status'] == status]
        elif operation in self.DUBLIN_CORE_FIELDS:
            return self.df[self.df[f"{operation}_status"] == status]
        else:
            return pd.DataFrame()
    
    def clear_rows(self, file_ids: List[str]):
        """Clear specified rows from the ledger"""
        self.df = self.df[~self.df['file_id'].isin(file_ids)]
        self.save_ledger()
    
    def get_file_id_by_path(self, filepath: str) -> str:
        """Get file ID by filepath"""
        matching_rows = self.df[self.df['filepath'] == filepath]
        if not matching_rows.empty:
            return matching_rows.iloc[0]['file_id']
        return None
    
    def get_page_count(self, filepath: str) -> int:
        """Get page count for a file (PDF pages or 1 for images)"""
        try:
            file_ext = Path(filepath).suffix.lower()
            if file_ext == '.pdf':
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(filepath)
                    page_count = len(doc)
                    doc.close()
                    return page_count
                except ImportError:
                    return 1  # Fallback if PyMuPDF not available
                except Exception:
                    return 1  # Fallback if PDF can't be opened
            else:
                return 1  # Images count as 1 page
        except Exception:
            return 1  # Fallback
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the ledger with page counts"""
        total_files = len(self.df)
        
        # Calculate total pages with error handling
        total_pages = 0
        for _, row in self.df.iterrows():
            try:
                total_pages += self.get_page_count(row['filepath'])
            except Exception:
                total_pages += 1  # Fallback to 1 page if error
        
        summary = {
            'total_files': total_files,
            'total_pages': total_pages,
            'easyocr_completed': len(self.df[self.df['easyocr_status'] == 'completed']),
            'easyocr_pending': len(self.df[self.df['easyocr_status'] == 'pending']),
            'easyocr_error': len(self.df[self.df['easyocr_status'] == 'error']),
            'tesseract_completed': len(self.df[self.df.get('tesseract_status', pd.Series()) == 'completed']),
            'tesseract_pending': len(self.df[self.df.get('tesseract_status', pd.Series()) == 'pending']),
            'tesseract_error': len(self.df[self.df.get('tesseract_status', pd.Series()) == 'error']),
            'pypdf2_completed': len(self.df[self.df.get('pypdf2_status', pd.Series()) == 'completed']),
            'pypdf2_pending': len(self.df[self.df.get('pypdf2_status', pd.Series()) == 'pending']),
            'pypdf2_error': len(self.df[self.df.get('pypdf2_status', pd.Series()) == 'error']),
            'openai_ocr_completed': len(self.df[self.df.get('openai_ocr_status', pd.Series()) == 'completed']),
            'openai_ocr_pending': len(self.df[self.df.get('openai_ocr_status', pd.Series()) == 'pending']),
            'openai_ocr_error': len(self.df[self.df.get('openai_ocr_status', pd.Series()) == 'error']),
            'ollama_ocr_completed': len(self.df[self.df.get('ollama_ocr_status', pd.Series()) == 'completed']),
            'ollama_ocr_pending': len(self.df[self.df.get('ollama_ocr_status', pd.Series()) == 'pending']),
            'ollama_ocr_error': len(self.df[self.df.get('ollama_ocr_status', pd.Series()) == 'error']),
            'document_type_completed': len(self.df[self.df.get('document_type_status', pd.Series()) == 'completed']),
            'document_type_pending': len(self.df[self.df.get('document_type_status', pd.Series()) == 'pending']),
            'document_type_error': len(self.df[self.df.get('document_type_status', pd.Series()) == 'error']),
            'dublin_core_fields': {}
        }
        
        for field in self.DUBLIN_CORE_FIELDS:
            status_col = f"{field}_status"
            summary['dublin_core_fields'][field] = {
                'completed': len(self.df[self.df[status_col] == 'completed']),
                'pending': len(self.df[self.df[status_col] == 'pending']),
                'error': len(self.df[self.df[status_col] == 'error'])
            }
        
        return summary