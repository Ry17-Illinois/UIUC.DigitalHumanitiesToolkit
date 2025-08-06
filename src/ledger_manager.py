#!/usr/bin/env python3
"""
Ledger Manager for Dublin Core Metadata
Handles the creation and management of the metadata ledger
"""

import pandas as pd
import os
import uuid
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
        else:
            # Create base columns
            columns = [
                'file_id', 'filename', 'filepath', 'file_type', 'file_size',
                'date_added', 'easyocr_ocr', 'easyocr_status', 'tesseract_ocr', 'tesseract_status', 'pypdf2_ocr', 'pypdf2_status', 'openai_ocr_ocr', 'openai_ocr_status', 'ollama_ocr_ocr', 'ollama_ocr_status'
            ] + self.DUBLIN_CORE_FIELDS + [f"{field}_status" for field in self.DUBLIN_CORE_FIELDS]
            
            df = pd.DataFrame(columns=columns)
        
        return df
    
    def save_ledger(self):
        """Save the ledger to CSV"""
        self.df.to_csv(self.ledger_path, index=False)
    
    def add_files(self, file_paths: List[str]) -> int:
        """Add new files to the ledger"""
        added_count = 0
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
                
            # Check if file already exists
            if not self.df[self.df['filepath'] == file_path].empty:
                continue
            
            file_info = {
                'file_id': str(uuid.uuid4()),
                'filename': os.path.basename(file_path),
                'filepath': file_path,
                'file_type': Path(file_path).suffix.lower(),
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
                'ollama_ocr_status': 'pending'
            }
            
            # Initialize Dublin Core fields
            for field in self.DUBLIN_CORE_FIELDS:
                file_info[field] = ''
                file_info[f"{field}_status"] = 'pending'
            
            # Add to dataframe
            self.df = pd.concat([self.df, pd.DataFrame([file_info])], ignore_index=True)
            added_count += 1
        
        self.save_ledger()
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
    
    def get_files_by_status(self, operation: str, status: str = 'pending') -> pd.DataFrame:
        """Get files by operation status"""
        if operation in ['easyocr', 'tesseract', 'pypdf2', 'openai_ocr', 'ollama_ocr']:
            return self.df[self.df[f'{operation}_status'] == status]
        elif operation in self.DUBLIN_CORE_FIELDS:
            return self.df[self.df[f"{operation}_status"] == status]
        else:
            return pd.DataFrame()
    
    def clear_rows(self, file_ids: List[str]):
        """Clear specified rows from the ledger"""
        self.df = self.df[~self.df['file_id'].isin(file_ids)]
        self.save_ledger()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the ledger"""
        total_files = len(self.df)
        
        summary = {
            'total_files': total_files,
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