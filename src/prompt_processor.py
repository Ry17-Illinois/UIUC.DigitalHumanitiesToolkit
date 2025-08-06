#!/usr/bin/env python3
"""
Prompt-based AI Processor for extracting metadata
"""

import os
import json
from openai import OpenAI
from typing import List, Dict, Tuple, Optional

class PromptProcessor:
    """Handles AI-based metadata extraction using prompts"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI()
        self.prompts = self.load_prompts()
    
    def load_prompts(self) -> Dict[str, str]:
        """Load prompts from the prompts directory"""
        prompts = {}
        prompts_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
        
        if os.path.exists(prompts_dir):
            for filename in os.listdir(prompts_dir):
                if filename.endswith('.txt'):
                    field_name = filename.replace('.txt', '')
                    with open(os.path.join(prompts_dir, filename), 'r', encoding='utf-8') as f:
                        prompts[field_name] = f.read().strip()
        
        return prompts
    
    def generate_examples(self, text: str, field: str, num_examples: int = 3) -> List[str]:
        """Generate examples for a Dublin Core field using AI"""
        if field not in self.prompts:
            return [f"No prompt available for field: {field}"]
        
        prompt = self.prompts[field]
        full_prompt = f"{prompt}\n\nDocument text:\n{text[:2000]}..."  # Limit text length
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a metadata extraction expert specializing in Dublin Core standards."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            result = response.choices[0].message.content
            
            # Try to parse as multiple examples
            examples = []
            lines = result.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Remove numbering if present
                    if line[0].isdigit() and '.' in line[:5]:
                        line = line.split('.', 1)[1].strip()
                    examples.append(line)
            
            # Ensure we have the requested number of examples
            while len(examples) < num_examples and examples:
                examples.append(examples[-1] + " (variant)")
            
            return examples[:num_examples] if examples else ["No examples generated"]
            
        except Exception as e:
            return [f"Error generating examples: {str(e)}"]
    
    def get_available_fields(self) -> List[str]:
        """Get list of available Dublin Core fields with prompts"""
        return list(self.prompts.keys())