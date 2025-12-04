"""
Data cleaning of the final output txts from the PDF chunking process.
This module cleans text files by removing unnecessary characters, extra spaces,
dot lines, and formatting artifacts from PDF extraction.
"""

import re
import os
from pathlib import Path


class TextCleaner:
    """Clean text files from PDF extraction artifacts and unnecessary characters."""
    
    def __init__(self):
        """Initialize the TextCleaner with cleaning patterns."""
        # Patterns for cleaning
        self.patterns = {
            # Remove dot lines (e.g., "........" or "---" or "___")
            'dot_lines': re.compile(r'^[\.\-_\s]{3,}$', re.MULTILINE),
            
            # Remove excessive whitespace (3+ spaces)
            'excessive_spaces': re.compile(r' {3,}'),
            
            # Remove multiple blank lines (3+ consecutive newlines)
            'multiple_newlines': re.compile(r'\n{4,}'),
            
            # Remove trailing whitespace at end of lines
            'trailing_spaces': re.compile(r'[ \t]+$', re.MULTILINE),
            
            # Remove leading whitespace at start of lines (but preserve paragraph indentation)
            'leading_spaces': re.compile(r'^[ \t]+', re.MULTILINE),
            
            # Remove page numbers that appear alone on a line (e.g., "1/142")
            # Allow leading whitespace to catch indented page numbers
            'page_numbers': re.compile(r'^\s*\d+/\d+\s*$', re.MULTILINE),
            
            # Exeptional patterns
            # Remove header/footer patterns (e.g., "SZE SZMSZ-HKR-TVSZ" followed by date)
            #'headers': re.compile(r'^SZE SZMSZ-HKR-TVSZ\s*$', re.MULTILINE),
            #'date_lines': re.compile(r'^Hatályos:\s*\d{4}\.\d{2}\.\d{2}\s*-\s*$', re.MULTILINE),

        }
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Clean text by removing unnecessary characters and formatting.
        
        Args:
            text: The input text to clean
            aggressive: If True, applies more aggressive cleaning (may remove valid content)
            
        Returns:
            Cleaned text string
        """
        # Step 1: Remove headers and footers
        #text = self.patterns['headers'].sub('', text)
        #text = self.patterns['date_lines'].sub('', text)
        text = self.patterns['page_numbers'].sub('', text)
        
        # Step 2: Remove dot/dash lines
        text = self.patterns['dot_lines'].sub('', text)
        
        # Step 3: Remove trailing spaces from lines
        text = self.patterns['trailing_spaces'].sub('', text)
        
        # Step 4: Normalize spaces (replace 3+ spaces with single space)
        text = self.patterns['excessive_spaces'].sub(' ', text)
        
        # Step 5: Remove leading spaces (optional based on aggressive mode)
        if aggressive:
            text = self.patterns['leading_spaces'].sub('', text)
        
        # Step 6: Reduce multiple blank lines to max 2 newlines
        text = self.patterns['multiple_newlines'].sub('\n\n\n', text)
        
        # Step 7: Remove lines with only whitespace
        text = '\n'.join(line for line in text.split('\n') if line.strip() or line == '')
        
        # Step 8: Normalize newlines at start and end
        text = text.strip()
        
        return text
    
    def clean_file(self, input_path: str, output_path: str = None, aggressive: bool = False) -> str:
        """
        Clean a text file and save the result.
        
        Args:
            input_path: Path to the input file
            output_path: Path to save cleaned file (if None, overwrites input)
            aggressive: If True, applies more aggressive cleaning
            
        Returns:
            Path to the cleaned file
        """
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean the text
        cleaned_text = self.clean_text(text, aggressive=aggressive)
        
        # Determine output path
        if output_path is None:
            output_path = input_path
        
        # Write cleaned text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        return output_path
    
    def clean_directory(self, input_dir: str, output_dir: str = None, 
                       pattern: str = "*.txt", aggressive: bool = False) -> list:
        """
        Clean all files in a directory matching the pattern.
        
        Args:
            input_dir: Directory containing files to clean
            output_dir: Directory to save cleaned files (if None, creates 'cleaned' subdirectory)
            pattern: File pattern to match (default: "*.txt")
            aggressive: If True, applies more aggressive cleaning
            
        Returns:
            List of cleaned file paths
        """
        input_path = Path(input_dir)
        
        # Set up output directory
        if output_dir is None:
            output_path = input_path / 'cleaned'
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        cleaned_files = []
        
        # Process all matching files
        for file_path in input_path.glob(pattern):
            if file_path.is_file():
                output_file = output_path / file_path.name
                self.clean_file(str(file_path), str(output_file), aggressive=aggressive)
                cleaned_files.append(str(output_file))
                print(f"Cleaned: {file_path.name} -> {output_file}")
        
        return cleaned_files


def main():
    """Example usage of TextCleaner."""
    cleaner = TextCleaner()
    
    # Get absolute path to project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    CLEAN_SINGLE_FILE = False  # Set to False to clean a directory instead
    CLEAN_MULTIPLE_FILES = True

    if CLEAN_SINGLE_FILE:
        CLEAN_MULTIPLE_FILES = False
        # Example 1: Clean a single file
        input_file = os.path.join(PROJECT_ROOT, "data", "output", "final_merged", "TVSZ_(20250225_)_(2)_final.txt")
        output_file = os.path.join(PROJECT_ROOT, "data", "output", "final_merged", "TVSZ_(20250225_)_(2)_final_cleaned.txt")
        
        if os.path.exists(input_file):
            cleaner.clean_file(input_file, output_file, aggressive=False)
            print(f"File cleaned: {output_file}")
    
    # Clean all files in a directory
    if CLEAN_MULTIPLE_FILES:
        input_directory = os.path.join(PROJECT_ROOT, "data", "output", "final_merged")
        cleaned_files = cleaner.clean_directory(input_directory, aggressive=False)
        print(f"Cleaned {len(cleaned_files)} files. Current files in directory: {os.listdir(input_directory)}")


if __name__ == "__main__":
    main()
