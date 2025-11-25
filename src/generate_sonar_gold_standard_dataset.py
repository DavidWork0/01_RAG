"""
GPT-4 / Perplexity Gold Standard Validation Dataset Generator

This script generates a validation dataset by having a powerful LLM (GPT-4 or Perplexity)
answer questions with full access to the PDF documentation. This serves as the "ground truth"
reference for evaluating your RAG system's performance.

No embeddings or retrieval - just the LLM reading the documentation like an expert.

Supported APIs:
  - OpenAI (gpt-4o, gpt-4o-mini) via OPENAI_API_KEY
  - Perplexity (llama-3.1-sonar models) via PERPLEXITY_API_KEY

Usage:
    python generate_gpt4_gold_standard_dataset.py
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv


class GPT4GoldStandardGenerator:
    """Generate gold standard validation dataset using GPT-4 with full PDF access."""
    
    def __init__(
        self,
        pdf_path: str,
        questions_path: str,
        output_path: str,
        openai_api_key: str,
        gpt_model: str = "gpt-4o",
        max_pages_per_call: int = 100,
        api_base_url: Optional[str] = None,
    ):
        """
        Initialize the gold standard generator.
        
        Args:
            pdf_path: Path to the PDF document
            questions_path: Path to the JSON file containing test questions
            output_path: Path where the validation dataset will be saved
            openai_api_key: OpenAI API key
            gpt_model: GPT model to use (gpt-4o or gpt-4o-mini recommended)
            max_pages_per_call: Maximum pages to send in one API call (for context limits)
        """
        self.pdf_path = Path(pdf_path)
        self.questions_path = Path(questions_path)
        self.output_path = Path(output_path)
        self.gpt_model = gpt_model
        self.max_pages_per_call = max_pages_per_call
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client (works with Perplexity too)
        if api_base_url:
            self.openai_client = OpenAI(api_key=openai_api_key, base_url=api_base_url)
        else:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        # PDF content cache
        self.pdf_full_text = None
        self.pdf_pages = []
        
    def load_pdf(self):
        """Load and extract all text from the PDF."""
        print(f"\nLoading PDF from: {self.pdf_path}")
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        doc = fitz.open(str(self.pdf_path))
        self.pdf_pages = []
        
        print(f"Extracting text from {len(doc)} pages...")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                self.pdf_pages.append({
                    "page_number": page_num + 1,
                    "text": text
                })
            
            if (page_num + 1) % 100 == 0:
                print(f"  Processed {page_num + 1}/{len(doc)} pages...")
        
        doc.close()
        
        # Create full text
        self.pdf_full_text = "\n\n".join([
            f"--- Page {p['page_number']} ---\n{p['text']}"
            for p in self.pdf_pages
        ])
        
        print(f"Successfully loaded {len(self.pdf_pages)} pages")
        print(f"Total text length: {len(self.pdf_full_text):,} characters")
        
    def load_questions(self) -> List[Dict[str, Any]]:
        """Load test questions from JSON file."""
        print(f"\nLoading questions from: {self.questions_path}")
        
        with open(self.questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get('test_questions', [])
        print(f"Loaded {len(questions)} questions")
        
        return questions
    
    def _chunk_pages_for_context(self, max_chars: int = 100000) -> List[str]:
        """
        Split PDF pages into chunks that fit within GPT-4's context window.
        
        Args:
            max_chars: Maximum characters per chunk (conservative estimate)
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for page in self.pdf_pages:
            page_text = f"--- Page {page['page_number']} ---\n{page['text']}\n\n"
            page_length = len(page_text)
            
            if current_length + page_length > max_chars and current_chunk:
                # Save current chunk and start new one
                chunks.append("".join(current_chunk))
                current_chunk = [page_text]
                current_length = page_length
            else:
                current_chunk.append(page_text)
                current_length += page_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks
    
    def generate_reference_answer_full_context(
        self, 
        question: str, 
        category: str
    ) -> Dict[str, Any]:
        """
        Generate reference answer using GPT-4 with access to full PDF.
        
        Args:
            question: The question to answer
            category: Question category
            
        Returns:
            Dictionary with answer, confidence, and metadata
        """
        # Handle non-relevant questions
        if category == "animals_non_relevant":
            return {
                "answer": "This question is not relevant to the ISVD documentation.",
                "confidence": "N/A",
                "reasoning": "Question outside documentation scope",
                "pages_referenced": []
            }
        
        # For large PDFs, we need to be smart about context
        # Strategy: Send full text if possible, otherwise use iterative approach
        
        system_prompt = """You are an expert technical documentation assistant for IBM Security Verify Directory (ISVD).
Your task is to provide accurate, comprehensive answers based on the provided documentation.

Guidelines:
- Answer ONLY based on the provided documentation
- Be precise and technically accurate
- Include specific details: commands, paths, error codes, configuration settings
- Reference page numbers when providing information
- If information spans multiple sections, synthesize it coherently
- If the documentation doesn't contain the answer, clearly state what's missing
- Use technical terminology correctly
- Provide step-by-step instructions when relevant"""

        user_prompt = f"""Based on the complete ISVD documentation provided below, please answer the following question:

Question: {question}

Documentation:
{self.pdf_full_text[:200000]}  

Please provide:
1. A comprehensive answer to the question
2. Specific page numbers where you found the information
3. Your confidence level (High/Medium/Low)
4. Brief reasoning for your answer

Format your response as:
ANSWER: [your detailed answer]
PAGES: [page numbers, e.g., "45, 67-69, 123"]
CONFIDENCE: [High/Medium/Low]
REASONING: [why you're confident/not confident]"""

        try:
            print(f"    Calling {self.gpt_model} with full documentation context...")
            
            response = self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Very low temperature for factual accuracy
                max_tokens=2000,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the structured response
            result = self._parse_gpt4_response(response_text)
            return result
            
        except Exception as e:
            print(f"    Error generating reference answer: {e}")
            return {
                "answer": f"Error generating reference answer: {str(e)}",
                "confidence": "N/A",
                "reasoning": "API error",
                "pages_referenced": []
            }
    
    def _parse_gpt4_response(self, response_text: str) -> Dict[str, Any]:
        """Parse GPT-4's structured response."""
        result = {
            "answer": "",
            "confidence": "Unknown",
            "reasoning": "",
            "pages_referenced": []
        }
        
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('ANSWER:'):
                current_section = 'answer'
                result['answer'] = line.replace('ANSWER:', '').strip()
            elif line.startswith('PAGES:'):
                current_section = 'pages'
                pages_text = line.replace('PAGES:', '').strip()
                # Parse page numbers
                result['pages_referenced'] = self._parse_page_numbers(pages_text)
            elif line.startswith('CONFIDENCE:'):
                current_section = 'confidence'
                result['confidence'] = line.replace('CONFIDENCE:', '').strip()
            elif line.startswith('REASONING:'):
                current_section = 'reasoning'
                result['reasoning'] = line.replace('REASONING:', '').strip()
            elif current_section and line:
                # Continue previous section
                if current_section == 'answer':
                    result['answer'] += ' ' + line
                elif current_section == 'reasoning':
                    result['reasoning'] += ' ' + line
        
        return result
    
    def _parse_page_numbers(self, pages_text: str) -> List[int]:
        """Parse page numbers from text like '45, 67-69, 123'."""
        if not pages_text or pages_text.lower() in ['none', 'n/a', 'unknown']:
            return []
        
        pages = []
        parts = pages_text.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                # Range like "67-69"
                try:
                    start, end = part.split('-')
                    pages.extend(range(int(start), int(end) + 1))
                except:
                    pass
            else:
                # Single page
                try:
                    pages.append(int(part))
                except:
                    pass
        
        return sorted(list(set(pages)))
    
    def generate_validation_dataset(self) -> Dict[str, Any]:
        """Generate the complete gold standard validation dataset."""
        questions = self.load_questions()
        
        validation_data = {
            "metadata": {
                "source_pdf": str(self.pdf_path),
                "questions_file": str(self.questions_path),
                "generated_date": datetime.now().isoformat(),
                "gpt_model": self.gpt_model,
                "approach": "gold_standard_full_context",
                "description": "GPT-4 answers with full PDF access - no retrieval/embeddings",
                "total_questions": len(questions),
                "pdf_pages": len(self.pdf_pages),
                "pdf_characters": len(self.pdf_full_text),
            },
            "validation_entries": []
        }
        
        print(f"\n{'='*70}")
        print(f"Generating Gold Standard Dataset with {self.gpt_model}")
        print(f"{'='*70}")
        print(f"Processing {len(questions)} questions with full PDF context...")
        
        for idx, question_data in enumerate(questions, 1):
            question_id = question_data.get('id')
            question_text = question_data.get('question')
            category = question_data.get('category', 'unknown')
            tags = question_data.get('tags', [])
            
            print(f"\n[{idx}/{len(questions)}] Question ID {question_id}")
            print(f"  Category: {category}")
            print(f"  Question: {question_text[:100]}...")
            
            # Generate reference answer with full PDF context
            reference = self.generate_reference_answer_full_context(
                question_text, 
                category
            )
            
            # Create validation entry
            entry = {
                "question_id": question_id,
                "question": question_text,
                "category": category,
                "tags": tags,
                "reference_answer": reference["answer"],
                "confidence": reference["confidence"],
                "reasoning": reference["reasoning"],
                "pages_referenced": reference["pages_referenced"],
                "approach": "full_context_sonar"
            }
            
            print(f"  Confidence: {reference['confidence']}")
            print(f"  Pages: {reference['pages_referenced'][:5]}{'...' if len(reference['pages_referenced']) > 5 else ''}")
            
            validation_data["validation_entries"].append(entry)
        
        return validation_data
    
    def save_validation_dataset(self, validation_data: Dict[str, Any]):
        """Save validation dataset to JSON file."""
        print(f"\n{'='*70}")
        print(f"Saving gold standard dataset to: {self.output_path}")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)
        
        print(f"Gold standard dataset saved successfully!")
        print(f"Total entries: {len(validation_data['validation_entries'])}")
        
        # Print summary
        self._print_summary(validation_data)
    
    def _print_summary(self, validation_data: Dict[str, Any]):
        """Print summary statistics."""
        entries = validation_data['validation_entries']
        
        print(f"\n{'='*70}")
        print("GOLD STANDARD DATASET SUMMARY")
        print(f"{'='*70}")
        
        # Count by category
        categories = {}
        confidence_counts = {"High": 0, "Medium": 0, "Low": 0, "N/A": 0, "Unknown": 0}
        
        for entry in entries:
            cat = entry.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            conf = entry.get('confidence', 'Unknown')
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        print(f"\nTotal Questions: {len(entries)}")
        print(f"\nBreakdown by Category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        
        print(f"\nConfidence Distribution:")
        for conf, count in sorted(confidence_counts.items()):
            if count > 0:
                print(f"  {conf}: {count}")
        
        # Show sample
        if entries:
            sample = entries[0]
            print(f"\nSample Entry (Question ID: {sample['question_id']}):")
            print(f"  Question: {sample['question'][:80]}...")
            print(f"  Answer: {sample['reference_answer'][:100]}...")
            print(f"  Confidence: {sample['confidence']}")
        
        print(f"{'='*70}\n")
    
    def run(self):
        """Execute the complete gold standard generation pipeline."""
        print(f"{'='*70}")
        print("GPT-4 Gold Standard Validation Dataset Generator")
        print(f"{'='*70}")
        print(f"Model: {self.gpt_model}")
        print(f"Approach: Full PDF context (no embeddings/retrieval)")
        
        # Load PDF
        self.load_pdf()
        
        # Generate validation dataset
        validation_data = self.generate_validation_dataset()
        
        # Save to file
        self.save_validation_dataset(validation_data)
        
        print("\n✓ Gold standard dataset generation completed!")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Check for Perplexity API first, then OpenAI
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Determine which API to use
    if perplexity_api_key:
        print("Using Perplexity API")
        api_key = perplexity_api_key
        api_base_url = "https://api.perplexity.ai"
        # Valid models: llama-3.1-sonar-small-128k-online, llama-3.1-sonar-large-128k-online, 
        # llama-3.1-sonar-huge-128k-online (check your account tier)
        default_model = "sonar-reasoning-pro"  # Most widely available model
    elif openai_api_key:
        print("Using OpenAI API")
        api_key = openai_api_key
        api_base_url = None
        default_model = "gpt-4o"
    else:
        raise ValueError(
            "No API key found. Please set either PERPLEXITY_API_KEY or OPENAI_API_KEY in your .env file."
        )
    
    # Define paths
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "data" / "pdfs" / "svd_10_0_3_documentation.pdf"
    questions_path = project_root / "data" / "test" / "inference_test_questions.json"
    output_path = project_root / "data" / "test" / "gpt4_gold_standard_dataset.json"
    
    # Configuration
    config = {
        "pdf_path": str(pdf_path),
        "questions_path": str(questions_path),
        "output_path": str(output_path),
        "openai_api_key": api_key,
        "gpt_model": default_model,
        "max_pages_per_call": 100,
        "api_base_url": api_base_url,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        if key != "openai_api_key":
            print(f"  {key}: {value}")
    print()
    
    # Create generator and run
    generator = GPT4GoldStandardGenerator(**config)
    generator.run()


if __name__ == "__main__":
    main()
