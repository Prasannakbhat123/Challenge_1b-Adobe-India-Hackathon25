"""
Round 1B: Persona-Driven Document Intelligence
Finds most relevant sections and subsections based on persona and job-to-be-done
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class PersonaExtractor:
    """Persona-driven content extractor for Round 1B challenges"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )

    def extract(self, pdf_files: List[Path], persona_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract persona-relevant content in challenge format"""

        # Handle both old and new persona config formats
        if "persona" in persona_config and isinstance(persona_config["persona"], str):
            # Old format: {"persona": "string", "job_to_be_done": "string"}
            persona = persona_config.get("persona", "")
            job_to_be_done = persona_config.get("job_to_be_done", "")
        elif "persona" in persona_config and isinstance(persona_config["persona"], dict) and "role" in persona_config["persona"]:
            # New challenge format: {"persona": {"role": "string"}, "job_to_be_done": {"task": "string"}}
            persona = persona_config["persona"]["role"]
            job_to_be_done = persona_config["job_to_be_done"]["task"]
        else:
            raise ValueError("Invalid persona configuration format")

        if not persona or not job_to_be_done:
            raise ValueError("Persona and job_to_be_done must be provided")

        # Process each PDF and extract relevant sections
        all_sections = []
        all_subsections = []

        for pdf_file in pdf_files:
            try:
                sections, subsections = self._extract_sections_from_pdf(pdf_file, persona, job_to_be_done)
                all_sections.extend(sections)
                all_subsections.extend(subsections)
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                continue

        # Rank sections by relevance
        ranked_sections = self._rank_sections(all_sections, persona, job_to_be_done)
        ranked_subsections = self._rank_subsections(all_subsections, persona, job_to_be_done)

        # Determine output format based on config
        if "challenge_info" in persona_config:
            # New challenge format output (exactly matching specification)
            return {
                "metadata": {
                    "input_documents": [f.name for f in pdf_files],
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": ranked_sections[:5],  # Top 5 sections
                "subsection_analysis": ranked_subsections[:5]  # Top 5 subsections
            }
        else:
            # Original format output for backward compatibility
            return {
                "metadata": {
                    "input_documents": [f.name for f in pdf_files],
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "timestamp": datetime.now().isoformat() + "Z"
                },
                "sections": ranked_sections[:10],  # Top 10 sections
                "subsections": ranked_subsections[:15]  # Top 15 subsections
            }

    def _extract_sections_from_pdf(self, pdf_file: Path, persona: str, job_to_be_done: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract sections and subsections from a single PDF"""
        
        doc = fitz.open(pdf_file)
        sections = []
        subsections = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            
            if not page_text.strip():
                continue

            # Extract potential section headings with better context
            section_headings = self._find_section_headings(page_text)
            
            for heading in section_headings:
                # Get surrounding context for better content analysis
                heading_pos = page_text.find(heading)
                if heading_pos != -1:
                    # Extract context around the heading (200 chars before and after)
                    start_pos = max(0, heading_pos - 200)
                    end_pos = min(len(page_text), heading_pos + len(heading) + 200)
                    context = page_text[start_pos:end_pos]
                    
                    sections.append({
                        "document": pdf_file.name,
                        "page": page_num + 1,
                        "section_title": heading,
                        "content": context
                    })

            # Create subsections by chunking page text with larger chunks for better context
            chunks = self._chunk_text(page_text, chunk_size=1800, overlap=200)  # Larger chunks for more comprehensive summaries
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 150:  # Minimum meaningful content length
                    # Create refined text (comprehensive summary)
                    refined_text = self._create_summary(chunk, persona, job_to_be_done)
                    
                    # Only include if refined text is substantial (lowered threshold for more content)
                    if len(refined_text) > 80:
                        subsections.append({
                            "document": pdf_file.name,
                            "page": page_num + 1,
                            "chunk_id": i,
                            "content": chunk,
                            "refined_text": refined_text
                        })

        doc.close()
        return sections, subsections

    def _find_section_headings(self, text: str) -> List[str]:
        """Find potential section headings in text with improved detection"""
        lines = text.split('\n')
        headings = []
        
        # Enhanced heading patterns
        heading_patterns = [
            r'^\d+\.?\s+[A-Z][A-Za-z\s:]+$',  # Numbered headings like "1. Introduction"
            r'^[A-Z][A-Z\s&]+$',  # All caps headings like "INTRODUCTION"
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title case like "Introduction to AI"
            r'^Chapter\s+\d+[:\-\s]*[A-Za-z\s]+$',   # Chapter headings
            r'^Section\s+\d+[:\-\s]*[A-Za-z\s]+$',   # Section headings
            r'^Part\s+[IVX\d]+[:\-\s]*[A-Za-z\s]+$', # Part headings
            r'^[A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)*$',  # Mixed case headings
        ]
        
        for line in lines:
            line = line.strip()
            # Reasonable heading length (not too short or too long)
            if 3 < len(line) < 80:
                # Check if line contains mostly letters and spaces (not dense text)
                if len(line.split()) <= 8:  # Headings are typically short phrases
                    for pattern in heading_patterns:
                        if re.match(pattern, line):
                            # Additional checks to avoid false positives
                            word_count = len(line.split())
                            if 1 <= word_count <= 8:  # Reasonable word count for headings
                                # Avoid lines that are clearly not headings
                                if not any(char in line for char in '.,;()[]{}'):
                                    headings.append(line)
                                    break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_headings = []
        for heading in headings:
            if heading not in seen:
                seen.add(heading)
                unique_headings.append(heading)
        
        return unique_headings[:10]  # Limit to top 10 headings per page

    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within last 100 characters
                search_start = max(end - 100, start)
                sentence_end = -1
                
                for i in range(end, search_start - 1, -1):
                    if text[i] in '.!?':
                        sentence_end = i + 1
                        break
                
                if sentence_end > start:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else len(text)
        
        return chunks

    def _create_summary(self, text: str, persona: str, job_to_be_done: str) -> str:
        """Create a comprehensive extractive summary with longer, more detailed content"""
        sentences = re.split(r'[.!?]+', text)
        
        # Filter and rank sentences by relevance
        relevant_sentences = []
        
        # Enhanced keyword matching
        persona_keywords = persona.lower().split()
        job_keywords = job_to_be_done.lower().split()
        all_keywords = persona_keywords + job_keywords
        
        # Add common travel/planning related keywords for better matching
        context_keywords = ['plan', 'planning', 'trip', 'travel', 'guide', 'experience', 
                           'activities', 'recommendations', 'tips', 'advice', 'visit', 
                           'explore', 'enjoy', 'discover', 'location', 'destination']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:  # Minimum sentence length
                sentence_lower = sentence.lower()
                
                # Count keyword matches (including context keywords)
                keyword_matches = sum(1 for keyword in all_keywords + context_keywords 
                                    if keyword in sentence_lower)
                
                # Score sentences based on length and keyword relevance
                length_score = min(len(sentence) / 100, 1.0)  # Prefer longer sentences
                relevance_score = keyword_matches * 0.5
                total_score = length_score + relevance_score
                
                if total_score > 0.3:  # Lower threshold for more inclusive selection
                    relevant_sentences.append((sentence, total_score))
        
        # Sort by score and take best sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Create comprehensive summary
        if relevant_sentences:
            # Take top sentences up to a reasonable length
            selected_sentences = []
            total_length = 0
            target_length = 900  # Much longer target length to match expected output
            
            for sentence, score in relevant_sentences:
                if total_length + len(sentence) <= target_length or len(selected_sentences) < 3:
                    selected_sentences.append(sentence)
                    total_length += len(sentence) + 2  # +2 for '. ' separator
                    
                    if len(selected_sentences) >= 12:  # More sentences for comprehensive summaries
                        break
            
            summary = '. '.join(selected_sentences)
        else:
            # Fallback: take first few sentences that form coherent content
            coherent_sentences = []
            total_length = 0
            target_length = 800  # Much longer fallback length to match expected output
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15 and total_length + len(sentence) <= target_length:
                    coherent_sentences.append(sentence)
                    total_length += len(sentence) + 2
                    
                    if len(coherent_sentences) >= 10:  # More fallback sentences
                        break
            
            summary = '. '.join(coherent_sentences)
        
        # Ensure summary ends properly and isn't too long
        if summary and not summary.endswith('.'):
            summary += '.'
        
        # Clean up newlines and extra whitespace
        summary = ' '.join(summary.split())  # This removes all \n and normalizes whitespace
        
        # Limit to reasonable length while preserving complete sentences
        if len(summary) > 800:  # Allow much longer summaries to match expected output
            # Find last complete sentence within limit
            sentences_in_summary = summary.split('. ')
            trimmed_summary = ""
            for i, sent in enumerate(sentences_in_summary):
                test_summary = '. '.join(sentences_in_summary[:i+1])
                if len(test_summary) <= 750:  # Allow up to 750 characters
                    trimmed_summary = test_summary
                else:
                    break
            summary = trimmed_summary if trimmed_summary else summary[:750] + "..."
        
        return summary if summary else text[:200].replace('\n', ' ') + "..."

    def _rank_sections(self, sections: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
        """Rank sections by relevance using enhanced TF-IDF similarity with persona-job matching"""
        if not sections:
            return []

        # Create enhanced query from persona and job with domain-specific keywords
        persona_lower = persona.lower()
        job_lower = job_to_be_done.lower()
        
        # Add domain-specific keywords based on persona
        domain_keywords = {
            'researcher': ['research', 'study', 'analysis', 'methodology', 'findings', 'results'],
            'student': ['learning', 'concepts', 'theory', 'examples', 'practice', 'understanding'],
            'planner': ['planning', 'strategy', 'organization', 'coordination', 'management', 'logistics'],
            'analyst': ['analysis', 'data', 'trends', 'patterns', 'evaluation', 'assessment'],
            'wedding': ['venue', 'ceremony', 'reception', 'guests', 'catering', 'decor', 'budget']
        }
        
        additional_keywords = []
        for domain, keywords in domain_keywords.items():
            if domain in persona_lower or domain in job_lower:
                additional_keywords.extend(keywords)
        
        # Enhanced query construction
        query_parts = [persona, job_to_be_done] + additional_keywords
        query = ' '.join(query_parts)
        
        # Extract section content for vectorization
        section_texts = []
        for section in sections:
            # Combine title and content for better matching
            title = section.get('section_title', '')
            content = section.get('content', '')
            combined_text = f"{title} {content}"
            section_texts.append(combined_text)
        
        if not section_texts:
            return []

        try:
            # Vectorize content with enhanced parameters
            vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams for better context
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )
            
            all_texts = [query] + section_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate similarity
            query_vector = tfidf_matrix[0]
            content_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, content_vectors)[0]
            
            # Enhanced scoring with multiple factors
            for i, section in enumerate(sections):
                base_similarity = float(similarities[i])
                
                # Bonus for title-persona match
                title_lower = section['section_title'].lower()
                title_bonus = 0.1 * sum(1 for word in persona.lower().split() if word in title_lower)
                
                # Bonus for job-specific keywords in title
                job_bonus = 0.15 * sum(1 for word in job_to_be_done.lower().split() if word in title_lower)
                
                # Combined score
                final_score = base_similarity + title_bonus + job_bonus
                section['similarity_score'] = final_score
            
            # Sort by similarity
            ranked_sections = sorted(sections, key=lambda x: x['similarity_score'], reverse=True)
            
            # Format for challenge output
            final_sections = []
            for i, section in enumerate(ranked_sections):
                final_sections.append({
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "importance_rank": i + 1,
                    "page_number": section["page"]
                })
            
            return final_sections
        
        except Exception as e:
            print(f"Error in ranking sections: {e}")
            # Return sections with default ranking
            final_sections = []
            for i, section in enumerate(sections):
                final_sections.append({
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "importance_rank": i + 1,
                    "page_number": section["page"]
                })
            return final_sections

    def _rank_subsections(self, subsections: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
        """Rank subsections by relevance using enhanced TF-IDF similarity with persona-job matching"""
        if not subsections:
            return []

        # Create enhanced query similar to sections
        persona_lower = persona.lower()
        job_lower = job_to_be_done.lower()
        
        # Domain-specific keywords for better matching
        domain_keywords = {
            'researcher': ['methodology', 'findings', 'analysis', 'data', 'research', 'study'],
            'student': ['concepts', 'theory', 'examples', 'learning', 'understanding', 'practice'],
            'planner': ['planning', 'coordination', 'logistics', 'management', 'organization'],
            'analyst': ['trends', 'patterns', 'evaluation', 'assessment', 'analysis'],
            'wedding': ['ceremony', 'reception', 'guests', 'venue', 'budget', 'planning']
        }
        
        additional_keywords = []
        for domain, keywords in domain_keywords.items():
            if domain in persona_lower or domain in job_lower:
                additional_keywords.extend(keywords)
        
        query_parts = [persona, job_to_be_done] + additional_keywords
        query = ' '.join(query_parts)
        
        # Extract subsection content for vectorization
        subsection_texts = []
        for subsection in subsections:
            # Use both content and refined_text for better analysis
            content = subsection.get('content', '')
            refined = subsection.get('refined_text', '')
            combined_text = f"{content} {refined}"
            subsection_texts.append(combined_text)
        
        if not subsection_texts:
            return []

        try:
            # Enhanced vectorization
            vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )
            
            all_texts = [query] + subsection_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate similarity
            query_vector = tfidf_matrix[0]
            content_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, content_vectors)[0]
            
            # Enhanced scoring
            for i, subsection in enumerate(subsections):
                base_similarity = float(similarities[i])
                
                # Bonus for persona-related terms in refined text
                refined_text = subsection.get('refined_text', '').lower()
                persona_bonus = 0.1 * sum(1 for word in persona.lower().split() if word in refined_text)
                
                # Bonus for job-related terms
                job_bonus = 0.15 * sum(1 for word in job_to_be_done.lower().split() if word in refined_text)
                
                # Quality bonus for longer, more detailed refined text
                length_bonus = min(0.1, len(refined_text) / 2000)  # Bonus for substantial content
                
                final_score = base_similarity + persona_bonus + job_bonus + length_bonus
                subsection['similarity_score'] = final_score
            
            # Sort by similarity
            ranked_subsections = sorted(subsections, key=lambda x: x['similarity_score'], reverse=True)
            
            # Format for challenge output
            final_subsections = []
            for subsection in ranked_subsections:
                final_subsections.append({
                    "document": subsection["document"],
                    "refined_text": subsection["refined_text"],
                    "page_number": subsection["page"]
                })
            
            return final_subsections
        
        except Exception as e:
            print(f"Error in ranking subsections: {e}")
            # Return subsections with default formatting
            final_subsections = []
            for subsection in subsections:
                final_subsections.append({
                    "document": subsection["document"],
                    "refined_text": subsection["refined_text"],
                    "page_number": subsection["page"]
                })
            return final_subsections
