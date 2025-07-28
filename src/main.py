#!/usr/bin/env python3
"""
Adobe Hackathon Round 1B - Persona-Driven Document Intelligence
Finds relevant sections and subsections based on persona and job-to-be-done
"""

import os
import sys
import json
import time
from pathlib import Path
from persona_extractor import PersonaExtractor

def main():
    """Main function for Round 1B: Persona-Driven Document Intelligence"""

    print("🚀 Adobe Hackathon Round 1B - Persona-Driven Intelligence Starting...")

    input_dir = Path("/app/input")
    output_dir = Path("/app/output")

    # Ensure directories exist
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print("⚠️ No PDF files found in /app/input directory")
        return

    print(f"📁 Found {len(pdf_files)} PDF files to process")

    # Check for persona configuration
    persona_config_file = input_dir / "persona_config.json"
    
    if not persona_config_file.exists():
        print("⚠️ No persona_config.json found in /app/input directory")
        print("Round 1B requires persona configuration to proceed")
        return

    try:
        # Load persona configuration
        with open(persona_config_file, 'r', encoding='utf-8') as f:
            persona_config = json.load(f)

        # Extract display values for persona and job
        persona_display = persona_config.get('persona', 'Unknown')
        job_display = persona_config.get('job_to_be_done', 'Unknown')
        
        # Handle nested format for display
        if isinstance(persona_display, dict) and 'role' in persona_display:
            persona_display = persona_display['role']
        if isinstance(job_display, dict) and 'task' in job_display:
            job_display = job_display['task']

        print(f"🎯 Persona: {persona_display}")
        print(f"🎯 Job to be done: {job_display}")

        # Initialize persona extractor
        persona_extractor = PersonaExtractor()

        print("\n🎭 Extracting persona-relevant content...")
        start_time = time.time()

        # Extract persona-relevant content
        result = persona_extractor.extract(pdf_files, persona_config)

        # Save results
        output_file = output_dir / "persona_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        processing_time = time.time() - start_time
        sections_count = len(result.get("sections", []))
        subsections_count = len(result.get("subsections", []))
        
        print(f"✅ Analysis complete: {sections_count} sections, {subsections_count} subsections in {processing_time:.2f}s")

    except Exception as e:
        print(f"❌ Error in persona extraction: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n🎯 Round 1B Processing Complete!")

if __name__ == "__main__":
    main()
