#!/usr/bin/env python3
"""
Script to generate falsenegativeswithchildren.json and falsepositiveswithchildren.json
using the antithesis prompt to create child objections for each philosophical position.
"""

import json
import openai
import os
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import argparse
import logging

class ChildrenDatasetGenerator:
    """
    Generates evaluation datasets with child objections using the antithesis prompt.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        """
        Initialize the dataset generator.
        
        Args:
            openai_api_key: OpenAI API key for LLM calls
            model: LLM model to use for generating objections
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Load the antithesis prompt
        self.antithesis_prompt = self._load_antithesis_prompt()
    
    def _load_antithesis_prompt(self) -> str:
        """Load the antithesis prompt template."""
        prompt_path = "prompts/antithesis_prompt.txt"
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Antithesis prompt file not found: {prompt_path}")
            raise
    
    def _format_theses_for_prompt(self, theses: List[str]) -> str:
        """Format theses list for the antithesis prompt."""
        formatted_components = []
        for thesis in theses:
            formatted_components.append(f"{{{thesis}}}")
        return ", ".join(formatted_components)
    
    def _parse_antithesis_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract objection components.
        
        Args:
            response_text: Raw response from the LLM
            
        Returns:
            List of objection dictionaries with summary and components
        """
        objections = []
        
        # Split by [START] to get individual objections
        objection_blocks = response_text.split('[START]')[1:]  # Skip first empty part
        
        for block in objection_blocks:
            if '[BREAK]' not in block or '[END]' not in block:
                continue
                
            try:
                # Split by [BREAK] to separate summary and components
                parts = block.split('[BREAK]', 1)
                if len(parts) != 2:
                    continue
                    
                summary_part = parts[0].strip()
                components_part = parts[1].split('[END]')[0].strip()  # Remove [END] and everything after
                
                if not summary_part or not components_part:
                    continue
                
                # Parse components - they should be comma-separated and wrapped in {{}}
                components = []
                
                # Method 1: Try to extract components wrapped in {{}}
                import re
                component_matches = re.findall(r'\{\{([^}]+)\}\}', components_part)
                if component_matches:
                    components = [comp.strip() for comp in component_matches]
                else:
                    # Method 2: Try splitting by }}, {{ pattern
                    if '{{' in components_part and '}}' in components_part:
                        # Remove outer braces if present
                        clean_text = components_part
                        if clean_text.startswith('{{'):
                            clean_text = clean_text[2:]
                        if clean_text.endswith('}}'):
                            clean_text = clean_text[:-2]
                        
                        # Split by }}, {{
                        raw_components = clean_text.split('}}, {{')
                        components = [comp.strip() for comp in raw_components if comp.strip()]
                    else:
                        # Method 3: Fallback - split by commas
                        raw_components = components_part.split(',')
                        components = [comp.strip().strip('{}') for comp in raw_components if comp.strip()]
                
                if summary_part and components:
                    objections.append({
                        'summary': summary_part,
                        'components': components
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error parsing objection block: {e}")
                continue
        
        return objections
    
    def _generate_objections_for_position(self, position: Dict[str, Any], num_objections: int = 100) -> List[Dict[str, Any]]:
        """
        Generate objections for a philosophical position using the antithesis prompt.
        
        Args:
            position: Position dictionary containing theses
            num_objections: Number of objections to generate
            
        Returns:
            List of objection dictionaries
        """
        theses = position.get('theses', [])
        if not theses:
            self.logger.warning(f"No theses found for position {position.get('position_id', 'unknown')}")
            return []
        
        # Format theses for the prompt
        formatted_theses = self._format_theses_for_prompt(theses)
        
        # Create the prompt
        prompt = self.antithesis_prompt.format(
            num_responses=num_objections,
            thesis=formatted_theses
        )
        
        try:
            self.logger.info(f"Generating {num_objections} objections for position {position.get('position_id', 'unknown')}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Some creativity for varied objections
                max_tokens=2000,  # Allow for detailed objections
                timeout=60
            )
            
            response_text = response.choices[0].message.content.strip()
            print("="*50)
            print("RAW LLM RESPONSE:")
            print(response_text)
            print("="*50)
            
            objections = self._parse_antithesis_response(response_text)
            print(f"PARSED {len(objections)} OBJECTIONS:")
            for i, obj in enumerate(objections):
                print(f"  Objection {i+1}: {obj['summary']}")
                print(f"  Components ({len(obj['components'])}): {obj['components'][:2]}...")  # Show first 2 components
            print("="*50)
            
            
            # Format objections as children
            children = []
            for i, objection in enumerate(objections):
                child = {
                    "child_id": f"{position.get('position_id', 'pos')}_{i+1}",
                    "summary": objection['summary'],
                    "components": objection['components']
                }
                children.append(child)
            
            self.logger.info(f"Generated {len(children)} objections for position {position.get('position_id', 'unknown')}")
            return children
            
        except Exception as e:
            self.logger.error(f"Error generating objections for position {position.get('position_id', 'unknown')}: {e}")
            return []
    
    def generate_children_dataset(self, input_file: str, output_file: str, num_objections_per_position: int = 100):
        """
        Generate a dataset with children objections from an existing dataset.
        
        Args:
            input_file: Path to input JSON file (falsenegatives.json or falsepositives.json)
            output_file: Path to output JSON file with children
            num_objections_per_position: Number of objections to generate per position
        """
        try:
            # Load input data
            with open(input_file, 'r') as f:
                input_data = json.load(f)
            
            if 'false_negative_pairs' in input_data:
                pairs_key = 'false_negative_pairs'
            elif 'false_positive_pairs' in input_data:
                pairs_key = 'false_positive_pairs'
            else:
                raise ValueError(f"Unknown data structure in {input_file}")
            
            pairs = input_data[pairs_key]
            output_pairs = []
            
            for pair in pairs:
                self.logger.info(f"Processing pair {pair.get('pair_id', 'unknown')}")
                
                # Copy the original pair structure
                new_pair = {
                    "pair_id": pair["pair_id"],
                    "original_position": pair["original_position"].copy(),
                    "reformulated_position" if pairs_key == 'false_negative_pairs' else "modified_position": 
                        pair["reformulated_position" if pairs_key == 'false_negative_pairs' else "modified_position"].copy()
                }
                
                # Generate children for original position
                original_children = self._generate_objections_for_position(
                    pair["original_position"], 
                    num_objections_per_position
                )
                new_pair["original_position"]["children"] = original_children
                
                # Rate limiting
                time.sleep(1)
                
                # Generate children for comparison position
                comparison_key = "reformulated_position" if pairs_key == 'false_negative_pairs' else "modified_position"
                comparison_children = self._generate_objections_for_position(
                    pair[comparison_key], 
                    num_objections_per_position
                )
                new_pair[comparison_key]["children"] = comparison_children
                
                output_pairs.append(new_pair)
                
                # Rate limiting between pairs
                time.sleep(2)
            
            # Save output data
            output_data = {pairs_key: output_pairs}
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            self.logger.info(f"Generated {len(output_pairs)} pairs with children in {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating children dataset: {e}")
            raise

def main():
    """Main function to generate children datasets."""
    parser = argparse.ArgumentParser(description='Generate evaluation datasets with child objections')
    parser.add_argument('--model', default='gpt-4o', help='LLM model to use (default: gpt-4o)')
    parser.add_argument('--num-objections', type=int, default=100, 
                        help='Number of objections per position (default: 3)')
    parser.add_argument('--input-dir', default='evals', 
                        help='Directory containing input files (default: evals)')
    parser.add_argument('--negatives-only', action='store_true', 
                        help='Generate only false negatives with children')
    parser.add_argument('--positives-only', action='store_true',
                        help='Generate only false positives with children')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY in .env file")
    
    # Initialize generator
    generator = ChildrenDatasetGenerator(OPENAI_API_KEY, args.model)
    
    # Determine which datasets to generate
    generate_negatives = not args.positives_only
    generate_positives = not args.negatives_only
    
    if generate_negatives:
        input_file = os.path.join(args.input_dir, "falsenegatives.json")
        output_file = os.path.join(args.input_dir, "falsenegativeswithchildren.json")
        
        print(f"Generating false negatives with children...")
        generator.generate_children_dataset(
            input_file, 
            output_file, 
            args.num_objections
        )
        print(f"Saved to {output_file}")
    
    if generate_positives:
        input_file = os.path.join(args.input_dir, "falsepositives.json")
        output_file = os.path.join(args.input_dir, "falsepositiveswithchildren.json")
        
        print(f"Generating false positives with children...")
        generator.generate_children_dataset(
            input_file, 
            output_file, 
            args.num_objections
        )
        print(f"Saved to {output_file}")
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()