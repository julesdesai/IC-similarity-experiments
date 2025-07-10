import json
import openai
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
import time

@dataclass
class LLMSimilarity:
    """Stores LLM similarity information for a pair of positions."""
    pair_id: str
    original_position: Dict[str, Any]
    comparison_position: Dict[str, Any]
    llm_score: Optional[int]
    pair_type: str  # 'false_negative' or 'false_positive'

class LLMSimilarityEvaluator:
    """
    Evaluates similarity between philosophical positions using LLM queries instead of embeddings.
    """
    
    def __init__(self, openai_api_key: str, model: str = "o4-mini"):
        """
        Initialize the LLM similarity evaluator.
        
        Args:
            openai_api_key: OpenAI API key for LLM calls
            model: LLM model to use for similarity evaluation
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.false_negatives_data = None
        self.false_positives_data = None
        self.similarity_results: List[LLMSimilarity] = []
        
        # Load the view identity prompt
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load the view identity prompt template."""
        prompt_path = "prompts/view_identity_prompt.txt"
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Prompt file not found: {prompt_path}")
            raise
    
    def load_evaluation_data(self, false_negatives_path: str, false_positives_path: str):
        """
        Load false negatives and false positives data from JSON files.
        
        Args:
            false_negatives_path: Path to false negatives JSON file
            false_positives_path: Path to false positives JSON file
        """
        try:
            with open(false_negatives_path, 'r') as f:
                self.false_negatives_data = json.load(f)
            self.logger.info(f"Loaded {len(self.false_negatives_data['false_negative_pairs'])} false negative pairs")
            
            with open(false_positives_path, 'r') as f:
                self.false_positives_data = json.load(f)
            self.logger.info(f"Loaded {len(self.false_positives_data['false_positive_pairs'])} false positive pairs")
            
        except FileNotFoundError as e:
            self.logger.error(f"Could not find evaluation data files: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON files: {e}")
            raise
    
    def _position_to_text(self, position: Dict[str, Any]) -> str:
        """
        Convert a position dictionary to text for LLM evaluation.
        
        Args:
            position: Position dictionary containing summary and theses
            
        Returns:
            Formatted text representation of the position
        """
        # Use theses as the main content for comparison
        theses = position.get('theses', [])
        return ', '.join(f'"{thesis}"' for thesis in theses)
    
    def _query_llm_similarity(self, position1_text: str, position2_text: str) -> Optional[int]:
        """
        Query LLM to get similarity score between two positions.
        
        Args:
            position1_text: Text representation of first position
            position2_text: Text representation of second position
            
        Returns:
            Similarity score (0-100) or None if failed
        """
        prompt = self.prompt_template.format(
            position_1=position1_text,
            position_2=position2_text
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse the result - should be just a number
            try:
                score = int(result_text)
                if 0 <= score <= 100:
                    return score
                else:
                    self.logger.warning(f"Score out of range: {score}")
                    return None
            except ValueError:
                self.logger.warning(f"Could not parse score: '{result_text}'")
                return None
                
        except Exception as e:
            self.logger.error(f"Error querying {self.model}: {e}")
            return None
    
    def calculate_similarities(self, process_negatives: bool = True, process_positives: bool = True):
        """
        Calculate LLM-based similarities for false negative and/or false positive pairs.
        
        Args:
            process_negatives: Whether to process false negatives
            process_positives: Whether to process false positives
        """
        if not process_negatives and not process_positives:
            raise ValueError("Must process at least one type of pairs")
            
        if process_negatives and self.false_negatives_data is None:
            raise ValueError("False negatives data not loaded")
        if process_positives and self.false_positives_data is None:
            raise ValueError("False positives data not loaded")
        
        self.similarity_results = []
        
        # Process false negatives FIRST
        if process_negatives:
            self.logger.info("=== EXPERIMENT 1: PROCESSING FALSE NEGATIVE PAIRS WITH LLM ===")
            fn_pairs = self.false_negatives_data['false_negative_pairs']
            
            for pair in fn_pairs:
                original_text = self._position_to_text(pair['original_position'])
                reformulated_text = self._position_to_text(pair['reformulated_position'])
                
                # Query LLM
                self.logger.info(f"Querying {self.model} for {pair['pair_id']}...")
                llm_score = self._query_llm_similarity(original_text, reformulated_text)
                time.sleep(0.5)  # Rate limiting
                
                # Store result
                similarity_result = LLMSimilarity(
                    pair_id=pair['pair_id'],
                    original_position=pair['original_position'],
                    comparison_position=pair['reformulated_position'],
                    llm_score=llm_score,
                    pair_type='false_negative'
                )
                
                self.similarity_results.append(similarity_result)
                self.logger.info(f"Processed {pair['pair_id']}: {self.model}={llm_score}")
            
            self.logger.info(f"=== EXPERIMENT 1 COMPLETE: {len(fn_pairs)} false negative pairs processed ===")
        
        # Process false positives SECOND
        if process_positives:
            self.logger.info("=== EXPERIMENT 2: PROCESSING FALSE POSITIVE PAIRS WITH LLM ===")
            fp_pairs = self.false_positives_data['false_positive_pairs']
            
            for pair in fp_pairs:
                original_text = self._position_to_text(pair['original_position'])
                modified_text = self._position_to_text(pair['modified_position'])
                
                # Query LLM
                self.logger.info(f"Querying {self.model} for {pair['pair_id']}...")
                llm_score = self._query_llm_similarity(original_text, modified_text)
                time.sleep(0.5)  # Rate limiting
                
                # Store result
                similarity_result = LLMSimilarity(
                    pair_id=pair['pair_id'],
                    original_position=pair['original_position'],
                    comparison_position=pair['modified_position'],
                    llm_score=llm_score,
                    pair_type='false_positive'
                )
                
                self.similarity_results.append(similarity_result)
                self.logger.info(f"Processed {pair['pair_id']}: {self.model}={llm_score}")
            
            self.logger.info(f"=== EXPERIMENT 2 COMPLETE: {len(fp_pairs)} false positive pairs processed ===")
        
        self.logger.info(f"=== ALL EXPERIMENTS COMPLETE: {len(self.similarity_results)} total pairs processed ===")
    
    def get_similarity_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for the LLM similarity results.
        
        Returns:
            Dictionary containing similarity statistics
        """
        if not self.similarity_results:
            return {"error": "No similarity results available"}
        
        # Separate by type
        fn_scores = [r.llm_score for r in self.similarity_results if r.pair_type == 'false_negative' and r.llm_score is not None]
        fp_scores = [r.llm_score for r in self.similarity_results if r.pair_type == 'false_positive' and r.llm_score is not None]
        all_scores = fn_scores + fp_scores
        
        def calc_stats(scores):
            if not scores:
                return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
            import numpy as np
            return {
                'count': len(scores),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            }
        
        return {
            'overall': calc_stats(all_scores),
            'false_negatives': calc_stats(fn_scores),
            'false_positives': calc_stats(fp_scores)
        }
    
    def export_results(self, output_path: str):
        """
        Export LLM similarity results to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        if not self.similarity_results:
            raise ValueError("No similarity results to export")
        
        # Function to extract numeric part from pair_id
        def extract_pair_number(pair_id: str) -> int:
            try:
                return int(pair_id.split('_')[1])
            except (IndexError, ValueError):
                return 0
        
        # Sort results
        original_order_sorted = sorted(self.similarity_results, key=lambda x: (x.pair_type, extract_pair_number(x.pair_id)))
        
        # Sort by LLM scores (handle None values)
        similarity_sorted = sorted(self.similarity_results, key=lambda x: x.llm_score if x.llm_score is not None else -1, reverse=True)
        
        # Prepare results for export
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_pairs': len(self.similarity_results),
                'false_negatives': len([r for r in self.similarity_results if r.pair_type == 'false_negative']),
                'false_positives': len([r for r in self.similarity_results if r.pair_type == 'false_positive']),
                'evaluation_method': 'LLM_based_similarity'
            },
            'statistics': self.get_similarity_statistics(),
            'results_by_original_order': [],
            'results_by_similarity': []
        }
        
        # Helper function to create result dictionary
        def create_result_dict(result):
            result_dict = {
                'pair_id': result.pair_id,
                'pair_type': result.pair_type,
                'llm_score': result.llm_score,
                'original_position': {
                    'position_id': result.original_position.get('position_id', ''),
                    'summary': result.original_position.get('summary', ''),
                    'theses': result.original_position.get('theses', [])
                },
                'comparison_position': {
                    'position_id': result.comparison_position.get('position_id', ''),
                    'summary': result.comparison_position.get('summary', ''),
                    'theses': result.comparison_position.get('theses', [])
                }
            }
            
            # Add additional fields based on pair type
            if result.pair_type == 'false_negative':
                result_dict['comparison_position']['reformulation_strategies'] = result.comparison_position.get('reformulation_strategies', [])
            elif result.pair_type == 'false_positive':
                result_dict['comparison_position']['modification_type'] = result.comparison_position.get('modification_type', '')
                result_dict['comparison_position']['key_difference'] = result.comparison_position.get('key_difference', '')
            
            return result_dict
        
        # Add results in different orders
        for result in original_order_sorted:
            export_data['results_by_original_order'].append(create_result_dict(result))
        
        for result in similarity_sorted:
            export_data['results_by_similarity'].append(create_result_dict(result))
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Results exported to {output_path}")
    
    def create_similarity_report(self, output_path: str):
        """
        Create a human-readable LLM similarity report.
        
        Args:
            output_path: Path to save the report
        """
        if not self.similarity_results:
            raise ValueError("No similarity results to report")
        
        stats = self.get_similarity_statistics()
        
        # Function to extract numeric part from pair_id
        def extract_pair_number(pair_id: str) -> int:
            try:
                return int(pair_id.split('_')[1])
            except (IndexError, ValueError):
                return 0
        
        # Sort results
        original_order_sorted = sorted(self.similarity_results, key=lambda x: (x.pair_type, extract_pair_number(x.pair_id)))
        similarity_sorted = sorted(self.similarity_results, key=lambda x: x.llm_score if x.llm_score is not None else -1, reverse=True)
        
        report_lines = []
        report_lines.append("=== PHILOSOPHICAL POSITION LLM SIMILARITY ANALYSIS ===")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("=== SUMMARY STATISTICS ===")
        report_lines.append(f"Total pairs analyzed: {len(self.similarity_results)}")
        report_lines.append(f"False negatives: {len([r for r in self.similarity_results if r.pair_type == 'false_negative'])}")
        report_lines.append(f"False positives: {len([r for r in self.similarity_results if r.pair_type == 'false_positive'])}")
        report_lines.append("")
        
        # LLM Statistics
        if stats['overall']['count'] > 0:
            report_lines.append(f"=== {self.model.upper()} SIMILARITY STATISTICS ===")
            report_lines.append(f"Overall mean score: {stats['overall']['mean']:.2f}")
            report_lines.append(f"Overall std: {stats['overall']['std']:.2f}")
            report_lines.append(f"Overall range: {stats['overall']['min']:.0f} - {stats['overall']['max']:.0f}")
            report_lines.append(f"False negatives mean: {stats['false_negatives']['mean']:.2f}")
            report_lines.append(f"False positives mean: {stats['false_positives']['mean']:.2f}")
            report_lines.append("")
        
        # Results in original order
        report_lines.append("=== RESULTS IN ORIGINAL ORDER ===")
        for result in original_order_sorted:
            report_lines.append(f"Pair ID: {result.pair_id} ({result.pair_type})")
            report_lines.append(f"{self.model} Score: {result.llm_score if result.llm_score is not None else 'N/A'}")
            report_lines.append(f"Original: {result.original_position.get('summary', 'N/A')}")
            report_lines.append(f"Comparison: {result.comparison_position.get('summary', 'N/A')}")
            report_lines.append("-" * 80)
        
        # Results sorted by LLM score
        if stats['overall']['count'] > 0:
            report_lines.append("")
            report_lines.append("=== RESULTS SORTED BY SIMILARITY SCORE (HIGHEST TO LOWEST) ===")
            for result in similarity_sorted:
                if result.llm_score is not None:
                    report_lines.append(f"Pair ID: {result.pair_id} ({result.pair_type}) - Score: {result.llm_score}")
                    report_lines.append(f"Original: {result.original_position.get('summary', 'N/A')}")
                    report_lines.append(f"Comparison: {result.comparison_position.get('summary', 'N/A')}")
                    report_lines.append("-" * 80)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        self.logger.info(f"LLM similarity report saved to {output_path}")
    
    def create_markdown_report(self, output_path: str):
        """
        Create a markdown-formatted LLM similarity report.
        
        Args:
            output_path: Path to save the markdown report
        """
        if not self.similarity_results:
            raise ValueError("No similarity results to report")
        
        stats = self.get_similarity_statistics()
        
        # Function to extract numeric part from pair_id
        def extract_pair_number(pair_id: str) -> int:
            try:
                return int(pair_id.split('_')[1])
            except (IndexError, ValueError):
                return 0
        
        # Sort results
        original_order_sorted = sorted(self.similarity_results, key=lambda x: (x.pair_type, extract_pair_number(x.pair_id)))
        similarity_sorted = sorted(self.similarity_results, key=lambda x: x.llm_score if x.llm_score is not None else -1, reverse=True)
        
        markdown_lines = []
        
        # Title and metadata
        markdown_lines.append("# Philosophical Position LLM Similarity Analysis")
        markdown_lines.append("")
        markdown_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_lines.append(f"**Method:** LLM-based similarity evaluation using view identity prompt")
        markdown_lines.append("")
        
        # Table of Contents
        markdown_lines.append("## Table of Contents")
        markdown_lines.append("- [Summary Statistics](#summary-statistics)")
        markdown_lines.append(f"- [{self.model.upper()} Statistics](#{self.model.lower()}-statistics)")
        markdown_lines.append("- [Results by Original Order](#results-by-original-order)")
        markdown_lines.append("- [Results by Similarity Score](#results-by-similarity-score)")
        markdown_lines.append("")
        
        # Summary Statistics
        markdown_lines.append("## Summary Statistics")
        markdown_lines.append("")
        markdown_lines.append(f"- **Total pairs analyzed:** {len(self.similarity_results)}")
        markdown_lines.append(f"- **False negatives:** {len([r for r in self.similarity_results if r.pair_type == 'false_negative'])}")
        markdown_lines.append(f"- **False positives:** {len([r for r in self.similarity_results if r.pair_type == 'false_positive'])}")
        markdown_lines.append("")
        
        # LLM Statistics
        markdown_lines.append(f"## {self.model.upper()} Statistics")
        markdown_lines.append("")
        if stats['overall']['count'] > 0:
            markdown_lines.append("| Category | Mean | Std | Min | Max | Median |")
            markdown_lines.append("|----------|------|-----|-----|-----|--------|")
            markdown_lines.append(f"| Overall | {stats['overall']['mean']:.2f} | {stats['overall']['std']:.2f} | {stats['overall']['min']:.0f} | {stats['overall']['max']:.0f} | {stats['overall']['median']:.2f} |")
            markdown_lines.append(f"| False Negatives | {stats['false_negatives']['mean']:.2f} | {stats['false_negatives']['std']:.2f} | {stats['false_negatives']['min']:.0f} | {stats['false_negatives']['max']:.0f} | {stats['false_negatives']['median']:.2f} |")
            markdown_lines.append(f"| False Positives | {stats['false_positives']['mean']:.2f} | {stats['false_positives']['std']:.2f} | {stats['false_positives']['min']:.0f} | {stats['false_positives']['max']:.0f} | {stats['false_positives']['median']:.2f} |")
        else:
            markdown_lines.append(f"*No {self.model} results available*")
        markdown_lines.append("")
        
        # Results by Original Order (FIRST)
        markdown_lines.append("## Results by Original Order")
        markdown_lines.append("")
        markdown_lines.append("*Ordered by pair ID (FN_001, FN_002, ..., FP_001, FP_002, ...)*")
        markdown_lines.append("")
        
        # Create a table for original order results
        markdown_lines.append("| Pair ID | Type | Score | Original Position | Comparison Position |")
        markdown_lines.append("|---------|------|-------|-------------------|---------------------|")
        
        for result in original_order_sorted:
            original_summary = result.original_position.get('summary', 'N/A')
            comparison_summary = result.comparison_position.get('summary', 'N/A')
            
            # Truncate long summaries for table
            if len(original_summary) > 40:
                original_summary = original_summary[:37] + "..."
            if len(comparison_summary) > 40:
                comparison_summary = comparison_summary[:37] + "..."
            
            score_display = str(result.llm_score) if result.llm_score is not None else "N/A"
            
            markdown_lines.append(f"| {result.pair_id} | {result.pair_type} | {score_display} | {original_summary} | {comparison_summary} |")
        
        markdown_lines.append("")
        
        # Results by Similarity Score (SECOND)
        if stats['overall']['count'] > 0:
            markdown_lines.append("## Results by Similarity Score")
            markdown_lines.append("")
            markdown_lines.append(f"*Sorted by {self.model} similarity score (highest to lowest)*")
            markdown_lines.append("")
            
            for i, result in enumerate([r for r in similarity_sorted if r.llm_score is not None], 1):
                markdown_lines.append(f"### {i}. {result.pair_id} ({result.pair_type}) - Score: {result.llm_score}")
                markdown_lines.append("")
                markdown_lines.append(f"**Original Position:** {result.original_position.get('summary', 'N/A')}")
                markdown_lines.append("")
                markdown_lines.append(f"**Comparison Position:** {result.comparison_position.get('summary', 'N/A')}")
                markdown_lines.append("")
                
                if result.pair_type == 'false_negative':
                    strategies = result.comparison_position.get('reformulation_strategies', [])
                    markdown_lines.append(f"**Reformulation Strategies:** {', '.join(strategies)}")
                elif result.pair_type == 'false_positive':
                    mod_type = result.comparison_position.get('modification_type', 'N/A')
                    key_diff = result.comparison_position.get('key_difference', 'N/A')
                    markdown_lines.append(f"**Modification Type:** {mod_type}")
                    markdown_lines.append("")
                    markdown_lines.append(f"**Key Difference:** {key_diff}")
                
                markdown_lines.append("")
                markdown_lines.append("---")
                markdown_lines.append("")
        
        # Save markdown file
        with open(output_path, 'w') as f:
            f.write("\n".join(markdown_lines))
        
        self.logger.info(f"LLM markdown report saved to {output_path}")

def main():
    """
    Main function to run the LLM similarity evaluation.
    """
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Calculate LLM-based similarity for false negatives and false positives')
    parser.add_argument('--experiment', choices=['negatives', 'positives', 'both'], default='both',
                        help='Which experiment to run (default: both)')
    parser.add_argument('--negatives-only', action='store_true',
                        help='Run only false negatives experiment')
    parser.add_argument('--positives-only', action='store_true',
                        help='Run only false positives experiment')
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY in .env file")
    
    FALSE_NEGATIVES_PATH = "evals/falsenegatives.json"
    FALSE_POSITIVES_PATH = "evals/falsepositives.json"
    
    # Initialize evaluator
    evaluator = LLMSimilarityEvaluator(OPENAI_API_KEY)
    
    # Load evaluation data
    evaluator.load_evaluation_data(FALSE_NEGATIVES_PATH, FALSE_POSITIVES_PATH)
    
    # Determine which experiments to run
    run_negatives = True
    run_positives = True
    
    if args.negatives_only or args.experiment == 'negatives':
        run_negatives = True
        run_positives = False
    elif args.positives_only or args.experiment == 'positives':
        run_negatives = False
        run_positives = True
    elif args.experiment == 'both':
        run_negatives = True
        run_positives = True
    
    # Calculate similarities using LLM (FALSE NEGATIVES FIRST, then FALSE POSITIVES)
    print(f"\n=== STARTING LLM SIMILARITY EXPERIMENTS ===")
    print(f"Running negatives: {run_negatives}")
    print(f"Running positives: {run_positives}")
    print(f"Using model: {evaluator.model}")
    print(f"Execution order: {'Negatives → Positives' if run_negatives and run_positives else 'Negatives only' if run_negatives else 'Positives only'}")
    
    evaluator.calculate_similarities(
        process_negatives=run_negatives, 
        process_positives=run_positives
    )
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_suffix = ""
    if run_negatives and not run_positives:
        experiment_suffix = "_negatives_only"
    elif run_positives and not run_negatives:
        experiment_suffix = "_positives_only"
    
    # Export detailed JSON results
    json_output_path = f"evals/results/llm_similarity_results_{timestamp}{experiment_suffix}.json"
    evaluator.export_results(json_output_path)
    
    # Create human-readable report
    report_output_path = f"evals/results/llm_similarity_report_{timestamp}{experiment_suffix}.txt"
    evaluator.create_similarity_report(report_output_path)
    
    # Create markdown report
    markdown_output_path = f"evals/results/llm_similarity_report_{timestamp}{experiment_suffix}.md"
    evaluator.create_markdown_report(markdown_output_path)
    
    # Print summary
    stats = evaluator.get_similarity_statistics()
    print(f"\n=== LLM SIMILARITY ANALYSIS COMPLETE ===")
    print(f"Total pairs analyzed: {len(evaluator.similarity_results)}")
    
    if stats['overall']['count'] > 0:
        print(f"{evaluator.model} overall mean score: {stats['overall']['mean']:.2f}")
        if run_negatives:
            print(f"{evaluator.model} false negatives mean: {stats['false_negatives']['mean']:.2f}")
        if run_positives:
            print(f"{evaluator.model} false positives mean: {stats['false_positives']['mean']:.2f}")
    
    print(f"\nResults saved to:")
    print(f"  - {json_output_path}")
    print(f"  - {report_output_path}")
    print(f"  - {markdown_output_path}")

if __name__ == "__main__":
    main()