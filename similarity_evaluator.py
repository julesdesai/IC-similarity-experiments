import json
import numpy as np
import openai
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

@dataclass
class PositionSimilarity:
    """Stores similarity information for a pair of positions."""
    pair_id: str
    original_position: Dict[str, Any]
    comparison_position: Dict[str, Any]
    original_embedding: np.ndarray
    comparison_embedding: np.ndarray
    cosine_similarity: float
    pair_type: str  # 'false_negative' or 'false_positive'

class SimilarityEvaluator:
    """
    Evaluates cosine similarity between philosophical positions in false negatives and false positives.
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the similarity evaluator.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.false_negatives_data = None
        self.false_positives_data = None
        self.similarity_results: List[PositionSimilarity] = []
    
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
        Convert a position dictionary to text for embedding.
        
        Args:
            position: Position dictionary containing summary and theses
            
        Returns:
            Formatted text representation of the position
        """
        text_parts = []
        
        # Add summary
        if 'summary' in position:
            text_parts.append(f"Summary: {position['summary']}")
        
        # Add theses
        if 'theses' in position:
            text_parts.append("Theses:")
            for i, thesis in enumerate(position['theses'], 1):
                text_parts.append(f"{i}. {thesis}")
        
        return "\n".join(text_parts)
    
    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts using OpenAI's embedding model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding arrays
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Process in batches to avoid API limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-large", #text-embedding-3-small
                    input=batch_texts
                )
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")
                
            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch starting at {i}: {e}")
                raise
        
        return all_embeddings
    
    def calculate_similarities(self, process_negatives: bool = True, process_positives: bool = True):
        """
        Calculate cosine similarities for false negative and/or false positive pairs.
        
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
        
        # Process false negatives FIRST (as requested)
        if process_negatives:
            self.logger.info("=== EXPERIMENT 1: PROCESSING FALSE NEGATIVE PAIRS ===")
            fn_pairs = self.false_negatives_data['false_negative_pairs']
            
            for pair in fn_pairs:
                original_text = self._position_to_text(pair['original_position'])
                reformulated_text = self._position_to_text(pair['reformulated_position'])
                
                # Get embeddings
                embeddings = self._get_embeddings([original_text, reformulated_text])
                original_embedding = embeddings[0]
                reformulated_embedding = embeddings[1]
                
                # Calculate cosine similarity
                cos_sim = cosine_similarity([original_embedding], [reformulated_embedding])[0, 0]
                
                # Store result
                similarity_result = PositionSimilarity(
                    pair_id=pair['pair_id'],
                    original_position=pair['original_position'],
                    comparison_position=pair['reformulated_position'],
                    original_embedding=original_embedding,
                    comparison_embedding=reformulated_embedding,
                    cosine_similarity=float(cos_sim),
                    pair_type='false_negative'
                )
                
                self.similarity_results.append(similarity_result)
                self.logger.info(f"Processed {pair['pair_id']}: similarity = {cos_sim:.4f}")
            
            self.logger.info(f"=== EXPERIMENT 1 COMPLETE: {len(fn_pairs)} false negative pairs processed ===")
        
        # Process false positives SECOND
        if process_positives:
            self.logger.info("=== EXPERIMENT 2: PROCESSING FALSE POSITIVE PAIRS ===")
            fp_pairs = self.false_positives_data['false_positive_pairs']
            
            for pair in fp_pairs:
                original_text = self._position_to_text(pair['original_position'])
                modified_text = self._position_to_text(pair['modified_position'])
                
                # Get embeddings
                embeddings = self._get_embeddings([original_text, modified_text])
                original_embedding = embeddings[0]
                modified_embedding = embeddings[1]
                
                # Calculate cosine similarity
                cos_sim = cosine_similarity([original_embedding], [modified_embedding])[0, 0]
                
                # Store result
                similarity_result = PositionSimilarity(
                    pair_id=pair['pair_id'],
                    original_position=pair['original_position'],
                    comparison_position=pair['modified_position'],
                    original_embedding=original_embedding,
                    comparison_embedding=modified_embedding,
                    cosine_similarity=float(cos_sim),
                    pair_type='false_positive'
                )
                
                self.similarity_results.append(similarity_result)
                self.logger.info(f"Processed {pair['pair_id']}: similarity = {cos_sim:.4f}")
            
            self.logger.info(f"=== EXPERIMENT 2 COMPLETE: {len(fp_pairs)} false positive pairs processed ===")
        
        self.logger.info(f"=== ALL EXPERIMENTS COMPLETE: {len(self.similarity_results)} total pairs processed ===")
    
    def calculate_false_negative_similarities(self):
        """Run experiment 1: Calculate similarities for false negative pairs only."""
        self.calculate_similarities(process_negatives=True, process_positives=False)
    
    def calculate_false_positive_similarities(self):
        """Run experiment 2: Calculate similarities for false positive pairs only."""
        self.calculate_similarities(process_negatives=False, process_positives=True)
    
    def get_similarity_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for the similarity results.
        
        Returns:
            Dictionary containing similarity statistics
        """
        if not self.similarity_results:
            return {"error": "No similarity results available"}
        
        fn_similarities = [r.cosine_similarity for r in self.similarity_results if r.pair_type == 'false_negative']
        fp_similarities = [r.cosine_similarity for r in self.similarity_results if r.pair_type == 'false_positive']
        all_similarities = [r.cosine_similarity for r in self.similarity_results]
        
        stats = {
            'overall': {
                'count': len(all_similarities),
                'mean': float(np.mean(all_similarities)),
                'std': float(np.std(all_similarities)),
                'min': float(np.min(all_similarities)),
                'max': float(np.max(all_similarities)),
                'median': float(np.median(all_similarities))
            },
            'false_negatives': {
                'count': len(fn_similarities),
                'mean': float(np.mean(fn_similarities)) if fn_similarities else 0.0,
                'std': float(np.std(fn_similarities)) if fn_similarities else 0.0,
                'min': float(np.min(fn_similarities)) if fn_similarities else 0.0,
                'max': float(np.max(fn_similarities)) if fn_similarities else 0.0,
                'median': float(np.median(fn_similarities)) if fn_similarities else 0.0
            },
            'false_positives': {
                'count': len(fp_similarities),
                'mean': float(np.mean(fp_similarities)) if fp_similarities else 0.0,
                'std': float(np.std(fp_similarities)) if fp_similarities else 0.0,
                'min': float(np.min(fp_similarities)) if fp_similarities else 0.0,
                'max': float(np.max(fp_similarities)) if fp_similarities else 0.0,
                'median': float(np.median(fp_similarities)) if fp_similarities else 0.0
            }
        }
        
        return stats
    
    def export_results(self, output_path: str, include_embeddings: bool = False):
        """
        Export similarity results to a JSON file.
        
        Args:
            output_path: Path to save the results
            include_embeddings: Whether to include embedding arrays in output
        """
        if not self.similarity_results:
            raise ValueError("No similarity results to export")
        
        # Function to extract numeric part from pair_id
        def extract_pair_number(pair_id: str) -> int:
            try:
                return int(pair_id.split('_')[1])
            except (IndexError, ValueError):
                return 0
        
        # Sort results in original order
        original_order_sorted = sorted(self.similarity_results, key=lambda x: (x.pair_type, extract_pair_number(x.pair_id)))
        
        # Sort results by similarity
        similarity_sorted = sorted(self.similarity_results, key=lambda x: x.cosine_similarity, reverse=True)
        
        # Prepare results for export
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_pairs': len(self.similarity_results),
                'false_negatives': len([r for r in self.similarity_results if r.pair_type == 'false_negative']),
                'false_positives': len([r for r in self.similarity_results if r.pair_type == 'false_positive'])
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
                'cosine_similarity': result.cosine_similarity,
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
            
            # Include embeddings if requested
            if include_embeddings:
                result_dict['original_embedding'] = result.original_embedding.tolist()
                result_dict['comparison_embedding'] = result.comparison_embedding.tolist()
            
            return result_dict
        
        # Add results in original order
        for result in original_order_sorted:
            export_data['results_by_original_order'].append(create_result_dict(result))
        
        # Add results sorted by similarity
        for result in similarity_sorted:
            export_data['results_by_similarity'].append(create_result_dict(result))
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Results exported to {output_path}")
    
    def create_similarity_report(self, output_path: str):
        """
        Create a human-readable similarity report.
        
        Args:
            output_path: Path to save the report
        """
        if not self.similarity_results:
            raise ValueError("No similarity results to report")
        
        stats = self.get_similarity_statistics()
        
        report_lines = []
        report_lines.append("=== PHILOSOPHICAL POSITION SIMILARITY ANALYSIS ===")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("=== SUMMARY STATISTICS ===")
        report_lines.append(f"Total pairs analyzed: {stats['overall']['count']}")
        report_lines.append(f"False negatives: {stats['false_negatives']['count']}")
        report_lines.append(f"False positives: {stats['false_positives']['count']}")
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("=== OVERALL SIMILARITY STATISTICS ===")
        report_lines.append(f"Mean similarity: {stats['overall']['mean']:.4f}")
        report_lines.append(f"Standard deviation: {stats['overall']['std']:.4f}")
        report_lines.append(f"Minimum similarity: {stats['overall']['min']:.4f}")
        report_lines.append(f"Maximum similarity: {stats['overall']['max']:.4f}")
        report_lines.append(f"Median similarity: {stats['overall']['median']:.4f}")
        report_lines.append("")
        
        # False negatives statistics
        report_lines.append("=== FALSE NEGATIVES SIMILARITY STATISTICS ===")
        report_lines.append(f"Mean similarity: {stats['false_negatives']['mean']:.4f}")
        report_lines.append(f"Standard deviation: {stats['false_negatives']['std']:.4f}")
        report_lines.append(f"Minimum similarity: {stats['false_negatives']['min']:.4f}")
        report_lines.append(f"Maximum similarity: {stats['false_negatives']['max']:.4f}")
        report_lines.append(f"Median similarity: {stats['false_negatives']['median']:.4f}")
        report_lines.append("")
        
        # False positives statistics
        report_lines.append("=== FALSE POSITIVES SIMILARITY STATISTICS ===")
        report_lines.append(f"Mean similarity: {stats['false_positives']['mean']:.4f}")
        report_lines.append(f"Standard deviation: {stats['false_positives']['std']:.4f}")
        report_lines.append(f"Minimum similarity: {stats['false_positives']['min']:.4f}")
        report_lines.append(f"Maximum similarity: {stats['false_positives']['max']:.4f}")
        report_lines.append(f"Median similarity: {stats['false_positives']['median']:.4f}")
        report_lines.append("")
        
        # Detailed results
        report_lines.append("=== DETAILED RESULTS ===")
        report_lines.append("")
        
        # Function to extract numeric part from pair_id (e.g., "fn_001" -> 1, "fp_026" -> 26)
        def extract_pair_number(pair_id: str) -> int:
            try:
                return int(pair_id.split('_')[1])
            except (IndexError, ValueError):
                return 0
        
        # Sort by similarity score (descending)
        similarity_sorted = sorted(self.similarity_results, key=lambda x: x.cosine_similarity, reverse=True)
        
        # Sort by original order (pair ID number)
        original_order_sorted = sorted(self.similarity_results, key=lambda x: (x.pair_type, extract_pair_number(x.pair_id)))
        
        # Results sorted by similarity
        report_lines.append("--- RESULTS SORTED BY SIMILARITY (HIGHEST TO LOWEST) ---")
        for result in similarity_sorted:
            report_lines.append(f"Pair ID: {result.pair_id} ({result.pair_type})")
            report_lines.append(f"Cosine Similarity: {result.cosine_similarity:.4f}")
            report_lines.append(f"Original: {result.original_position.get('summary', 'N/A')}")
            report_lines.append(f"Comparison: {result.comparison_position.get('summary', 'N/A')}")
            
            if result.pair_type == 'false_negative':
                strategies = result.comparison_position.get('reformulation_strategies', [])
                report_lines.append(f"Reformulation Strategies: {', '.join(strategies)}")
            elif result.pair_type == 'false_positive':
                mod_type = result.comparison_position.get('modification_type', 'N/A')
                key_diff = result.comparison_position.get('key_difference', 'N/A')
                report_lines.append(f"Modification Type: {mod_type}")
                report_lines.append(f"Key Difference: {key_diff}")
            
            report_lines.append("-" * 80)
        
        report_lines.append("")
        report_lines.append("--- RESULTS IN ORIGINAL ORDER (FN_001, FN_002, ..., FP_001, FP_002, ...) ---")
        
        for result in original_order_sorted:
            report_lines.append(f"Pair ID: {result.pair_id} ({result.pair_type})")
            report_lines.append(f"Cosine Similarity: {result.cosine_similarity:.4f}")
            report_lines.append(f"Original: {result.original_position.get('summary', 'N/A')}")
            report_lines.append(f"Comparison: {result.comparison_position.get('summary', 'N/A')}")
            
            if result.pair_type == 'false_negative':
                strategies = result.comparison_position.get('reformulation_strategies', [])
                report_lines.append(f"Reformulation Strategies: {', '.join(strategies)}")
            elif result.pair_type == 'false_positive':
                mod_type = result.comparison_position.get('modification_type', 'N/A')
                key_diff = result.comparison_position.get('key_difference', 'N/A')
                report_lines.append(f"Modification Type: {mod_type}")
                report_lines.append(f"Key Difference: {key_diff}")
            
            report_lines.append("-" * 80)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        self.logger.info(f"Similarity report saved to {output_path}")
    
    def create_markdown_report(self, output_path: str):
        """
        Create a markdown-formatted similarity report.
        
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
        similarity_sorted = sorted(self.similarity_results, key=lambda x: x.cosine_similarity, reverse=True)
        original_order_sorted = sorted(self.similarity_results, key=lambda x: (x.pair_type, extract_pair_number(x.pair_id)))
        
        markdown_lines = []
        
        # Title and metadata
        markdown_lines.append("# Philosophical Position Similarity Analysis")
        markdown_lines.append("")
        markdown_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_lines.append("")
        
        # Table of Contents
        markdown_lines.append("## Table of Contents")
        markdown_lines.append("- [Summary Statistics](#summary-statistics)")
        markdown_lines.append("- [Overall Statistics](#overall-statistics)")
        markdown_lines.append("- [False Negatives Statistics](#false-negatives-statistics)")
        markdown_lines.append("- [False Positives Statistics](#false-positives-statistics)")
        markdown_lines.append("- [Results by Original Order](#results-by-original-order)")
        markdown_lines.append("- [Results by Similarity](#results-by-similarity)")
        markdown_lines.append("")
        
        # Summary Statistics
        markdown_lines.append("## Summary Statistics")
        markdown_lines.append("")
        markdown_lines.append(f"- **Total pairs analyzed:** {stats['overall']['count']}")
        markdown_lines.append(f"- **False negatives:** {stats['false_negatives']['count']}")
        markdown_lines.append(f"- **False positives:** {stats['false_positives']['count']}")
        markdown_lines.append("")
        
        # Overall Statistics Table
        markdown_lines.append("## Overall Statistics")
        markdown_lines.append("")
        markdown_lines.append("| Metric | Value |")
        markdown_lines.append("|--------|-------|")
        markdown_lines.append(f"| Mean similarity | {stats['overall']['mean']:.4f} |")
        markdown_lines.append(f"| Standard deviation | {stats['overall']['std']:.4f} |")
        markdown_lines.append(f"| Minimum similarity | {stats['overall']['min']:.4f} |")
        markdown_lines.append(f"| Maximum similarity | {stats['overall']['max']:.4f} |")
        markdown_lines.append(f"| Median similarity | {stats['overall']['median']:.4f} |")
        markdown_lines.append("")
        
        # False Negatives Statistics
        markdown_lines.append("## False Negatives Statistics")
        markdown_lines.append("")
        if stats['false_negatives']['count'] > 0:
            markdown_lines.append("| Metric | Value |")
            markdown_lines.append("|--------|-------|")
            markdown_lines.append(f"| Count | {stats['false_negatives']['count']} |")
            markdown_lines.append(f"| Mean similarity | {stats['false_negatives']['mean']:.4f} |")
            markdown_lines.append(f"| Standard deviation | {stats['false_negatives']['std']:.4f} |")
            markdown_lines.append(f"| Minimum similarity | {stats['false_negatives']['min']:.4f} |")
            markdown_lines.append(f"| Maximum similarity | {stats['false_negatives']['max']:.4f} |")
            markdown_lines.append(f"| Median similarity | {stats['false_negatives']['median']:.4f} |")
        else:
            markdown_lines.append("*No false negatives processed*")
        markdown_lines.append("")
        
        # False Positives Statistics
        markdown_lines.append("## False Positives Statistics")
        markdown_lines.append("")
        if stats['false_positives']['count'] > 0:
            markdown_lines.append("| Metric | Value |")
            markdown_lines.append("|--------|-------|")
            markdown_lines.append(f"| Count | {stats['false_positives']['count']} |")
            markdown_lines.append(f"| Mean similarity | {stats['false_positives']['mean']:.4f} |")
            markdown_lines.append(f"| Standard deviation | {stats['false_positives']['std']:.4f} |")
            markdown_lines.append(f"| Minimum similarity | {stats['false_positives']['min']:.4f} |")
            markdown_lines.append(f"| Maximum similarity | {stats['false_positives']['max']:.4f} |")
            markdown_lines.append(f"| Median similarity | {stats['false_positives']['median']:.4f} |")
        else:
            markdown_lines.append("*No false positives processed*")
        markdown_lines.append("")
        
        # Results by Original Order (FIRST)
        markdown_lines.append("## Results by Original Order")
        markdown_lines.append("")
        markdown_lines.append("*Ordered by pair ID (FN_001, FN_002, ..., FP_001, FP_002, ...)*")
        markdown_lines.append("")
        
        # Create a table for original order results
        markdown_lines.append("| Pair ID | Type | Similarity | Original Position | Comparison Position |")
        markdown_lines.append("|---------|------|------------|-------------------|---------------------|")
        
        for result in original_order_sorted:
            original_summary = result.original_position.get('summary', 'N/A')
            comparison_summary = result.comparison_position.get('summary', 'N/A')
            
            # Truncate long summaries for table
            if len(original_summary) > 50:
                original_summary = original_summary[:47] + "..."
            if len(comparison_summary) > 50:
                comparison_summary = comparison_summary[:47] + "..."
            
            markdown_lines.append(f"| {result.pair_id} | {result.pair_type} | {result.cosine_similarity:.4f} | {original_summary} | {comparison_summary} |")
        
        markdown_lines.append("")
        
        # Detailed breakdown by original order
        markdown_lines.append("### Detailed Breakdown by Original Order")
        markdown_lines.append("")
        
        current_type = None
        for result in original_order_sorted:
            # Add section headers when type changes
            if result.pair_type != current_type:
                current_type = result.pair_type
                type_title = "False Negatives" if current_type == 'false_negative' else "False Positives"
                markdown_lines.append(f"#### {type_title}")
                markdown_lines.append("")
            
            markdown_lines.append(f"**{result.pair_id}** | Similarity: {result.cosine_similarity:.4f}")
            markdown_lines.append("")
            markdown_lines.append(f"- **Original:** {result.original_position.get('summary', 'N/A')}")
            markdown_lines.append(f"- **Comparison:** {result.comparison_position.get('summary', 'N/A')}")
            
            if result.pair_type == 'false_negative':
                strategies = result.comparison_position.get('reformulation_strategies', [])
                if strategies:
                    markdown_lines.append(f"- **Strategies:** {', '.join(strategies)}")
            elif result.pair_type == 'false_positive':
                mod_type = result.comparison_position.get('modification_type', 'N/A')
                if mod_type != 'N/A':
                    markdown_lines.append(f"- **Modification:** {mod_type}")
            
            markdown_lines.append("")
        
        # Results by Similarity (SECOND)
        markdown_lines.append("## Results by Similarity")
        markdown_lines.append("")
        markdown_lines.append("*Sorted by cosine similarity (highest to lowest)*")
        markdown_lines.append("")
        
        for i, result in enumerate(similarity_sorted, 1):
            markdown_lines.append(f"### {i}. {result.pair_id} ({result.pair_type})")
            markdown_lines.append("")
            markdown_lines.append(f"**Cosine Similarity:** {result.cosine_similarity:.4f}")
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
        
        self.logger.info(f"Markdown report saved to {output_path}")

def main():
    """
    Main function to run the similarity evaluation.
    """
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Calculate cosine similarity for false negatives and false positives')
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
    evaluator = SimilarityEvaluator(OPENAI_API_KEY)
    
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
    
    # Calculate similarities (FALSE NEGATIVES FIRST, then FALSE POSITIVES)
    print(f"\n=== STARTING SIMILARITY EXPERIMENTS ===")
    print(f"Running negatives: {run_negatives}")
    print(f"Running positives: {run_positives}")
    print(f"Execution order: {'Negatives → Positives' if run_negatives and run_positives else 'Negatives only' if run_negatives else 'Positives only'}")
    
    evaluator.calculate_similarities(process_negatives=run_negatives, process_positives=run_positives)
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_suffix = ""
    if run_negatives and not run_positives:
        experiment_suffix = "_negatives_only"
    elif run_positives and not run_negatives:
        experiment_suffix = "_positives_only"
    
    # Export detailed JSON results
    json_output_path = f"evals/results/similarity_results_{timestamp}{experiment_suffix}.json"
    evaluator.export_results(json_output_path, include_embeddings=False)
    
    # Create human-readable report
    report_output_path = f"evals/results/similarity_report_{timestamp}{experiment_suffix}.txt"
    evaluator.create_similarity_report(report_output_path)
    
    # Create markdown report
    markdown_output_path = f"evals/results/similarity_report_{timestamp}{experiment_suffix}.md"
    evaluator.create_markdown_report(markdown_output_path)
    
    # Print summary
    stats = evaluator.get_similarity_statistics()
    print(f"\n=== SIMILARITY ANALYSIS COMPLETE ===")
    print(f"Total pairs analyzed: {stats['overall']['count']}")
    print(f"Overall mean similarity: {stats['overall']['mean']:.4f}")
    if run_negatives:
        print(f"False negatives mean similarity: {stats['false_negatives']['mean']:.4f}")
    if run_positives:
        print(f"False positives mean similarity: {stats['false_positives']['mean']:.4f}")
    print(f"\nResults saved to:")
    print(f"  - {json_output_path}")
    print(f"  - {report_output_path}")
    print(f"  - {markdown_output_path}")

if __name__ == "__main__":
    main()