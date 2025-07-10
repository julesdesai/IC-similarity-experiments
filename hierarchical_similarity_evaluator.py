#!/usr/bin/env python3
"""
Hierarchical similarity evaluator for philosophical positions with child objections.

This module implements a sophisticated similarity measure that compares both the main 
philosophical positions and their associated objections using the Hungarian algorithm 
for optimal objection matching.
"""

import json
import numpy as np
import openai
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import logging
from dataclasses import dataclass
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
import time

@dataclass
class HierarchicalSimilarity:
    """Stores hierarchical similarity information for a pair of positions."""
    pair_id: str
    original_position: Dict[str, Any]
    comparison_position: Dict[str, Any]
    node_similarity: float
    children_similarity: float
    hierarchical_similarity: float
    objection_matches: List[Tuple[int, int, float]]  # (orig_idx, comp_idx, similarity)
    pair_type: str  # 'false_negative' or 'false_positive'

class HierarchicalSimilarityEvaluator:
    """
    Evaluates hierarchical similarity between philosophical positions and their objections.
    
    This evaluator computes similarity at two levels:
    1. Node-level: Cosine similarity between the main philosophical theses
    2. Children-level: Hungarian algorithm optimal matching between objections
    
    The Hungarian algorithm finds the best 1:1 correspondence between objections,
    treating each objection as a distinct conceptual challenge that should map
    to its closest counterpart in the comparison position.
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the hierarchical similarity evaluator.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.false_negatives_data = None
        self.false_positives_data = None
        self.similarity_results: List[HierarchicalSimilarity] = []
    
    def load_evaluation_data(self, false_negatives_path: str, false_positives_path: str):
        """
        Load false negatives and false positives data with children from JSON files.
        
        Args:
            false_negatives_path: Path to falsenegativeswithchildren.json
            false_positives_path: Path to falsepositiveswithchildren.json
        """
        try:
            with open(false_negatives_path, 'r') as f:
                self.false_negatives_data = json.load(f)
            self.logger.info(f"Loaded {len(self.false_negatives_data['false_negative_pairs'])} false negative pairs with children")
            
            with open(false_positives_path, 'r') as f:
                self.false_positives_data = json.load(f)
            self.logger.info(f"Loaded {len(self.false_positives_data['false_positive_pairs'])} false positive pairs with children")
            
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
        
        return "\\n".join(text_parts)
    
    def _child_to_text(self, child: Dict[str, Any]) -> str:
        """
        Convert a child objection to text for embedding.
        
        Args:
            child: Child dictionary containing summary and components
            
        Returns:
            Formatted text representation of the objection
        """
        text_parts = []
        
        # Add summary
        if 'summary' in child:
            text_parts.append(f"Objection: {child['summary']}")
        
        # Add components
        if 'components' in child:
            text_parts.append("Components:")
            for i, component in enumerate(child['components'], 1):
                text_parts.append(f"{i}. {component}")
        
        return "\\n".join(text_parts)
    
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
                    model="text-embedding-3-small",
                    input=batch_texts
                )
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")
                
            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch starting at {i}: {e}")
                raise
        
        return all_embeddings
    
    def _hungarian_children_similarity(self, children_A: List[Dict[str, Any]], 
                                     children_B: List[Dict[str, Any]]) -> Tuple[float, List[Tuple[int, int, float]]]:
        """
        Compute similarity between two sets of children using the Hungarian algorithm.
        
        The Hungarian algorithm solves the assignment problem: given two sets of objections,
        find the optimal 1:1 matching that maximizes total similarity. This is conceptually
        appropriate because:
        
        1. Each objection represents a distinct conceptual challenge
        2. If two positions are equivalent, their objections should correspond
        3. We want to find which objections are "the same objection" across positions
        
        Algorithm steps:
        1. Create a similarity matrix (m×n) between all objection pairs
        2. Pad to square matrix if needed (for unequal objection counts)
        3. Apply Hungarian algorithm to find optimal assignment
        4. Return average similarity of matched pairs
        
        Args:
            children_A: List of objection dictionaries from first position
            children_B: List of objection dictionaries from second position
            
        Returns:
            Tuple of (average_similarity, list_of_matches)
            where matches are (index_A, index_B, similarity_score)
        """
        # Handle edge cases
        if not children_A and not children_B:
            self.logger.debug("Both positions have no objections - perfect match")
            return 1.0, []
        
        if not children_A or not children_B:
            self.logger.debug("One position has no objections - asymmetric objection profiles")
            return 0.0, []
        
        # Convert children to text and get embeddings
        texts_A = [self._child_to_text(child) for child in children_A]
        texts_B = [self._child_to_text(child) for child in children_B]
        
        all_texts = texts_A + texts_B
        all_embeddings = self._get_embeddings(all_texts)
        
        embeddings_A = all_embeddings[:len(texts_A)]
        embeddings_B = all_embeddings[len(texts_A):]
        
        # Step 1: Create similarity matrix (m×n)
        # Each entry [i,j] = similarity between objection i from A and objection j from B
        m, n = len(children_A), len(children_B)
        sim_matrix = np.zeros((m, n))
        
        for i in range(m):
            for j in range(n):
                # Cosine similarity between embeddings
                sim = cosine_similarity([embeddings_A[i]], [embeddings_B[j]])[0, 0]
                sim_matrix[i, j] = sim
        
        self.logger.debug(f"Created {m}×{n} similarity matrix")
        self.logger.debug(f"Similarity range: {sim_matrix.min():.3f} to {sim_matrix.max():.3f}")
        
        # Step 2: Handle rectangular matrix by padding to square
        # The Hungarian algorithm requires a square matrix, so we pad with neutral values
        max_size = max(m, n)
        square_matrix = np.full((max_size, max_size), 0.5)  # Neutral similarity for dummy entries
        
        # Copy real similarities into the square matrix
        square_matrix[:m, :n] = sim_matrix
        
        # Step 3: Apply Hungarian algorithm
        # We negate similarities because linear_sum_assignment minimizes cost,
        # but we want to maximize similarity
        cost_matrix = -square_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Step 4: Extract real matches (not involving dummy entries)
        real_matches = []
        matched_similarities = []
        
        for row_idx, col_idx in zip(row_indices, col_indices):
            if row_idx < m and col_idx < n:  # Both indices are real (not dummy)
                similarity = sim_matrix[row_idx, col_idx]
                real_matches.append((row_idx, col_idx, similarity))
                matched_similarities.append(similarity)
        
        # Step 5: Compute average similarity
        if matched_similarities:
            avg_similarity = np.mean(matched_similarities)
            self.logger.debug(f"Hungarian algorithm found {len(real_matches)} optimal matches")
            self.logger.debug(f"Match similarities: {[f'{s:.3f}' for s in matched_similarities]}")
        else:
            avg_similarity = 0.5  # No real matches possible
            self.logger.debug("No real matches found between objection sets")
        
        return float(avg_similarity), real_matches
    
    def _compute_node_similarity(self, position_A: Dict[str, Any], position_B: Dict[str, Any]) -> float:
        """
        Compute cosine similarity between two philosophical positions.
        
        Args:
            position_A: First position dictionary
            position_B: Second position dictionary
            
        Returns:
            Cosine similarity between position embeddings
        """
        text_A = self._position_to_text(position_A)
        text_B = self._position_to_text(position_B)
        
        embeddings = self._get_embeddings([text_A, text_B])
        
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0]
        return float(similarity)
    
    def calculate_hierarchical_similarities(self, 
                                          process_negatives: bool = True, 
                                          process_positives: bool = True,
                                          alpha: float = 0.6):
        """
        Calculate hierarchical similarities for false negative and/or false positive pairs.
        
        The hierarchical similarity combines:
        1. Node similarity: Cosine similarity between main philosophical positions
        2. Children similarity: Hungarian algorithm matching between objections
        
        Formula: hierarchical_similarity = α × node_similarity + (1-α) × children_similarity
        
        Args:
            process_negatives: Whether to process false negatives
            process_positives: Whether to process false positives  
            alpha: Weight for node similarity vs children similarity (0 to 1)
                  α=1 means only node similarity, α=0 means only children similarity
        """
        if not process_negatives and not process_positives:
            raise ValueError("Must process at least one type of pairs")
            
        if process_negatives and self.false_negatives_data is None:
            raise ValueError("False negatives data not loaded")
        if process_positives and self.false_positives_data is None:
            raise ValueError("False positives data not loaded")
        
        self.similarity_results = []
        
        # Process false negatives first
        if process_negatives:
            self.logger.info("=== PROCESSING FALSE NEGATIVE PAIRS WITH HIERARCHICAL SIMILARITY ===")
            fn_pairs = self.false_negatives_data['false_negative_pairs']
            
            for pair in fn_pairs:
                self.logger.info(f"Processing {pair['pair_id']}...")
                
                original_pos = pair['original_position']
                reformulated_pos = pair['reformulated_position']
                
                # Compute node-level similarity
                node_sim = self._compute_node_similarity(original_pos, reformulated_pos)
                
                # Compute children-level similarity using Hungarian algorithm
                children_sim, objection_matches = self._hungarian_children_similarity(
                    original_pos.get('children', []),
                    reformulated_pos.get('children', [])
                )
                
                # Compute hierarchical similarity
                hierarchical_sim = alpha * node_sim + (1 - alpha) * children_sim
                
                # Store result
                result = HierarchicalSimilarity(
                    pair_id=pair['pair_id'],
                    original_position=original_pos,
                    comparison_position=reformulated_pos,
                    node_similarity=node_sim,
                    children_similarity=children_sim,
                    hierarchical_similarity=hierarchical_sim,
                    objection_matches=objection_matches,
                    pair_type='false_negative'
                )
                
                self.similarity_results.append(result)
                
                self.logger.info(f"  Node similarity: {node_sim:.4f}")
                self.logger.info(f"  Children similarity: {children_sim:.4f}")
                self.logger.info(f"  Hierarchical similarity: {hierarchical_sim:.4f}")
                
                # Rate limiting
                time.sleep(1)
        
        # Process false positives second
        if process_positives:
            self.logger.info("=== PROCESSING FALSE POSITIVE PAIRS WITH HIERARCHICAL SIMILARITY ===")
            fp_pairs = self.false_positives_data['false_positive_pairs']
            
            for pair in fp_pairs:
                self.logger.info(f"Processing {pair['pair_id']}...")
                
                original_pos = pair['original_position']
                modified_pos = pair['modified_position']
                
                # Compute node-level similarity
                node_sim = self._compute_node_similarity(original_pos, modified_pos)
                
                # Compute children-level similarity using Hungarian algorithm
                children_sim, objection_matches = self._hungarian_children_similarity(
                    original_pos.get('children', []),
                    modified_pos.get('children', [])
                )
                
                # Compute hierarchical similarity
                hierarchical_sim = alpha * node_sim + (1 - alpha) * children_sim
                
                # Store result
                result = HierarchicalSimilarity(
                    pair_id=pair['pair_id'],
                    original_position=original_pos,
                    comparison_position=modified_pos,
                    node_similarity=node_sim,
                    children_similarity=children_sim,
                    hierarchical_similarity=hierarchical_sim,
                    objection_matches=objection_matches,
                    pair_type='false_positive'
                )
                
                self.similarity_results.append(result)
                
                self.logger.info(f"  Node similarity: {node_sim:.4f}")
                self.logger.info(f"  Children similarity: {children_sim:.4f}")
                self.logger.info(f"  Hierarchical similarity: {hierarchical_sim:.4f}")
                
                # Rate limiting
                time.sleep(1)
        
        self.logger.info(f"=== HIERARCHICAL SIMILARITY ANALYSIS COMPLETE: {len(self.similarity_results)} pairs processed ===")
    
    def get_similarity_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for the hierarchical similarity results.
        
        Returns:
            Dictionary containing detailed similarity statistics
        """
        if not self.similarity_results:
            return {"error": "No similarity results available"}
        
        # Separate by type
        fn_results = [r for r in self.similarity_results if r.pair_type == 'false_negative']
        fp_results = [r for r in self.similarity_results if r.pair_type == 'false_positive']
        
        def calc_stats(results, metric):
            if not results:
                return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
            
            values = [getattr(r, metric) for r in results]
            return {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return {
            'overall': {
                'node_similarity': calc_stats(self.similarity_results, 'node_similarity'),
                'children_similarity': calc_stats(self.similarity_results, 'children_similarity'),
                'hierarchical_similarity': calc_stats(self.similarity_results, 'hierarchical_similarity')
            },
            'false_negatives': {
                'node_similarity': calc_stats(fn_results, 'node_similarity'),
                'children_similarity': calc_stats(fn_results, 'children_similarity'),
                'hierarchical_similarity': calc_stats(fn_results, 'hierarchical_similarity')
            },
            'false_positives': {
                'node_similarity': calc_stats(fp_results, 'node_similarity'),
                'children_similarity': calc_stats(fp_results, 'children_similarity'),
                'hierarchical_similarity': calc_stats(fp_results, 'hierarchical_similarity')
            }
        }
    
    def export_results(self, output_path: str):
        """
        Export hierarchical similarity results to a JSON file.
        
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
        original_order_sorted = sorted(self.similarity_results, 
                                     key=lambda x: (x.pair_type, extract_pair_number(x.pair_id)))
        hierarchical_sorted = sorted(self.similarity_results, 
                                   key=lambda x: x.hierarchical_similarity, reverse=True)
        
        # Prepare results for export
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_pairs': len(self.similarity_results),
                'false_negatives': len([r for r in self.similarity_results if r.pair_type == 'false_negative']),
                'false_positives': len([r for r in self.similarity_results if r.pair_type == 'false_positive']),
                'evaluation_method': 'hierarchical_similarity_with_hungarian_algorithm'
            },
            'statistics': self.get_similarity_statistics(),
            'results_by_original_order': [],
            'results_by_hierarchical_similarity': []
        }
        
        # Helper function to create result dictionary
        def create_result_dict(result):
            result_dict = {
                'pair_id': result.pair_id,
                'pair_type': result.pair_type,
                'node_similarity': result.node_similarity,
                'children_similarity': result.children_similarity,
                'hierarchical_similarity': result.hierarchical_similarity,
                'objection_matches': [
                    {
                        'original_objection_index': match[0],
                        'comparison_objection_index': match[1],
                        'match_similarity': match[2]
                    }
                    for match in result.objection_matches
                ],
                'original_position': {
                    'position_id': result.original_position.get('position_id', ''),
                    'summary': result.original_position.get('summary', ''),
                    'theses': result.original_position.get('theses', []),
                    'children_count': len(result.original_position.get('children', []))
                },
                'comparison_position': {
                    'position_id': result.comparison_position.get('position_id', ''),
                    'summary': result.comparison_position.get('summary', ''),
                    'theses': result.comparison_position.get('theses', []),
                    'children_count': len(result.comparison_position.get('children', []))
                }
            }
            
            # Add type-specific fields
            if result.pair_type == 'false_negative':
                result_dict['comparison_position']['reformulation_strategies'] = result.comparison_position.get('reformulation_strategies', [])
            elif result.pair_type == 'false_positive':
                result_dict['comparison_position']['modification_type'] = result.comparison_position.get('modification_type', '')
                result_dict['comparison_position']['key_difference'] = result.comparison_position.get('key_difference', '')
            
            return result_dict
        
        # Add results in different orders
        for result in original_order_sorted:
            export_data['results_by_original_order'].append(create_result_dict(result))
        
        for result in hierarchical_sorted:
            export_data['results_by_hierarchical_similarity'].append(create_result_dict(result))
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=self._json_serializer)
        
        self.logger.info(f"Hierarchical similarity results exported to {output_path}")

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    def create_markdown_report(self, output_path: str):
        """Create a markdown report with the hierarchical similarity results."""
        # Sort results by original order and hierarchical similarity
        original_order_sorted = sorted(self.similarity_results, key=lambda x: x.pair_id)
        hierarchical_sorted = sorted(self.similarity_results, key=lambda x: x.hierarchical_similarity, reverse=True)
        
        markdown_lines = []
        
        # Title and overview
        markdown_lines.append("# Hierarchical Similarity Analysis Report")
        markdown_lines.append("")
        markdown_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_lines.append("")
        
        # Statistics
        stats = self.get_similarity_statistics()
        
        markdown_lines.append("## Summary Statistics")
        markdown_lines.append("")
        
        # Overall statistics
        markdown_lines.append("### Overall Results")
        markdown_lines.append("")
        markdown_lines.append(f"- **Total Pairs:** {stats['overall']['hierarchical_similarity']['count']}")
        markdown_lines.append(f"- **Average Hierarchical Similarity:** {stats['overall']['hierarchical_similarity']['mean']:.4f}")
        markdown_lines.append(f"- **Average Node Similarity:** {stats['overall']['node_similarity']['mean']:.4f}")
        markdown_lines.append(f"- **Average Children Similarity:** {stats['overall']['children_similarity']['mean']:.4f}")
        markdown_lines.append("")
        
        # False negatives statistics
        if stats['false_negatives']['hierarchical_similarity']['count'] > 0:
            markdown_lines.append("### False Negatives")
            markdown_lines.append("")
            markdown_lines.append(f"- **Count:** {stats['false_negatives']['hierarchical_similarity']['count']}")
            markdown_lines.append(f"- **Average Hierarchical Similarity:** {stats['false_negatives']['hierarchical_similarity']['mean']:.4f}")
            markdown_lines.append(f"- **Average Node Similarity:** {stats['false_negatives']['node_similarity']['mean']:.4f}")
            markdown_lines.append(f"- **Average Children Similarity:** {stats['false_negatives']['children_similarity']['mean']:.4f}")
            markdown_lines.append("")
        
        # False positives statistics
        if stats['false_positives']['hierarchical_similarity']['count'] > 0:
            markdown_lines.append("### False Positives")
            markdown_lines.append("")
            markdown_lines.append(f"- **Count:** {stats['false_positives']['hierarchical_similarity']['count']}")
            markdown_lines.append(f"- **Average Hierarchical Similarity:** {stats['false_positives']['hierarchical_similarity']['mean']:.4f}")
            markdown_lines.append(f"- **Average Node Similarity:** {stats['false_positives']['node_similarity']['mean']:.4f}")
            markdown_lines.append(f"- **Average Children Similarity:** {stats['false_positives']['children_similarity']['mean']:.4f}")
            markdown_lines.append("")
        
        markdown_lines.append("---")
        markdown_lines.append("")
        
        # Summary table
        markdown_lines.append("## Results Summary")
        markdown_lines.append("")
        markdown_lines.append("| Pair ID | Type | Hierarchical Sim | Node Sim | Children Sim | Original Position | Comparison Position |")
        markdown_lines.append("|---------|------|------------------|----------|--------------|-------------------|---------------------|")
        
        for result in original_order_sorted:
            original_summary = result.original_position.get('summary', 'N/A')
            comparison_summary = result.comparison_position.get('summary', 'N/A')
            
            # Truncate long summaries for table
            if len(original_summary) > 40:
                original_summary = original_summary[:37] + "..."
            if len(comparison_summary) > 40:
                comparison_summary = comparison_summary[:37] + "..."
            
            markdown_lines.append(f"| {result.pair_id} | {result.pair_type} | {result.hierarchical_similarity:.4f} | {result.node_similarity:.4f} | {result.children_similarity:.4f} | {original_summary} | {comparison_summary} |")
        
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
            
            markdown_lines.append(f"**{result.pair_id}** | Hierarchical: {result.hierarchical_similarity:.4f} | Node: {result.node_similarity:.4f} | Children: {result.children_similarity:.4f}")
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
        
        # Results by Hierarchical Similarity
        markdown_lines.append("## Results by Hierarchical Similarity")
        markdown_lines.append("")
        markdown_lines.append("*Sorted by hierarchical similarity (highest to lowest)*")
        markdown_lines.append("")
        
        for i, result in enumerate(hierarchical_sorted, 1):
            markdown_lines.append(f"### {i}. {result.pair_id} ({result.pair_type})")
            markdown_lines.append("")
            markdown_lines.append(f"**Hierarchical Similarity:** {result.hierarchical_similarity:.4f}")
            markdown_lines.append(f"**Node Similarity:** {result.node_similarity:.4f}")
            markdown_lines.append(f"**Children Similarity:** {result.children_similarity:.4f}")
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
    """Main function to run the hierarchical similarity evaluation."""
    parser = argparse.ArgumentParser(description='Calculate hierarchical similarity using Hungarian algorithm')
    parser.add_argument('--experiment', choices=['negatives', 'positives', 'both'], default='both',
                        help='Which experiment to run (default: both)')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Weight for node vs children similarity (default: 0.6)')
    parser.add_argument('--negatives-file', default='evals/falsenegativeswithchildren.json',
                        help='Path to false negatives with children file')
    parser.add_argument('--positives-file', default='evals/falsepositiveswithchildren.json',
                        help='Path to false positives with children file')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY in .env file")
    
    # Initialize evaluator
    evaluator = HierarchicalSimilarityEvaluator(OPENAI_API_KEY)
    
    # Load evaluation data
    evaluator.load_evaluation_data(args.negatives_file, args.positives_file)
    
    # Determine which experiments to run
    run_negatives = args.experiment in ['negatives', 'both']
    run_positives = args.experiment in ['positives', 'both']
    
    # Calculate hierarchical similarities
    print(f"\\n=== STARTING HIERARCHICAL SIMILARITY ANALYSIS ===")
    print(f"Algorithm: Hungarian algorithm for optimal objection matching")
    print(f"Alpha (node weight): {args.alpha}")
    print(f"Running negatives: {run_negatives}")
    print(f"Running positives: {run_positives}")
    
    evaluator.calculate_hierarchical_similarities(
        process_negatives=run_negatives,
        process_positives=run_positives,
        alpha=args.alpha
    )
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_suffix = ""
    if run_negatives and not run_positives:
        experiment_suffix = "_negatives_only"
    elif run_positives and not run_negatives:
        experiment_suffix = "_positives_only"
    
    # Export detailed JSON results
    json_output_path = f"evals/results/hierarchical_similarity_results_{timestamp}{experiment_suffix}.json"
    evaluator.export_results(json_output_path)
    
    # Create markdown report
    markdown_output_path = f"evals/results/hierarchical_similarity_report_{timestamp}{experiment_suffix}.md"
    evaluator.create_markdown_report(markdown_output_path)
    
    # Print summary
    stats = evaluator.get_similarity_statistics()
    print(f"\\n=== HIERARCHICAL SIMILARITY ANALYSIS COMPLETE ===")
    print(f"Total pairs analyzed: {len(evaluator.similarity_results)}")
    
    if stats['overall']['hierarchical_similarity']['count'] > 0:
        print(f"\\nOverall Results:")
        print(f"  Node similarity mean: {stats['overall']['node_similarity']['mean']:.4f}")
        print(f"  Children similarity mean: {stats['overall']['children_similarity']['mean']:.4f}")
        print(f"  Hierarchical similarity mean: {stats['overall']['hierarchical_similarity']['mean']:.4f}")
        
        if run_negatives:
            print(f"\\nFalse Negatives:")
            print(f"  Hierarchical similarity mean: {stats['false_negatives']['hierarchical_similarity']['mean']:.4f}")
        
        if run_positives:
            print(f"\\nFalse Positives:")
            print(f"  Hierarchical similarity mean: {stats['false_positives']['hierarchical_similarity']['mean']:.4f}")
    
    print(f"\\nResults saved to:")
    print(f"  - {json_output_path}")
    print(f"  - {markdown_output_path}")

if __name__ == "__main__":
    main()