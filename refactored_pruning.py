import json
import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import openai
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import logging
from dotenv import load_dotenv
import os

@dataclass
class GraphNode:
    """Represents a node in the philosophical reasoning graph."""
    uuid: str
    summary: str
    content: str
    node_type: str
    parent_id: Optional[str]
    depth: int
    terminal: bool
    nonsense: bool
    identical_to: Optional[str] = None
    is_central_question: bool = False
    children: List[str] = None  # Derived from parent_id relationships
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class PhilosophicalNodeAnalyzer:
    """
    Handles similarity and equivalence detection between philosophical nodes.
    This class is read-only and does not modify the graph structure.
    """
    
    def __init__(self, openai_api_key: str, similarity_threshold: float = 0.85,
                 max_candidates: int = 20):
        """
        Initialize the analyzer with OpenAI API access and parameters.
        
        Args:
            openai_api_key: OpenAI API key for embeddings and LLM calls
            similarity_threshold: Threshold for considering nodes equivalent
            max_candidates: Maximum number of candidate similar nodes to consider
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates
        self.nodes: Dict[str, GraphNode] = {}
        self.embeddings: np.ndarray = None
        self.faiss_index: faiss.Index = None
        self.uuid_to_index: Dict[str, int] = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_graph(self, graph_json: Dict[str, Any]):
        """Load graph from JSON representation and build children relationships."""
        self.nodes = {}
        
        # First pass: create all nodes
        for uuid, node_data in graph_json.items():
            self.nodes[uuid] = GraphNode(
                uuid=uuid,
                summary=node_data.get('summary', ''),
                content=node_data.get('content', ''),
                node_type=node_data.get('node_type', ''),
                parent_id=node_data.get('parent_id'),
                depth=node_data.get('depth', 0),
                terminal=node_data.get('terminal', False),
                nonsense=node_data.get('nonsense', False),
                identical_to=node_data.get('identical_to'),
                is_central_question=node_data.get('is_central_question', False)
            )
        
        # Second pass: build children relationships from parent_id
        for uuid, node in self.nodes.items():
            if node.parent_id and node.parent_id in self.nodes:
                self.nodes[node.parent_id].children.append(uuid)
    
    def _get_node_text(self, node: GraphNode) -> str:
        """Concatenate summary and content for a node."""
        return f"{node.summary} {node.content}".strip()
    
    def _embed_nodes(self):
        """Generate embeddings for all nodes using OpenAI's text-embedding-3-small."""
        self.logger.info("Generating embeddings for all nodes...")
        
        texts = []
        uuids = []
        
        for uuid, node in self.nodes.items():
            if node.identical_to is None:  # Only embed non-pruned nodes
                texts.append(self._get_node_text(node))
                uuids.append(uuid)
        
        # Get embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_texts
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        # Store embeddings in nodes and prepare for FAISS
        embeddings_matrix = []
        self.uuid_to_index = {}
        
        for i, (uuid, embedding) in enumerate(zip(uuids, all_embeddings)):
            self.nodes[uuid].embedding = np.array(embedding)
            embeddings_matrix.append(embedding)
            self.uuid_to_index[uuid] = i
        
        self.embeddings = np.array(embeddings_matrix).astype('float32')
        
        # Initialize FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings)
    
    def _find_similar_nodes(self, node_uuid: str) -> List[Tuple[str, float]]:
        """Find top similar nodes using FAISS."""
        if node_uuid not in self.uuid_to_index:
            return []
        
        node_index = self.uuid_to_index[node_uuid]
        query_embedding = self.embeddings[node_index:node_index+1]
        
        # Search for top k+1 similar nodes (including the node itself)
        scores, indices = self.faiss_index.search(query_embedding, self.max_candidates + 1)
        
        similar_nodes = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != node_index:  # Exclude the node itself
                similar_uuid = list(self.uuid_to_index.keys())[list(self.uuid_to_index.values()).index(idx)]
                similar_nodes.append((similar_uuid, float(score)))
        
        return similar_nodes[:self.max_candidates]
    
    def _compute_children_similarity(self, node1_uuid: str, node2_uuid: str) -> float:
        """
        Compute similarity between children of two nodes using optimal matching.
        
        This uses the Hungarian algorithm to find the optimal pairing of children
        that maximizes the total similarity, then computes a weighted score.
        """
        node1 = self.nodes[node1_uuid]
        node2 = self.nodes[node2_uuid]
        
        # Get children that haven't been pruned
        children1 = [c for c in node1.children if c in self.nodes and self.nodes[c].identical_to is None]
        children2 = [c for c in node2.children if c in self.nodes and self.nodes[c].identical_to is None]
        
        if not children1 or not children2:
            # If one has no children, similarity depends on whether both have no children
            return 1.0 if (not children1 and not children2) else 0.0
        
        # Compute pairwise similarities between all children
        similarity_matrix = np.zeros((len(children1), len(children2)))
        
        for i, child1_uuid in enumerate(children1):
            for j, child2_uuid in enumerate(children2):
                if (child1_uuid in self.uuid_to_index and 
                    child2_uuid in self.uuid_to_index):
                    emb1 = self.nodes[child1_uuid].embedding
                    emb2 = self.nodes[child2_uuid].embedding
                    similarity_matrix[i, j] = cosine_similarity([emb1], [emb2])[0, 0]
        
        # Use Hungarian algorithm for optimal matching
        # Convert to cost matrix (1 - similarity) for minimization
        cost_matrix = 1 - similarity_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Compute weighted similarity score
        matched_similarities = similarity_matrix[row_indices, col_indices]
        
        # Weight by the proportion of children that could be matched
        max_children = max(len(children1), len(children2))
        min_children = min(len(children1), len(children2))
        
        # Average of matched similarities, weighted by coverage
        if len(matched_similarities) > 0:
            avg_matched_similarity = np.mean(matched_similarities)
            coverage_weight = min_children / max_children
            return avg_matched_similarity * coverage_weight
        else:
            return 0.0
    
    def _combine_similarities(self, node_similarity: float, children_similarity: float) -> float:
        """
        Combine node and children similarities with adaptive weighting.
        
        Uses a weighted harmonic mean that emphasizes cases where both
        similarities are high, with adaptive weights based on the relative
        strengths of the two components.
        """
        # Adaptive weighting based on relative strengths
        total = node_similarity + children_similarity
        if total == 0:
            return 0.0
        
        # Weight more heavily the component that's stronger
        node_weight = 0.7 if node_similarity > children_similarity else 0.6
        children_weight = 1 - node_weight
        
        # Weighted harmonic mean for conservative combination
        if node_similarity == 0 or children_similarity == 0:
            # If either is 0, use weighted arithmetic mean
            return node_weight * node_similarity + children_weight * children_similarity
        
        weighted_harmonic = 2 / ((1/node_similarity) * node_weight + (1/children_similarity) * children_weight)
        return weighted_harmonic
    
    def _format_node_for_llm(self, node_uuid: str, include_children: bool = True) -> str:
        """Format a node and its children for LLM evaluation."""
        node = self.nodes[node_uuid]
        
        formatted = f"Node Type: {node.node_type}\n"
        formatted += f"Summary: {node.summary}\n"
        formatted += f"Content: {node.content}\n"
        formatted += f"Depth: {node.depth}\n"
        
        if include_children and node.children:
            formatted += "\nChildren:\n"
            for i, child_uuid in enumerate(node.children, 1):
                if child_uuid in self.nodes and self.nodes[child_uuid].identical_to is None:
                    child = self.nodes[child_uuid]
                    formatted += f"  Child {i} ({child.node_type}):\n"
                    formatted += f"    Summary: {child.summary}\n"
                    formatted += f"    Content: {child.content}\n"
        
        return formatted
    
    def _evaluate_equivalence_with_llm(self, node1_uuid: str, node2_uuid: str) -> bool:
        """
        Use GPT to evaluate if two nodes are functionally equivalent in the erotetic sense.
        """
        node1_formatted = self._format_node_for_llm(node1_uuid)
        node2_formatted = self._format_node_for_llm(node2_uuid)
        
        prompt = f"""You are evaluating whether two philosophical nodes are functionally equivalent in the erotetic sense. Two nodes are erotetically equivalent if their effect on a human reasoning philosophical agent would be functionally the same - meaning they would lead to the same patterns of inquiry, reasoning, and conclusions, even if they might differ in superficial presentation.

The content may include propositions marked with curly braces {{like this}}, which represent structured philosophical claims.

Consider the following two nodes from a philosophical reasoning graph, including their content and children:

NODE A:
{node1_formatted}

NODE B:
{node2_formatted}

Evaluate whether these two nodes represent views that are functionally equivalent in the erotetic sense. Consider:

1. Do they express the same fundamental philosophical position, argument, or conceptual distinction?
2. Would they lead a reasoning agent to the same conclusions and further inquiries?
3. Do their children (if any) support the same logical structure and reasoning path?
4. Are any differences merely presentational/linguistic rather than substantive?
5. For thesis nodes: Do they make the same core claims about the concept being analyzed?
6. For question nodes: Do they probe the same philosophical issue or distinction?
7. For support nodes: Do they provide the same type of evidential or logical support?

Note that nodes can be equivalent even if they use different terminology, as long as they capture the same philosophical insight or make the same functional contribution to the reasoning process.

Respond with only "TRUE" if they are erotetically equivalent, or "FALSE" if they are not. Do not provide explanation."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better reasoning
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "TRUE"
            
        except Exception as e:
            self.logger.error(f"Error in LLM evaluation: {e}")
            return False
    
    def find_equivalent_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Find all pairs of equivalent nodes without modifying the graph.
        
        Returns:
            List of tuples (node1_uuid, node2_uuid, combined_similarity)
        """
        self.logger.info("Starting equivalent node detection...")
        
        # Step 1 & 2: Embed nodes and create FAISS index
        self._embed_nodes()
        
        # Keep track of processed pairs to avoid redundant work
        processed_pairs = set()
        equivalent_pairs = []
        
        # Process each non-pruned node
        for node_uuid in list(self.nodes.keys()):
            node = self.nodes[node_uuid]
            
            if node.identical_to is not None:
                continue  # Skip already pruned nodes
            
            # Step 3: Find similar nodes
            similar_nodes = self._find_similar_nodes(node_uuid)
            
            for similar_uuid, node_similarity in similar_nodes:
                # Skip if already processed this pair
                pair_key = tuple(sorted([node_uuid, similar_uuid]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                # Skip if the similar node has been pruned
                if self.nodes[similar_uuid].identical_to is not None:
                    continue
                
                # Step 4: Compute children similarity
                children_similarity = self._compute_children_similarity(node_uuid, similar_uuid)
                
                # Step 5: Combine similarities
                combined_similarity = self._combine_similarities(node_similarity, children_similarity)
                
                self.logger.debug(f"Similarity between {node_uuid} and {similar_uuid}: "
                                f"node={node_similarity:.3f}, children={children_similarity:.3f}, "
                                f"combined={combined_similarity:.3f}")
                
                # Step 6: Check threshold and evaluate with LLM
                if combined_similarity >= self.similarity_threshold:
                    self.logger.info(f"High similarity detected, evaluating with LLM: "
                                   f"{node_uuid} vs {similar_uuid}")
                    
                    if self._evaluate_equivalence_with_llm(node_uuid, similar_uuid):
                        equivalent_pairs.append((node_uuid, similar_uuid, combined_similarity))
        
        self.logger.info(f"Found {len(equivalent_pairs)} equivalent node pairs.")
        return equivalent_pairs
    
    def are_nodes_equivalent(self, node1_uuid: str, node2_uuid: str) -> bool:
        """
        Check if two specific nodes are equivalent.
        
        Args:
            node1_uuid: First node UUID
            node2_uuid: Second node UUID
            
        Returns:
            True if nodes are equivalent, False otherwise
        """
        if node1_uuid not in self.nodes or node2_uuid not in self.nodes:
            return False
            
        if self.nodes[node1_uuid].identical_to is not None or self.nodes[node2_uuid].identical_to is not None:
            return False
            
        # Ensure embeddings are computed
        if self.embeddings is None:
            self._embed_nodes()
            
        # Get node similarity
        if node1_uuid not in self.uuid_to_index or node2_uuid not in self.uuid_to_index:
            return False
            
        node1_idx = self.uuid_to_index[node1_uuid]
        node2_idx = self.uuid_to_index[node2_uuid]
        
        emb1 = self.embeddings[node1_idx]
        emb2 = self.embeddings[node2_idx]
        node_similarity = cosine_similarity([emb1], [emb2])[0, 0]
        
        # Compute children similarity
        children_similarity = self._compute_children_similarity(node1_uuid, node2_uuid)
        
        # Combine similarities
        combined_similarity = self._combine_similarities(node_similarity, children_similarity)
        
        # Check threshold and evaluate with LLM
        if combined_similarity >= self.similarity_threshold:
            return self._evaluate_equivalence_with_llm(node1_uuid, node2_uuid)
            
        return False
    
    def get_node_similarity_score(self, node1_uuid: str, node2_uuid: str) -> Optional[float]:
        """
        Get the combined similarity score between two nodes.
        
        Args:
            node1_uuid: First node UUID
            node2_uuid: Second node UUID
            
        Returns:
            Combined similarity score or None if nodes don't exist
        """
        if node1_uuid not in self.nodes or node2_uuid not in self.nodes:
            return None
            
        # Ensure embeddings are computed
        if self.embeddings is None:
            self._embed_nodes()
            
        if node1_uuid not in self.uuid_to_index or node2_uuid not in self.uuid_to_index:
            return None
            
        node1_idx = self.uuid_to_index[node1_uuid]
        node2_idx = self.uuid_to_index[node2_uuid]
        
        emb1 = self.embeddings[node1_idx]
        emb2 = self.embeddings[node2_idx]
        node_similarity = cosine_similarity([emb1], [emb2])[0, 0]
        
        children_similarity = self._compute_children_similarity(node1_uuid, node2_uuid)
        
        return self._combine_similarities(node_similarity, children_similarity)
    
    def get_similarity_statistics(self) -> Dict[str, int]:
        """Get statistics about the similarity analysis."""
        total_nodes = len(self.nodes)
        active_nodes = sum(1 for node in self.nodes.values() if node.identical_to is None)
        
        # Statistics by node type
        node_type_stats = {}
        for node in self.nodes.values():
            node_type = node.node_type
            if node_type not in node_type_stats:
                node_type_stats[node_type] = {'total': 0, 'active': 0}
            node_type_stats[node_type]['total'] += 1
            if node.identical_to is None:
                node_type_stats[node_type]['active'] += 1
        
        return {
            'total_nodes': total_nodes,
            'active_nodes': active_nodes,
            'by_node_type': node_type_stats
        }
    
    def analyze_similarity_candidates(self, node_uuid: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Analyze similarity candidates for a specific node (useful for debugging/tuning).
        
        Returns detailed similarity analysis for the top candidates.
        """
        if node_uuid not in self.nodes or self.nodes[node_uuid].identical_to is not None:
            return []
        
        similar_nodes = self._find_similar_nodes(node_uuid)
        
        results = []
        for similar_uuid, node_similarity in similar_nodes[:top_k]:
            if self.nodes[similar_uuid].identical_to is not None:
                continue
                
            children_similarity = self._compute_children_similarity(node_uuid, similar_uuid)
            combined_similarity = self._combine_similarities(node_similarity, children_similarity)
            
            results.append({
                'candidate_uuid': similar_uuid,
                'candidate_summary': self.nodes[similar_uuid].summary,
                'node_similarity': float(node_similarity),
                'children_similarity': float(children_similarity),
                'combined_similarity': float(combined_similarity),
                'above_threshold': combined_similarity >= self.similarity_threshold,
                'node_type': self.nodes[similar_uuid].node_type,
                'depth': self.nodes[similar_uuid].depth
            })
        
        return results

class PhilosophicalGraphPruner:
    """
    Handles graph modifications and pruning operations.
    Uses PhilosophicalNodeAnalyzer for equivalence detection.
    """
    
    def __init__(self, analyzer: PhilosophicalNodeAnalyzer):
        """
        Initialize the pruner with an analyzer.
        
        Args:
            analyzer: PhilosophicalNodeAnalyzer instance for equivalence detection
        """
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
    
    def merge_nodes(self, target_uuid: str, source_uuid: str):
        """
        Merge source node into target node by migrating all descendants.
        
        Args:
            target_uuid: UUID of the target node to merge into
            source_uuid: UUID of the source node to be merged
        """
        if target_uuid not in self.analyzer.nodes or source_uuid not in self.analyzer.nodes:
            raise ValueError("One or both nodes do not exist")
            
        source_node = self.analyzer.nodes[source_uuid]
        target_node = self.analyzer.nodes[target_uuid]
        
        # Mark source as identical to target
        source_node.identical_to = target_uuid
        
        # Migrate all children from source to target
        for child_uuid in source_node.children:
            if child_uuid not in target_node.children:
                target_node.children.append(child_uuid)
                # Update the child's parent_id to point to target
                if child_uuid in self.analyzer.nodes:
                    self.analyzer.nodes[child_uuid].parent_id = target_uuid
        
        # Clear source node's children since they've been migrated
        source_node.children = []
        
        # If source node had a parent, we need to update the parent's children list
        if source_node.parent_id and source_node.parent_id in self.analyzer.nodes:
            parent_node = self.analyzer.nodes[source_node.parent_id]
            if source_uuid in parent_node.children:
                parent_node.children.remove(source_uuid)
                # Add target to parent's children if not already there
                if target_uuid not in parent_node.children:
                    parent_node.children.append(target_uuid)
                    # Update target's parent_id if it didn't have one
                    if target_node.parent_id is None:
                        target_node.parent_id = source_node.parent_id
        
        self.logger.info(f"Merged node {source_uuid} into {target_uuid}")
    
    def prune_equivalent_nodes(self) -> int:
        """
        Execute the complete pruning algorithm to detect and merge equivalent nodes.
        
        Returns:
            Number of nodes that were merged
        """
        equivalent_pairs = self.analyzer.find_equivalent_pairs()
        merged_count = 0
        
        for node1_uuid, node2_uuid, similarity in equivalent_pairs:
            # Check if nodes are still available (not already merged)
            if (self.analyzer.nodes[node1_uuid].identical_to is None and 
                self.analyzer.nodes[node2_uuid].identical_to is None):
                
                # Prioritize by depth (keep shallower node as target)
                target_node = self.analyzer.nodes[node1_uuid]
                source_node = self.analyzer.nodes[node2_uuid]
                
                if target_node.depth <= source_node.depth:
                    self.merge_nodes(node1_uuid, node2_uuid)
                else:
                    self.merge_nodes(node2_uuid, node1_uuid)
                
                merged_count += 1
        
        self.logger.info(f"Pruning complete. Merged {merged_count} equivalent node pairs.")
        return merged_count
    
    def export_pruned_graph(self) -> Dict[str, Any]:
        """Export the pruned graph to JSON format, maintaining original structure."""
        result = {}
        
        for uuid, node in self.analyzer.nodes.items():
            node_dict = {
                'summary': node.summary,
                'content': node.content,
                'node_type': node.node_type,
                'parent_id': node.parent_id,
                'depth': node.depth,
                'terminal': node.terminal,
                'nonsense': node.nonsense,
                'identical_to': node.identical_to
            }
            
            # Add is_central_question field only if it's True
            if node.is_central_question:
                node_dict['is_central_question'] = node.is_central_question
                
            result[uuid] = node_dict
        
        return result
    
    def get_pruning_statistics(self) -> Dict[str, int]:
        """Get statistics about the pruning process."""
        total_nodes = len(self.analyzer.nodes)
        pruned_nodes = sum(1 for node in self.analyzer.nodes.values() if node.identical_to is not None)
        active_nodes = total_nodes - pruned_nodes
        
        # Statistics by node type
        node_type_stats = {}
        for node in self.analyzer.nodes.values():
            node_type = node.node_type
            if node_type not in node_type_stats:
                node_type_stats[node_type] = {'total': 0, 'pruned': 0}
            node_type_stats[node_type]['total'] += 1
            if node.identical_to is not None:
                node_type_stats[node_type]['pruned'] += 1
        
        return {
            'total_nodes': total_nodes,
            'pruned_nodes': pruned_nodes,
            'active_nodes': active_nodes,
            'pruning_ratio': pruned_nodes / total_nodes if total_nodes > 0 else 0.0,
            'by_node_type': node_type_stats
        }

# Example usage
def example_usage():
    """Example of how to use the refactored architecture."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY in .env file")
    
    # Sample graph data matching your structure
    sample_graph = {
        "root-question": {
            "summary": "What is freedom (in the sense of liberty)?",
            "content": "What is freedom (in the sense of liberty)?",
            "node_type": "question",
            "parent_id": None,
            "depth": 0,
            "terminal": False,
            "nonsense": False,
            "identical_to": None,
            "is_central_question": True
        },
        "thesis-1": {
            "summary": "Freedom as Non-Interference",
            "content": "{Freedom is the absence of interference from others}, {Interference involves the intentional prevention of actions}, {Liberty is maximized when individuals can act without obstruction from others}",
            "node_type": "thesis",
            "parent_id": "root-question",
            "depth": 1,
            "terminal": False,
            "nonsense": False,
            "identical_to": None
        },
        "thesis-2": {
            "summary": "Freedom as Absence of Constraints",
            "content": "{Freedom is the absence of constraints on action}, {Constraints are limitations or restrictions on what one can do}, {Liberty is present when individuals face minimal external and internal barriers to action}",
            "node_type": "thesis",
            "parent_id": "root-question",
            "depth": 1,
            "terminal": False,
            "nonsense": False,
            "identical_to": None
        },
        "thesis-3": {
            "summary": "Freedom as Autonomy",
            "content": "{Freedom is the ability to govern oneself}, {Autonomy involves making decisions based on one's own values and reasoning}, {Liberty is achieved when individuals can act according to their own will}",
            "node_type": "thesis",
            "parent_id": "root-question",
            "depth": 1,
            "terminal": False,
            "nonsense": False,
            "identical_to": None
        },
        "support-1": {
            "summary": "Non-interference allows choice",
            "content": "{Without interference, individuals can make genuine choices}, {Choice requires the absence of external coercion}",
            "node_type": "support",
            "parent_id": "thesis-1",
            "depth": 2,
            "terminal": True,
            "nonsense": False,
            "identical_to": None
        },
        "support-2": {
            "summary": "Absence of constraints enables action",
            "content": "{When constraints are removed, individuals can act freely}, {Free action requires the removal of barriers}",
            "node_type": "support",
            "parent_id": "thesis-2",
            "depth": 2,
            "terminal": True,
            "nonsense": False,
            "identical_to": None
        }
    }
    
    # Initialize analyzer
    analyzer = PhilosophicalNodeAnalyzer(
        openai_api_key=openai_api_key,
        similarity_threshold=0.85,
        max_candidates=20
    )
    
    # Load graph into analyzer
    analyzer.load_graph(sample_graph)
    
    # Option 1: Just check equivalence between specific nodes
    are_equivalent = analyzer.are_nodes_equivalent("thesis-1", "thesis-2")
    print(f"Are thesis-1 and thesis-2 equivalent? {are_equivalent}")
    
    # Option 2: Find all equivalent pairs without modifying the graph
    equivalent_pairs = analyzer.find_equivalent_pairs()
    print(f"Found {len(equivalent_pairs)} equivalent pairs")
    
    # Option 3: Use pruner to actually modify the graph
    pruner = PhilosophicalGraphPruner(analyzer)
    merged_count = pruner.prune_equivalent_nodes()
    
    # Get results
    pruned_graph = pruner.export_pruned_graph()
    stats = pruner.get_pruning_statistics()
    
    print("Pruning Statistics:", stats)
    print("Pruned Graph:", json.dumps(pruned_graph, indent=2))

if __name__ == "__main__":
    example_usage()