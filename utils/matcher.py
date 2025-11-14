"""
Face Matching Module using Cosine Similarity.

This module compares face embeddings using cosine similarity to find
the best match from a database of known faces.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from config import THRESHOLD


class FaceMatcher:
    """
    Face matcher using cosine similarity.
    
    Compares a query embedding against a database of known face embeddings
    to find the best match. Uses cosine similarity for comparison.
    """
    
    def __init__(self, embeddings_db: Dict[str, np.ndarray], threshold: float = None):
        """
        Initialize the face matcher.
        
        Args:
            embeddings_db: Dictionary mapping person names to their embeddings
                          Format: {"Jeffy": np.array([...]), "Anand": np.array([...])}
            threshold: Cosine similarity threshold for matching (0.0 to 1.0)
                      If None, uses config.THRESHOLD
        """
        self.embeddings_db = embeddings_db
        self.threshold = threshold if threshold is not None else THRESHOLD
        
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Formula: similarity = dot(A, B) / (norm(A) * norm(B))
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1 (typically 0 to 1 for normalized embeddings)
        """
        # Normalize vectors to unit length
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)  # Add small epsilon to avoid division by zero
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Calculate cosine similarity (dot product of normalized vectors)
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def find_best_match(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the best matching person for a query embedding.
        
        Args:
            query_embedding: Face embedding to match
            
        Returns:
            Tuple of (person_name, similarity_score)
            - person_name: Name of matched person, or None if no match above threshold
            - similarity_score: Best similarity score found
        """
        if not self.embeddings_db:
            return None, 0.0
        
        best_name = None
        best_similarity = -1.0
        
        # Compare query embedding against all known embeddings
        for name, stored_embedding in self.embeddings_db.items():
            similarity = self.cosine_similarity(query_embedding, stored_embedding)
            
            # Update best match if this similarity is higher
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name
        
        # Return match only if similarity exceeds threshold
        if best_similarity >= self.threshold:
            return best_name, best_similarity
        else:
            return None, best_similarity
    
    def match(self, query_embedding: np.ndarray) -> Dict:
        """
        Match a query embedding and return detailed result.
        
        Args:
            query_embedding: Face embedding to match
            
        Returns:
            Dictionary with:
            - 'name': Matched person name or "Unknown"
            - 'confidence': Similarity score (0.0 to 1.0)
            - 'is_match': Boolean indicating if match was found
        """
        name, similarity = self.find_best_match(query_embedding)
        
        return {
            'name': name if name else "Unknown",
            'confidence': similarity,
            'is_match': name is not None
        }

