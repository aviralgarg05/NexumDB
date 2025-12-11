"""
Semantic cache and query optimizer using local embedding models
"""

import numpy as np
from typing import Optional, List, Dict, Any
import json

from .stats import analyze
from .cost_model import cost_scan, cost_filter, cost_nested_loop

def estimate_selectivity(column_stats):
    return 0.1 
def estimate_cardinality(table_stats, selectivity):
    return table_stats.row_count * selectivity

class SemanticCache:
    """
    Caches query results using semantic similarity
    Uses local embedding models only
    """
    
    def __init__(self, similarity_threshold: float = 0.95) -> None:
        self.cache: List[Dict] = []
        self.similarity_threshold = similarity_threshold
        self.model = None
        
    def initialize_model(self) -> None:
        """Initialize local embedding model - deferred to avoid import errors"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Semantic cache initialized with all-MiniLM-L6-v2")
        except ImportError:
            print("Warning: sentence-transformers not installed, using fallback")
            self.model = None
    
    def vectorize(self, text: str) -> List[float]:
        """Convert text to embedding vector"""
        if self.model is None:
            self.initialize_model()
        
        if self.model is not None:
            embedding = self.model.encode(text)
            return embedding.tolist()
        else:
            return self._fallback_vectorize(text)
    
    def _fallback_vectorize(self, text: str) -> List[float]:
        """Simple fallback vectorization using character hashing"""
        vec = [0.0] * 384
        for i, char in enumerate(text[:384]):
            vec[i] = float(ord(char)) / 128.0
        return vec
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_arr = np.array(vec1)
        vec2_arr = np.array(vec2)
        
        dot_product = np.dot(vec1_arr, vec2_arr)
        norm1 = np.linalg.norm(vec1_arr)
        norm2 = np.linalg.norm(vec2_arr)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get(self, query: str) -> Optional[str]:
        """Retrieve cached result if similar query exists"""
        query_vec = self.vectorize(query)
        
        for entry in self.cache:
            similarity = self.cosine_similarity(query_vec, entry['vector'])
            if similarity >= self.similarity_threshold:
                print(f"Cache hit! Similarity: {similarity:.4f}")
                return entry['result']
        
        return None
    
    def put(self, query: str, result: str) -> None:
        """Store query and result in cache"""
        query_vec = self.vectorize(query)
        self.cache.append({
            'query': query,
            'vector': query_vec,
            'result': result
        })
        print(f"Cached query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()


class QueryOptimizer:
    """
    Reinforcement learning-based query optimizer
    Uses Q-learning to optimize query execution
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9) -> None:
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = 0.1
        
    def get_action(self, state: str, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in available_actions}
        
        state_values = self.q_table[state]
        best_action = max(available_actions, key=lambda a: state_values.get(a, 0.0))
        return best_action
    
    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        """Update Q-values based on observed reward"""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        current_q = self.q_table[state][action]
        
        max_next_q = 0.0
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        print(f"Updated Q({state}, {action}) = {new_q:.4f}")
    
    def feed_metrics(self, query: str, latency_ms: float) -> None:
        """Feed execution metrics to the optimizer"""
        reward = -latency_ms / 1000.0
        
        state = f"query_type_{len(query) // 10}"
        action = "execute"
        next_state = "completed"
        
        self.update(state, action, reward, next_state)
def choose_best_join(A_stats, B_stats):
    cost1 = cost_nested_loop(A_stats.row_count, B_stats.row_count)
    cost2 = cost_nested_loop(B_stats.row_count, A_stats.row_count)
    
    if cost1 <= cost2:
        return "A JOIN B", cost1
    else:
        return "B JOIN A", cost2



def test_vectorization() -> Dict[str, Any]:
    """Test function for Rust integration"""
    cache = SemanticCache()
    test_query = "SELECT * FROM users WHERE age > 25"
    vector = cache.vectorize(test_query)
    return {
        'query': test_query,
        'vector': vector[:10],
        'dimension': len(vector)
    }

def optimize(query):
    tables = query.tables  # e.g., ["A", "B"]
    A = query.stats["A"]
    B = query.stats["B"]

    plan, cost = choose_best_join(A, B)
    return plan, cost

def explain(query):
    plan, cost = optimize(query)
    print("Plan:", plan)
    print("Estimated cost:", cost)


if __name__ == "__main__":
    result = test_vectorization()
    print(json.dumps(result, indent=2))
    