from nexum_ai.optimizer import SemanticCache
import os

print("--- Default Behavior ---")
cache = SemanticCache()
# Force initialization
cache.initialize_model()
