"""
Natural Language to SQL Translation using local LLMs
Uses llama-cpp-python for local model inference
"""

from typing import Optional
import os

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class NLTranslator:
    """
    Translates natural language queries to SQL using a local quantized LLM
    """
    
    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 2048) -> None:
        """
        Initialize the translator with a local GGUF model
        
        Args:
            model_path: Path to GGUF model file (e.g., phi-2.Q4_K_M.gguf)
            n_ctx: Context window size
        """
        self.n_ctx = n_ctx
        self.model = None
        
        if model_path is None:
            model_path = os.getenv('NEXUMDB_MODEL_PATH')
        
        if not model_path:
            from .model_manager import ModelManager
            manager = ModelManager()
            
            model_path = manager.ensure_model(
                "phi-2.Q4_K_M.gguf",
                repo_id="TheBloke/phi-2-GGUF",
                filename="phi-2.Q4_K_M.gguf"
            )
        
        if model_path and os.path.exists(model_path):
            if Llama is None:
                print("Warning: llama-cpp-python not installed. NL translation will use fallback.")
                self.model = None
            else:
                try:
                    print(f"Loading LLM from {model_path}...")
                    self.model = Llama(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_threads=4,
                        verbose=False
                    )
                    print("LLM loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load LLM: {e}")
                    self.model = None
        else:
            print("Warning: No model path provided or download failed. NL translation will use fallback.")
    
    def translate(self, natural_query: str, schema: str = "") -> str:
        """
        Translate natural language to SQL
        
        Args:
            natural_query: Natural language question (e.g., "Show me all users named Alice")
            schema: Database schema context (e.g., "TABLE users (id INTEGER, name TEXT)")
        
        Returns:
            SQL query string
        """
        if not self.model:
            return self._fallback_translation(natural_query, schema)
        
        prompt = self._build_prompt(natural_query, schema)
        
        try:
            response = self.model(
                prompt,
                max_tokens=128,
                temperature=0.1,
                stop=["</s>", "\n\n", "Question:"],
                echo=False
            )
            
            sql = response['choices'][0]['text'].strip()
            sql = self._clean_sql(sql)
            
            print(f"Translated: '{natural_query}' -> '{sql}'")
            return sql
            
        except Exception as e:
            print(f"Translation error: {e}")
            return self._fallback_translation(natural_query, schema)
    
    def _build_prompt(self, natural_query: str, schema: str) -> str:
        """
        Build the prompt for the LLM with schema context
        """
        prompt = f"""You are a SQL expert. Convert natural language questions to SQL queries.

Schema:
{schema if schema else "No schema provided"}

Question: {natural_query}
SQL Query:"""
        
        return prompt
    
    def _clean_sql(self, sql: str) -> str:
        """
        Clean up the generated SQL query
        """
        sql = sql.strip()
        
        if sql.startswith("sql"):
            sql = sql[3:].strip()
        if sql.startswith("```"):
            sql = sql.split("```")[1].strip()
        if sql.startswith("\n"):
            sql = sql.strip()
        
        sql = sql.split(";")[0]
        
        return sql
    
    def _fallback_translation(self, natural_query: str, schema: str) -> str:
        """
        Simple rule-based fallback when LLM is not available
        """
        query = natural_query.lower()
        
        if "all users" in query or "show users" in query:
            if "named" in query or "called" in query:
                parts = query.split()
                for i, word in enumerate(parts):
                    if word in ["named", "called"] and i + 1 < len(parts):
                        name = parts[i + 1].strip("'\"")
                        return f"SELECT * FROM users WHERE name = '{name}'"
            return "SELECT * FROM users"
        
        if "all products" in query or "show products" in query:
            if "price" in query:
                if "less than" in query or "under" in query:
                    parts = query.split()
                    for i, word in enumerate(parts):
                        if word in ["than", "under"] and i + 1 < len(parts):
                            try:
                                price = int(parts[i + 1])
                                return f"SELECT * FROM products WHERE price < {price}"
                            except ValueError:
                                pass
                if "more than" in query or "over" in query or "above" in query:
                    parts = query.split()
                    for i, word in enumerate(parts):
                        if word in ["than", "over", "above"] and i + 1 < len(parts):
                            try:
                                price = int(parts[i + 1])
                                return f"SELECT * FROM products WHERE price > {price}"
                            except ValueError:
                                pass
            return "SELECT * FROM products"
        
        return "SELECT * FROM table"


def test_translator() -> None:
    """Test the translator with example queries"""
    translator = NLTranslator()
    
    schema = """
    TABLE users (id INTEGER, name TEXT, age INTEGER)
    TABLE products (id INTEGER, name TEXT, price INTEGER)
    """
    
    test_cases = [
        "Show me all users",
        "Show me all users named Alice",
        "Get all products",
        "Show products with price less than 100",
        "Find products that cost more than 500",
    ]
    
    print("\nNatural Language Translation Tests:")
    print("=" * 60)
    for nl_query in test_cases:
        sql = translator.translate(nl_query, schema)
        print(f"NL: {nl_query}")
        print(f"SQL: {sql}")
        print()


if __name__ == "__main__":
    test_translator()
