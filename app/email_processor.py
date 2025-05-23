import os
import json
import hashlib
import asyncio
from typing import List, Dict, Optional
from openai import OpenAI
from app.utils.nlp_utils import NLPProcessor
from dotenv import load_dotenv

load_dotenv()

class EmailProcessor:
    """Processes emails for classification and response generation using DeepSeek API."""
    
    # Constants
    MODE_SETTINGS = {
        "fast": {
            "max_tokens": 150,
            "timeout": 10,
            "batch_size": 5,
            "response_keys": ["categoria", "resposta", "keywords"]
        },
        "balanced": {
            "max_tokens": 400,
            "timeout": 20,
            "batch_size": 3,
            "response_keys": ["categoria", "resposta", "keywords"]
        },
        "thorough": {
            "max_tokens": 800,
            "timeout": 20,
            "batch_size": 2,
            "response_keys": ["categoria", "resposta", "keywords", "justificativa"]
        }
    }

    CLASSIFICATION_GUIDANCE = """
    Classifique o email como:
    - **Produtivo**: Relacionado a tarefas, projetos, metas ou resultados profissionais.
    - **Improdutivo**: Focado em atividades sociais, pessoais ou recreativas.
    Use as palavras-chave como indicadores de contexto.
    """

    def __init__(self):
        """Initialize the email processor with NLP tools and API client."""
        self.nlp = NLPProcessor()
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
        self.cache = {}

    def _make_cache_key(self, text: str, mode: str) -> str:
        """Generate a unique cache key for processed messages."""
        return hashlib.sha256(f"{text}{mode}".encode('utf-8')).hexdigest()

    def _build_prompt(self, cleaned_text: str, key_phrases: List[str], mode: str) -> str:
        """Construct the appropriate prompt based on processing mode."""
        base_prompt = {
            "fast": self._build_fast_prompt,
            "balanced": self._build_balanced_prompt,
            "thorough": self._build_thorough_prompt
        }.get(mode)
        
        if not base_prompt:
            raise ValueError(f"Invalid mode: {mode}")
            
        return base_prompt(cleaned_text, key_phrases)

    def _build_fast_prompt(self, text: str, key_phrases: List[str]) -> str:
        """Build prompt for fast processing mode."""
        return f"""{self.CLASSIFICATION_GUIDANCE}
        Analise rapidamente este email corporativo:
        1. Classifique como Produtivo/Improdutivo
        2. Gere resposta curta (2 frases)
        3. Liste 3 palavras-chave

        Contexto: {', '.join(key_phrases[:3])}
        Email: {text}

        Responda em JSON: {{
            "categoria": "...",
            "resposta": "...",
            "keywords": ["..."]
        }}"""

    def _build_balanced_prompt(self, text: str, key_phrases: List[str]) -> str:
        """Build prompt for balanced processing mode."""
        return f"""{self.CLASSIFICATION_GUIDANCE}
        Analise este email:
        1. Classifique
        2. Gere resposta (4 frases)
        3. Liste 5 palavras-chave

        Contexto: {', '.join(key_phrases[:5])}
        Email: {text}

        Responda em JSON: {{
            "categoria": "...",
            "resposta": "...",
            "keywords": ["..."]
        }}"""

    def _build_thorough_prompt(self, text: str, key_phrases: List[str]) -> str:
        """Build prompt for thorough processing mode."""
        return f"""{self.CLASSIFICATION_GUIDANCE}
        Analise profundamente este email:
        1. Classifique com justificativa
        2. Gere resposta detalhada (6 frases)
        3. Liste 8 palavras-chave com explicação

        Contexto: {', '.join(key_phrases[:8])}
        Email: {text}

        Responda em JSON: {{
            "categoria": "...",
            "resposta": "...",
            "keywords": ["..."],
            "justificativa": "..."
        }}"""

    async def process_single_message(self, text: str, mode: str = "fast") -> Dict[str, str]:
        """Process a single email message."""
        if not text or not isinstance(text, str):
            return self._error_response(text, "Texto inválido ou vazio")
        
        cache_key = self._make_cache_key(text, mode)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            key_phrases = self.nlp.extract_key_phrases(text)
            cleaned_text = ' '.join(self.nlp.tokenize_and_clean(text))
            prompt = self._build_prompt(cleaned_text, key_phrases, mode)
            
            response = await self._call_deepseek_api(prompt, mode)
            processed = self._process_api_response(response, mode, key_phrases)
            
            self.cache[cache_key] = processed
            return processed
            
        except Exception as e:
            return self._error_response(text, str(e))

    async def _call_deepseek_api(self, prompt: str, mode: str) -> dict:
        """Make API call to DeepSeek with appropriate settings."""
        settings = self.MODE_SETTINGS[mode]
        return self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "Seja conciso e profissional."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=settings["max_tokens"],
            timeout=settings["timeout"]
        )

    def _process_api_response(self, response: dict, mode: str, key_phrases: List[str]) -> dict:
        """Process and validate the API response."""
        content = json.loads(response.choices[0].message.content)
        result = {
            "categoria": content.get("categoria", "Desconhecido"),
            "resposta": content.get("resposta", ""),
            "keywords": content.get("keywords", key_phrases)
        }
        
        if mode == "thorough":
            result["justificativa"] = content.get("justificativa", "")
            
        return result

    def _error_response(self, text: str, error_msg: str) -> dict:
        """Generate standardized error response."""
        return {
            "corpo": text,
            "categoria": "Erro",
            "resposta": f"Erro no processamento: {error_msg}",
            "keywords": [],
            "justificativa": ""
        }

    async def process_multiple_messages(self, content: str, mode: str = "fast") -> List[Dict[str, str]]:
        """Process multiple email messages in batches."""
        messages = [m.strip() for m in content.split('---') if m.strip()]
        if not messages:
            return []
        
        batch_size = self.MODE_SETTINGS[mode]["batch_size"]
        results = []
        
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            tasks = [self.process_single_message(msg, mode) for msg in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results