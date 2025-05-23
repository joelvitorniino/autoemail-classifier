import os
import json
import hashlib
import asyncio
from typing import List, Dict
from openai import OpenAI
from app.utils.nlp_utils import NLPProcessor
from dotenv import load_dotenv

load_dotenv()

class EmailProcessor:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
        self.cache = {}
    
    def _make_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _build_prompt(self, cleaned_text: str, key_phrases: List[str], mode: str) -> str:
        # Definições comuns para todos os modos
        classification_guidance = """
Classifique o email como:
- **Produtivo**: Relacionado a tarefas, projetos, metas ou resultados profissionais (ex.: reuniões, relatórios, planejamento, entregas).
- **Improdutivo**: Focado em atividades sociais, pessoais ou recreativas, sem relação direta com objetivos de trabalho (ex.: festas, aniversários, eventos recreativos, enquetes sociais).
Use as palavras-chave fornecidas ({', '.join(key_phrases)}) como indicadores de contexto. Palavras como "festa", "aniversário", "torneio" ou "confraternização" sugerem atividades improdutivas, enquanto "reunião", "relatório" ou "projeto" indicam produtividade. Se o email misturar temas, priorize o objetivo principal.
"""
        
        if mode == "fast":
            return f"""{classification_guidance}

Analise rapidamente este email corporativo e:
1. Classifique como "Produtivo" ou "Improdutivo" com base nos critérios acima.
2. Gere uma resposta curta sugerida (máximo 2 frases) para responder o email, mantendo o tom profissional e adequado ao contexto (amigável para emails sociais, formal para profissionais). Não inclua saudações finais como "Atenciosamente" ou "Cordialmente". Termine com "[Seu Nome], [Cargo]".
3. Liste até 3 palavras-chave principais, priorizando termos que refletem o objetivo do email.

Contexto: {', '.join(key_phrases[:3])}

Email:
{cleaned_text}

Responda em JSON:
{{
    "categoria": "Produtivo/Improdutivo",
    "resposta": "Resposta curta aqui",
    "keywords": ["palavra1", "palavra2", "palavra3"]
}}"""
        elif mode == "balanced":
            return f"""{classification_guidance}

Analise este email corporativo e:
1. Classifique como "Produtivo" ou "Improdutivo" com base nos critérios acima.
2. Gere uma resposta profissional sugerida (máximo 4 frases) para responder o email, ajustando o tom ao contexto (amigável mas profissional para emails sociais, formal para profissionais). Não inclua saudações finais como "Atenciosamente" ou "Cordialmente". Termine com "[Seu Nome], [Cargo]".
3. Destaque até 5 palavras-chave importantes, priorizando termos que refletem o objetivo do email.

Contexto: {', '.join(key_phrases[:5])}

Email:
{cleaned_text}

Responda em JSON:
{{
    "categoria": "Produtivo/Improdutivo",
    "resposta": "Resposta aqui",
    "keywords": ["palavra1", "palavra2", "palavra3", "palavra4", "palavra5"]
}}"""
        elif mode == "thorough":
            return f"""{classification_guidance}

Analise profundamente este email corporativo e:
1. Classifique como "Produtivo" ou "Improdutivo" com base nos critérios acima, fornecendo uma justificativa detalhada que explique a escolha com base no conteúdo e nas palavras-chave.
2. Gere uma resposta detalhada e profissional sugerida (máximo 6 frases) para responder o email, mantendo o conteúdo conciso, completo e com tom adequado ao contexto (amigável mas profissional para emails sociais, formal para profissionais). Não inclua saudações finais como "Atenciosamente" ou "Cordialmente". Termine com "[Seu Nome], [Cargo]".
3. Liste até 8 palavras-chave relevantes, com uma breve explicação para cada uma, destacando sua relação com o objetivo do email.

Contexto: {', '.join(key_phrases[:8])}

Email:
{cleaned_text}

Responda em JSON:
{{
    "categoria": "Produtivo/Improdutivo",
    "resposta": "Resposta detalhada aqui",
    "keywords": [
        "palavra1: explicação",
        "palavra2: explicação",
        "palavra3: explicação",
        "palavra4: explicação",
        "palavra5: explicação",
        "palavra6: explicação",
        "palavra7: explicação",
        "palavra8: explicação"
    ],
    "justificativa": "Explicação detalhada da classificação"
}}"""
        else:
            raise ValueError(f"Modo inválido: {mode}")
    
    async def process_single_message(self, text: str, mode: str = "fast") -> Dict[str, str]:
        if not text or not isinstance(text, str):
            return {
                "corpo": text,
                "categoria": "Erro",
                "resposta": "Erro: Texto inválido ou vazio",
                "keywords": [],
                "justificativa": ""
            }
        
        cache_key = self._make_cache_key(text + mode)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        key_phrases = self.nlp.extract_key_phrases(text)
        cleaned = ' '.join(self.nlp.tokenize_and_clean(text))
        prompt = self._build_prompt(cleaned, key_phrases, mode)
        
        try:
            max_tokens = {"fast": 150, "balanced": 400, "thorough": 800}.get(mode, 400)
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Seja conciso, profissional e evite saudações finais como 'Atenciosamente' ou 'Cordialmente'. Classifique emails com base em critérios claros e ajuste o tom da resposta ao contexto."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                timeout=10 if mode == "fast" else 20
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            response_data = {
                "categoria": result.get("categoria", "Desconhecido"),
                "resposta": result.get("resposta", ""),
                "keywords": result.get("keywords", key_phrases),
                "justificativa": result.get("justificativa", "") if mode == "thorough" else ""
            }
            
            self.cache[cache_key] = response_data
            return response_data
            
        except Exception as e:
            return {
                "corpo": text,
                "categoria": "Erro",
                "resposta": f"Erro no processamento: {str(e)}",
                "keywords": key_phrases,
                "justificativa": ""
            }
    
    async def process_multiple_messages(self, content: str, mode: str = "fast") -> List[Dict[str, str]]:
        messages = [m.strip() for m in content.split('---') if m.strip()]
        if not messages:
            return []
        
        batch_sizes = {"fast": 5, "balanced": 3, "thorough": 2}
        batch_size = batch_sizes.get(mode, 3)
        
        results = []
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            tasks = [self.process_single_message(msg, mode) for msg in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results