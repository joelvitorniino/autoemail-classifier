# Projeto de Classificação e Resposta Automática de Emails com FastAPI

Este projeto implementa uma API para analisar, classificar e gerar respostas automáticas para emails corporativos, utilizando FastAPI, processamento de linguagem natural (NLP) personalizado e integração com a API DeepSeek.

---

## Funcionalidades

- Envio de emails em texto ou arquivo para processamento.
- Três modos de análise: `fast`, `balanced` e `thorough`.
- Classificação do email como **Produtivo** ou **Improdutivo**.
- Geração automática de resposta profissional, adaptada ao contexto.
- Extração de palavras-chave relevantes.
- Cache para otimizar o processamento de mensagens repetidas.
- Interface web básica para upload e envio de emails.
- Suporte CORS para integração com frontends externos.

---

## Tecnologias

- Python 3.9+
- FastAPI
- Jinja2 (templates HTML)
- Uvicorn (servidor ASGI)
- NLTK (tokenização, stopwords e stemmer em português)
- OpenAI DeepSeek API
- dotenv (variáveis de ambiente)
- Asyncio (processamento concorrente)

---

## Requisitos

- Python 3.9 ou superior
- Conta ativa na DeepSeek API e chave da API
- Variáveis de ambiente configuradas (.env)

---

## Como rodar o projeto

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

### 2. Crie e ative um ambiente virtual
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Crie um arquivo .env na raiz do projeto com o seguinte conteúdo:

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 5. Execute o servidor FastAPI
```bash
uvicorn main:app --reload
```

O servidor será iniciado em http://127.0.0.1:8000

### 6. Arquivo Txt
- Para testar o arquivo Txt, deverá seguir o modelo contido no arquivos test_emails.txt (localizado em /mock_test/test_emails.txt). 