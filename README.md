# Filings RAG QA Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/filings-qa-rag-demo/blob/main/notebooks/filings_rag_qa.ipynb)

A small **Retrieval-Augmented Generation (RAG)** demo for question-answering over
company filings. Ask a question about an issuer's quarterly disclosures and get a
**cited, source-grounded answer** — every claim traceable to a specific chunk of a
specific document. The generation step is **switchable between OpenAI and
Anthropic**, so you can compare frontier models on identical retrieved context.

The bundled demo runs over **Nubank's Q2 filings** (earnings release, financial
statements, conference-call transcript — 62 chunks). Embeddings are precomputed and
shipped with the repo, so it runs out of the box.

## Run it

**Colab:** click the badge above. The notebook clones the repo, installs deps, and
prompts for your API keys.

**Local:**
```bash
git clone https://github.com/FranQuant/filings-qa-rag-demo
cd filings-qa-rag-demo
pip install openai anthropic pyarrow pandas python-dotenv
printf 'OPENAI_API_KEY=sk-...\nANTHROPIC_API_KEY=sk-ant-...\n' > .env
```
Then open `notebooks/filings_rag_qa.ipynb` and run all cells.

An **OpenAI key is required** (query embeddings use `text-embedding-3-small`). An
**Anthropic key is optional**, only for the provider comparison.

## Using the package

The notebook is a thin client over `genai_filings`:

```python
import sys; sys.path.insert(0, "src")
from genai_filings.answering import synthesize_answer

answer = synthesize_answer(
    query="How is net interest margin evolving?",
    issuer="NU", period="Q2", k=5,
    provider="anthropic",      # "openai" or "anthropic"
    model="claude-opus-4-8",   # omit for the provider default
    temperature=0.0, max_tokens=500,
)
print(answer["citations_valid"], answer["citations_invalid"])
print(answer["answer_markdown"])
```

`provider` selects the **generator only** — query embeddings always use OpenAI, so
retrieval is identical either way. Defaults: `gpt-5.5` / `claude-opus-4-8`. Note
the newest models drop `temperature` (handled automatically); for reproducible runs
use `gpt-4.1` or `claude-sonnet-4-6`, which still honour `temperature=0.0`.

## Scope

A research-augmentation demo, not investment advice. It answers strictly from the
provided excerpts — no forecasts, signals, or recommendations — and even cited
answers should be checked against the original filings.
