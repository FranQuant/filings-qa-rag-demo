# Filings RAG QA Demo

[![Query in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/filings-qa-rag-demo/blob/main/notebooks/filings_rag_qa.ipynb) **query**
&nbsp;·&nbsp;
[![Ingest in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/filings-qa-rag-demo/blob/main/notebooks/build_filings_index.ipynb) **ingest**

**Cited RAG over company filings.** Ask about an issuer's quarterly disclosures, get a
**source-grounded answer** with every claim traceable to a specific chunk. Generation is
**switchable between OpenAI and Anthropic**. **Plug-and-play:** point the ingest notebook at
any US filer and it pulls the 10-Q from SEC EDGAR, builds the index, and the query notebook
answers over it.

## Two notebooks

| Notebook | Role |
|---|---|
| `build_filings_index.ipynb` | **Ingest** (slow). EDGAR → parse → chunk → embed. Change `ISSUER` + `PERIOD`. |
| `filings_rag_qa.ipynb` | **Query** (fast, read-only). Retrieve + synthesize a cited answer. |

Ingest once, query many times. Three issuers come pre-indexed so the query notebook runs out
of the box:

| Issuer | Period | Source |
|---|---|---|
| NU (Nubank) | 2025Q2 | earnings release, statements, call transcript |
| JPM (JPMorgan) | 2026Q1 | 10-Q (EDGAR) |
| BAC (Bank of America) | 2026Q1 | 10-Q (EDGAR) |

## Run it

**Colab:** click a badge above. **Local:**
```bash
git clone https://github.com/FranQuant/filings-qa-rag-demo
cd filings-qa-rag-demo
pip install openai anthropic pyarrow pandas python-dotenv beautifulsoup4 requests
printf 'OPENAI_API_KEY=sk-...\nANTHROPIC_API_KEY=sk-ant-...\n' > .env   # Anthropic optional
```
Then open either notebook and run all cells. **OpenAI key required** (embeddings); Anthropic
optional (provider comparison); ingestion also needs your email for EDGAR's User-Agent.

## Package API

```python
import sys; sys.path.insert(0, "src")
from genai_filings.answering import synthesize_answer

answer = synthesize_answer(
    query="How is net interest margin evolving?",
    issuer="JPM", period="2026Q1", k=5,
    provider="anthropic",      # or "openai"
    model="claude-opus-4-8",   # omit for the provider default
    temperature=0.0, max_tokens=500,
)
print(answer["citations_valid"], answer["citations_invalid"])
print(answer["answer_markdown"])
```

`provider` selects the generator only (embeddings always use OpenAI). Defaults
`gpt-5.5` / `claude-opus-4-8`; newest models drop `temperature` (handled automatically) — for
reproducible runs use `gpt-4.1` or `claude-sonnet-4-6`.

## Scope

Research demo, not investment advice. Answers come strictly from the retrieved excerpts. EDGAR
carries the formal 10-Q/10-K, not call transcripts, so EDGAR issuers lack management Q&A color.
Verify table figures against the original filing.
