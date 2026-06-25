# Filings RAG QA Demo

[![Open in Colab — Query](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/filings-qa-rag-demo/blob/main/notebooks/filings_rag_qa.ipynb)
&nbsp;query
&nbsp;·&nbsp;
[![Open in Colab — Ingest](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/filings-qa-rag-demo/blob/main/notebooks/build_filings_index.ipynb)
&nbsp;ingest

A **Retrieval-Augmented Generation (RAG)** pipeline for question-answering over
company filings. Ask a question about an issuer's quarterly disclosures and get a
**cited, source-grounded answer** — every claim traceable to a specific chunk of a
specific document. The generation step is **switchable between OpenAI and
Anthropic**, so you can compare frontier models on identical retrieved context.

It's **plug-and-play**: point the ingestion notebook at any US filer (ticker +
quarter), and it pulls the 10-Q straight from SEC EDGAR, builds the index, and the
query notebook answers cited questions over it.

## Two notebooks

The workflow is split in two on purpose — ingestion is slow and costs API calls,
querying is fast and cheap, so you ingest once and query many times.

| Notebook | Role |
|---|---|
| `notebooks/build_filings_index.ipynb` | **Ingest.** Fetch a filing from SEC EDGAR → parse → chunk → embed → write artifacts. Change `ISSUER` + `PERIOD` to index a new company. |
| `notebooks/filings_rag_qa.ipynb` | **Query.** Retrieve relevant chunks and synthesize a cited answer over an already-built index. Fast, read-only. |

## Bundled issuers

Three are pre-indexed and shipped with the repo, so the query notebook runs out of
the box:

| Issuer | Period | Source |
|---|---|---|
| **NU** (Nubank) | 2025Q2 | earnings release, financial statements, conference-call transcript |
| **JPM** (JPMorgan) | 2026Q1 | 10-Q (SEC EDGAR) |
| **BAC** (Bank of America) | 2026Q1 | 10-Q (SEC EDGAR) |

The two banks make a nice side-by-side: on "how is net interest margin evolving?",
JPM's net yield is compressing (2.50%, down from 2.58%) while BAC's is expanding
(2.07%, up from 1.99%) — same question, both grounded in each filing.

## Run it

**Colab:** click a badge above (query or ingest). The notebook clones the repo,
installs deps, and prompts for what it needs.

**Local:**
```bash
git clone https://github.com/FranQuant/filings-qa-rag-demo
cd filings-qa-rag-demo
pip install openai anthropic pyarrow pandas python-dotenv beautifulsoup4 requests
printf 'OPENAI_API_KEY=sk-...\nANTHROPIC_API_KEY=sk-ant-...\n' > .env
```
Then open either notebook and run all cells.

An **OpenAI key is required** (query embeddings use `text-embedding-3-small`). An
**Anthropic key is optional**, only for the provider comparison. Ingestion also
needs an **SEC contact** (your email) for EDGAR's required User-Agent header.

## Ingest a new issuer

In `build_filings_index.ipynb`, set the two variables and run all cells:
```python
ISSUER = "WFC"     # any US filer's ticker
PERIOD = "Q1"      # the pipeline resolves this to a canonical YYYYQN period
```
Artifacts are written to `data/processed/filings/<ISSUER>/<YYYYQN>/`. After that,
the query notebook (or the package call below) can answer questions over it.

## Using the package

The query notebook is a thin client over `genai_filings`:
```python
import sys; sys.path.insert(0, "src")
from genai_filings.answering import synthesize_answer

answer = synthesize_answer(
    query="How is net interest margin evolving?",
    issuer="JPM", period="2026Q1", k=5,
    provider="anthropic",      # "openai" or "anthropic"
    model="claude-opus-4-8",   # omit for the provider default
    temperature=0.0, max_tokens=500,
)
print(answer["citations_valid"], answer["citations_invalid"])
print(answer["answer_markdown"])
```

`provider` selects the **generator only** — query embeddings always use OpenAI, so
retrieval is identical either way. Defaults: `gpt-5.5` / `claude-opus-4-8`. The
newest models drop `temperature` (handled automatically); for reproducible runs use
`gpt-4.1` or `claude-sonnet-4-6`, which still honour `temperature=0.0`.

## Scope and limitations

A research-augmentation demo, not investment advice. It answers strictly from the
provided excerpts — no forecasts, signals, or recommendations. EDGAR carries the
formal filing (10-Q/10-K), not earnings-call transcripts, so EDGAR-sourced issuers
answer well on reported numbers and MD&A but lack management Q&A color. Figures
drawn from dense financial tables should always be verified against the original
filing, and even cited answers warrant a check against the source.
