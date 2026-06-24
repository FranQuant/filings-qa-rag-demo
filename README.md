# Filings RAG QA Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/filings-qa-rag-demo/blob/main/notebooks/filings_rag_qa.ipynb)

A small, self-contained **Retrieval-Augmented Generation (RAG)** pipeline for
question-answering over company filings. Ask a question about an issuer's
quarterly disclosures and get a **cited, source-grounded answer** — with every
claim traceable back to a specific chunk of a specific document.

The generation step is **provider-switchable**: the same retrieval and the same
prompt can be answered by OpenAI *or* Anthropic models, side by side, so you can
compare frontier models on identical context.

---

## What it does

Given a question like *"How is net interest margin evolving?"*, the pipeline:

1. **Embeds the query** with OpenAI `text-embedding-3-small` (1536-dim).
2. **Retrieves** the most relevant chunks from precomputed filing embeddings
   using cosine similarity, with full provenance on every chunk (issuer, period,
   document type, source file, chunk ID).
3. **Synthesizes a cited answer** restricted to the retrieved excerpts, using
   either an OpenAI or an Anthropic model.
4. **Validates the citations** — every `[N]` reference in the answer is checked
   against the sources actually provided, so the "source-grounded" claim is
   enforced, not just asserted.

The bundled demo runs over **Nubank's Q2 filings** — earnings release, financial
statements, and the conference-call transcript (62 chunks total). The embeddings
and chunks are precomputed and shipped with the repo, so the demo runs without
any preprocessing.

---

## Quick start (Colab)

Click the **Open in Colab** badge above. The notebook will:

- clone this repository (package + precomputed artifacts),
- install dependencies,
- prompt you for your API keys (not stored or echoed),
- run retrieval, synthesis, and the OpenAI-vs-Anthropic comparison.

You need an **OpenAI API key** (required, for query embeddings). An **Anthropic
key is optional** and only enables the second half of the provider comparison.

---

## Quick start (local)

```bash
git clone https://github.com/FranQuant/filings-qa-rag-demo
cd filings-qa-rag-demo

# install dependencies
pip install openai anthropic pyarrow pandas python-dotenv

# provide your keys
cat > .env <<'ENV'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...   # optional, for the provider comparison
ENV
```

Then open `notebooks/filings_rag_qa.ipynb` in Jupyter and run all cells. The
notebook detects it is running locally and uses the repo in place (no clone,
no install).

---

## Using the library directly

The notebook is a thin client over the `genai_filings` package:

```python
import sys; sys.path.insert(0, "src")
from genai_filings.answering import synthesize_answer

answer = synthesize_answer(
    query="How is net interest margin evolving?",
    issuer="NU",
    period="Q2",
    k=5,
    provider="anthropic",          # "openai" or "anthropic"
    model="claude-opus-4-8",       # omit to use the provider default
    temperature=0.0,
    max_tokens=500,
)

print(answer["citations_valid"], answer["citations_invalid"])
print(answer["answer_markdown"])
```

`synthesize_answer` returns the answer markdown plus metadata: which provider
and model ran, how many chunks were used, and the valid/invalid citation lists.

### Provider switching

The `provider` argument selects the **generator only**. Query embeddings always
use the OpenAI model recorded in the embeddings manifest, so retrieval is
identical regardless of which generator answers. Defaults:

| Provider    | Default model        |
|-------------|----------------------|
| `openai`    | `gpt-5.5`            |
| `anthropic` | `claude-opus-4-8`   |

### A note on determinism

The newest reasoning models (OpenAI `gpt-5.x` and Anthropic `claude-opus-4-8`)
no longer accept a `temperature` parameter, so their output is not
bit-reproducible across runs. `synthesize_answer` handles this automatically —
it omits `temperature` for those models and passes it only to models that still
support it. For a reproducible, auditable run, use a model that still honours
`temperature=0.0`, such as OpenAI `gpt-4.1` or Anthropic `claude-sonnet-4-6`.

---

## Repository layout

```
notebooks/
  filings_rag_qa.ipynb         # the demo (Colab + local)
src/genai_filings/             # the RAG pipeline package
  acquisition/                 # fetch filings from source
  parsing/                     # parse PDFs
  indexing/                    # chunk parsed sections
  embeddings/                  # embed chunks
  retrieval/                   # query embedding + cosine retrieval
  answering/                   # provider-switchable cited synthesis
data/processed/filings/NU/Q2/  # precomputed chunks + embeddings (demo data)
```

The full pipeline (`acquisition` → `parsing` → `indexing` → `embeddings`) was
used to produce the precomputed artifacts. The demo exercises only the
`retrieval` and `answering` stages, reading those artifacts directly.

---

## Scope and limitations

This is a **research-augmentation demo**, not investment advice. It answers
questions strictly from the provided filing excerpts; it does not forecast,
generate trading signals, or make recommendations. Answers are only as good as
the retrieved context, and even cited answers should be verified against the
original filings.
