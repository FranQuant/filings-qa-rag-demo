# Next-Gen Quant Investing: LLMs, GenAI & Agentic AI in Trading & Asset Management

## LatAm ADR OU Pairs Trading with Semantic Risk Conditioning (NB1–NB3)

This project implements a **three-stage statistical arbitrage pipeline** for Latin American ADR pairs trading:

- **NB1:** Universe construction and correlation-based pair screening (frozen).
- **NB2:** Ornstein–Uhlenbeck pairs engine with cointegration testing, parameter locking, and out-of-sample execution.
- **NB3:** Execution-level **AI semantic risk conditioning** applied only in OU stress regimes (ALLOW / DOWNSIZE / BLOCK).

The AI layer does **not** generate alpha, re-estimate parameters, or alter trade timing; it solely modulates execution risk based on contextual information.

---

### Architecture Overview

AI is applied exclusively as **trade-level conditioning after signal generation and before execution**, consistent with institutional quant architectures.

```mermaid
flowchart TD

    %% NB1 — Universe Construction (Frozen)
    A[Select LatAm ADR Universe] --> B[Liquidity & Data Cleaning]
    B --> C[Correlation Screening]
    C --> D[Candidate Pair Selection]

    %% NB2 — Statistical OU Pairs Engine (Frozen)
    D --> E{Engle–Granger Cointegration}
    E -- Fail --> X[Discard Pair]
    E -- Pass --> F[OU Parameter Estimation]

    F --> G{OU Tail Entry?}
    G -- No --> H[Wait / Monitor]
    G -- Yes --> I[Generate Trade Signal]

    %% NB3 — AI Semantic Conditioning
    I --> J[Semantic Risk Assessment]
    J -->|ALLOW| K[Execute Trade (Full Size)]
    J -->|DOWNSIZE| L[Execute Trade (Reduced Size)]
    J -->|BLOCK| H
```
---

## Fundamentals-Based Portfolio Construction (NB4)

NB4 constructs a **deterministic 5×5 long/short portfolio snapshot** based exclusively on structured fundamental signals.

- **Inputs:** precomputed ADR fundamental dataset  
- **Signals:** profitability, quality, valuation  
- **Output:** equal-weighted, dollar-neutral long/short portfolio  
- **AI analyst overlay (optional):** qualitative interpretation only (non-decisional)

This module focuses strictly on **portfolio construction and ranking**.  
It does **not** generate time-series alpha signals, perform backtesting, or modify any upstream statistical engines.

---

## RAG Filings QA Capsule (NB5)

NB5 implements a **read-only Retrieval-Augmented Generation (RAG) pipeline** for filings-based investment research.

- **Inputs:** precomputed earnings releases, financial statements, and conference call transcripts  
- **Method:** deterministic chunk retrieval + cited answer synthesis  
- **Output:** traceable, source-grounded natural-language answers  

This module is designed for **research augmentation only**.  
It does **not** generate forecasts, trading signals, peer comparisons, or investment recommendations.

A reproducible demonstration is provided in:

`notebooks/05_RAG_demo_filings_qa.ipynb`

The notebook inspects artifacts, runs deterministic retrieval, and synthesizes a cited answer **without recomputing embeddings or indexes**.

---

