# Provenance Graph RAG Demo

This repository contains a demo application for Retrieval-Augmented Generation (RAG) with a provenance graph and multi-factor confidence estimation. The UI is built with Streamlit, and the core RAG logic is implemented in Python.

The system is designed for educational and experimental use in the context of machine learning / AI courses and a master thesis.

## Main features

- **Provenance Graph RAG**
  - Builds a provenance graph over retrieved documents.
  - Supports retrieval **with** and **without** graph expansion.
  - Visualizes which documents and edges contributed to the final answer.

- **Multi-factor confidence estimation**
  - Combines several factors (retrieval similarity, score stability, context coverage, graph precision/centrality, query alignment, top match, answer evidence).
  - Outputs a scalar confidence score in [0, 1] and a qualitative band (low / cautious / moderate / high).
  - For OpenAI + web searches, confidence clipping is relaxed to better support cross-lingual scenarios (e.g., Ukrainian query, English snippets).

- **Local and web retrieval**
  - Local document corpus from user-uploaded files (`.txt`, `.md`, `.pdf`, `.docx`).
  - Web retrieval from multiple providers (see below).
  - Modes:
    - Local files only
    - Web only
    - Local + Web combined

- **Streamlit UI**
  - Configuration sidebar for model, search mode, web providers and graph parameters.
  - Tabs for single-query RAG and comparison "with graph vs without graph".
  - Visualization of the provenance graph and detailed confidence breakdown.

## Web providers

Web search is implemented in `web_retrieval.py` via a `WebSearchClient`.

Supported providers:

- **Wikipedia** – general encyclopedic articles.
- **OpenAlex** – academic metadata search.
- **arXiv** – preprints in CS/ML and related fields.
- **Semantic Scholar** – academic papers and abstracts.
- **Bing** – web search, using SerpAPI if available, falling back to Bing Web Search v7.
- **Google** – web search via SerpAPI.
- **DuckDuckGo** – Instant Answer API, used mainly for experimentation; in practice it often returns empty results for technical ML queries.

### Bing search behavior

`search_bing` works as follows:

1. If `SERPAPI_API_KEY` is set, the client uses **SerpAPI** with `engine=bing`.
2. Otherwise, if `BING_SEARCH_API_KEY` (and optionally `BING_SEARCH_ENDPOINT`) are set, it uses the official **Bing Search v7** REST API.

This avoids mis-using SerpAPI keys against the official Bing endpoint.

### Google search behavior

`search_google` uses **SerpAPI** with `engine=google`:

- Requires `SERPAPI_API_KEY`.
- Returns organic search results (title, snippet, URL) as web context documents.

### DuckDuckGo behavior

`search_duckduckgo` uses the **Instant Answer API** (`https://api.duckduckgo.com/`):

- First attempt with an expanded version of the original query (typically Ukrainian).
- If both `AbstractText` and `RelatedTopics` are empty, a second attempt is made with an English translation of the query.
- In many ML-related cases, the API still returns no structured answer; for production experiments you may want to simply not select DuckDuckGo in the UI.

## Environment variables

Set the following environment variables before running the app:

- `OPENAI_API_KEY` – required for OpenAI models (e.g., `gpt-4o-mini`) used in answer generation and translation.
- `SERPAPI_API_KEY` – required for Bing and Google web search via SerpAPI.
- `BING_SEARCH_API_KEY` – optional; used only if `SERPAPI_API_KEY` is not set to call the official Bing Search v7 API.
- `BING_SEARCH_ENDPOINT` – optional; Bing Search v7 endpoint URL. If not set, the default for the region is used.

Example (PowerShell on Windows):

```powershell
$env:OPENAI_API_KEY = "your_openai_key_here"
$env:SERPAPI_API_KEY = "your_serpapi_key_here"
# optional Bing v7:
# $env:BING_SEARCH_API_KEY = "your_bing_key_here"
# $env:BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
```

## Installation

1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Streamlit app

From the project root:

```powershell
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Using the app

1. **Configure the model and search mode** in the sidebar:
   - Choose the answer generation model (OpenAI vs local HF model).
   - Choose search mode: local, web, or local+web.

2. **Select web providers**:
   - Recommended combination for web mode: `Wikipedia`, `OpenAlex`, `arXiv`, `Semantic Scholar`, `Bing`, `Google`.
   - DuckDuckGo can be left disabled for technical ML queries due to weak coverage in the Instant Answer API.

3. **(Optional) Build local knowledge base**:
   - Upload documents (TXT, MD, PDF, DOCX) and click the button to process them.

4. **Ask a question**:
   - Enter a query (in Ukrainian or English).
   - Adjust `k` (number of base fragments) and `hops` (graph expansion depth).
   - Run the query and inspect:
     - The generated answer.
     - The confidence score and its explanation.
     - The provenance graph and context documents.

5. **Compare with / without graph**:
   - Use the comparison tab to run the same query with and without graph expansion and compare confidence, answer quality, and used documents.

## Experiments and evaluation

- `run_rag_experiments.py` – script for running systematic experiments, saving metrics and provenance snapshots:
  - `experiments_metrics.jsonl` – quantitative metrics.
  - `experiments_provenance.json` – provenance graphs and context for qualitative analysis.

- `smoke_test.py` – a simple script to verify core RAG functionality on a tiny local corpus.

## Known limitations

- Cross-lingual evidence scoring still relies partly on lexical overlap, which can underestimate answer evidence when query and documents are in different languages.
- DuckDuckGo Instant Answer API often returns empty responses for technical ML queries, even after English translation; for practical experiments, rely on Bing/Google/Wikipedia/OpenAlex/arXiv/Semantic Scholar.
- The UI uses some deprecated Streamlit components (`st.components.v1.html`), which may trigger warnings in newer Streamlit versions.
