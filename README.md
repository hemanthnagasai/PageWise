# PageWise

Upload a PDF and ask questions about it. Answers come only from the document and every answer cites the page it came from. Tables, charts and images get described by a vision model during upload so they're searchable too, instead of being skipped.

## Stack

- Streamlit for the UI
- LangChain for wiring the pieces together
- Google Gemini, OpenAI or Anthropic Claude for chat and vision, whichever you pick
- Gemini, OpenAI or a local model for embeddings
- Chroma as the local vector store
- PyMuPDF for reading the PDF and rendering page images

Everything runs locally except the calls to the model provider you choose.

## Setup

1. Pick a provider and get an API key for it:
   - Google Gemini: https://aistudio.google.com/apikey
   - OpenAI: https://platform.openai.com/api-keys
   - Anthropic Claude: https://console.anthropic.com/settings/keys
2. Copy `.env.example` to `.env`, set `LLM_PROVIDER` to `google`, `openai` or `anthropic`, and put that provider's key in it:
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_key_here
   ```
3. Make a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
5. Run it:
   ```
   streamlit run app.py
   ```

Open http://localhost:8501, upload a PDF in the sidebar, wait for it to index, then start asking.

## How it works

**Ingestion.** PyMuPDF pulls the text layer out of each page and also renders the page to a PNG. Then a check decides whether the page probably has a table, chart or image on it: does it contain embedded raster images, does it have a lot of vector drawing operations, or is the text almost empty (which usually means a scan). Pages that pass get sent to the vision model, which writes out a text description of what's there. The extracted text and that description get joined into one document per page, tagged with the page number and the path to the page image, then split into chunks, embedded and written into Chroma.

**Answering.** The question gets embedded and used to pull the nearest chunks. Those chunks go into the prompt with their page numbers attached, and the prompt tells the model to answer only from what it's given. The Sources panel shows those same retrieved chunks rather than anything the model claims, so every citation points at real text and, where it helps, the original page image.

## Configuration

All optional, set in `.env`.

| Variable | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `google` | `google`, `openai` or `anthropic`. Used for chat and for reading page images |
| `EMBEDDING_PROVIDER` | follows `LLM_PROVIDER` | `google`, `openai` or `local`. Defaults to `local` when the chat provider is `anthropic` |
| `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | | Only the keys for the providers you picked are needed |
| `CHAT_MODEL` | per provider (below) | Chat and vision model |
| `EMBEDDING_MODEL` | per provider (below) | Embedding model. Ignored for `local` |
| `VISION_MODE` | `auto` | `auto`, `all`, or `off` |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RETRIEVER_K` | `5` | Chunks retrieved per question |
| `PAGE_RENDER_DPI` | `150` | Page image resolution |

### Providers

| `LLM_PROVIDER` | Default chat model | Default embeddings |
|---|---|---|
| `google` | `gemini-3.5-flash` | `models/gemini-embedding-001` |
| `openai` | `gpt-5.4-mini` | `text-embedding-3-small` |
| `anthropic` | `claude-haiku-4-5-20251001` | `local` |

Anthropic has no embeddings API, so with `anthropic` the embeddings come from a small model that ships with Chroma and runs on your machine. It needs no key, and it downloads about 80 MB the first time it is used. You can instead set `EMBEDDING_PROVIDER=openai` or `google` and supply that key. Chat and embeddings are independent, so any combination works.

Switching provider or model builds a separate index, because vectors from different embedding models can't be compared.

## Files

```
app.py                  Streamlit UI
rag_pipeline.py         Ingestion and question answering
test_parsing.py         Scratch script used to check PyMuPDF output
inspect_chunks.py       Scratch script to dump what's in the vector store
requirements.txt        Dependencies
.env                    API key, not committed
doc_store/              Generated indexes and page images
```

Processed documents get cached in `doc_store/` under a hash of the file contents *and* of the settings that shape the index, so uploading the same PDF twice reuses the index instead of paying to build it again. Changing `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`, `CHAT_MODEL`, `VISION_MODE` or `PAGE_RENDER_DPI` therefore builds a separate index rather than answering from one built under the old values, and changing them back reuses the earlier one. `RETRIEVER_K` is not part of the key, since it only affects how many chunks a question retrieves. Delete the folder to force a clean re-index.

## Limitations

- PDF only. Another format would need a loader that produces the same per-page structure.
- The vision model can get numbers wrong on dense tables. The page image is shown next to each citation so figures can be checked against the original.
- Retrieval is plain top-k similarity search, no reranking.
- One document at a time. The metadata supports more but the UI doesn't.
- Anthropic models run at their default temperature, because the current Claude models reject any other value.
- Local embeddings are smaller and weaker than the hosted ones, so retrieval is less precise.
- Only the Google path has been run against its live API so far. The OpenAI and Anthropic paths are tested up to the request each provider's client builds.
- No memory between questions, so follow-ups like "what about the second one" won't resolve.
