# PageWise

Upload a PDF and ask questions about it. Answers come only from the document and every answer cites the page it came from. Tables, charts and images get described by a vision model during upload so they're searchable too, instead of being skipped.

## Stack

- Streamlit for the UI
- LangChain for wiring the pieces together
- Gemini for embeddings, chat and vision
- Chroma as the local vector store
- PyMuPDF for reading the PDF and rendering page images

Everything runs locally except the Gemini API calls.

## Setup

1. Get a Gemini API key from https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env` and put the key in it:
   ```
   GOOGLE_API_KEY=your_key_here
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

**Ingestion.** PyMuPDF pulls the text layer out of each page and also renders the page to a PNG. Then a check decides whether the page probably has a table, chart or image on it: does it contain embedded raster images, does it have a lot of vector drawing operations, or is the text almost empty (which usually means a scan). Pages that pass get sent to Gemini, which writes out a text description of what's there. The extracted text and that description get joined into one document per page, tagged with the page number and the path to the page image, then split into chunks, embedded and written into Chroma.

**Answering.** The question gets embedded and used to pull the nearest chunks. Those chunks go into the prompt with their page numbers attached, and the prompt tells the model to answer only from what it's given. The Sources panel shows those same retrieved chunks rather than anything the model claims, so every citation points at real text and, where it helps, the original page image.

## Configuration

All optional, set in `.env`.

| Variable | Default | What it does |
|---|---|---|
| `CHAT_MODEL` | `gemini-3.5-flash` | Chat and vision model |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model |
| `VISION_MODE` | `auto` | `auto`, `all`, or `off` |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RETRIEVER_K` | `5` | Chunks retrieved per question |
| `PAGE_RENDER_DPI` | `150` | Page image resolution |

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

Processed documents get cached in `doc_store/` under a hash of the file contents *and* of the settings that shape the index, so uploading the same PDF twice reuses the index instead of paying to build it again. Changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`, `CHAT_MODEL`, `VISION_MODE` or `PAGE_RENDER_DPI` therefore builds a separate index rather than answering from one built under the old values, and changing them back reuses the earlier one. `RETRIEVER_K` is not part of the key, since it only affects how many chunks a question retrieves. Delete the folder to force a clean re-index.

## Limitations

- PDF only. Another format would need a loader that produces the same per-page structure.
- The vision model can get numbers wrong on dense tables. The page image is shown next to each citation so figures can be checked against the original.
- Retrieval is plain top-k similarity search, no reranking.
- One document at a time. The metadata supports more but the UI doesn't.
- No memory between questions, so follow-ups like "what about the second one" won't resolve.
