import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import fitz
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Chat and vision share one provider; embeddings can come from another.
# "temperature" says whether to send one at all: langchain-anthropic raises on
# any non-default value for Claude Sonnet 5.5, so Anthropic runs at its own
# default. langchain-openai drops it itself for GPT-5 reasoning models.
CHAT_PROVIDERS = {
    "google": {"id": "google_genai", "key": "GOOGLE_API_KEY", "model": "gemini-3.5-flash", "temperature": True},
    "openai": {"id": "openai", "key": "OPENAI_API_KEY", "model": "gpt-5.4-mini", "temperature": True},
    "anthropic": {"id": "anthropic", "key": "ANTHROPIC_API_KEY", "model": "claude-haiku-4-5-20251001", "temperature": False},
}

# Anthropic has no embeddings API, so an Anthropic-only setup falls back to
# "local": the ONNX MiniLM model that ships with Chroma, no key needed.
EMBEDDING_PROVIDERS = {
    "google": {"id": "google_genai", "key": "GOOGLE_API_KEY", "model": "models/gemini-embedding-001"},
    "openai": {"id": "openai", "key": "OPENAI_API_KEY", "model": "text-embedding-3-small"},
    "local": {"id": None, "key": None, "model": "all-MiniLM-L6-v2"},
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "").strip().lower() or (
    LLM_PROVIDER if LLM_PROVIDER in EMBEDDING_PROVIDERS else "local"
)
CHAT_MODEL = os.getenv("CHAT_MODEL") or CHAT_PROVIDERS.get(LLM_PROVIDER, {}).get("model", "")
if EMBEDDING_PROVIDER == "local":
    # The local model is fixed, so EMBEDDING_MODEL has nothing to select.
    EMBEDDING_MODEL = EMBEDDING_PROVIDERS["local"]["model"]
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or EMBEDDING_PROVIDERS.get(EMBEDDING_PROVIDER, {}).get("model", "")
VISION_MODE = os.getenv("VISION_MODE", "auto")

# Anthropic rejects images over 5 MB. A dense page at high DPI can pass that
# as a PNG, so anything near the limit is re-encoded as JPEG before sending.
MAX_VISION_IMAGE_BYTES = 4_500_000

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
PAGE_RENDER_DPI = int(os.getenv("PAGE_RENDER_DPI", "150"))

BASE_DIR = Path(__file__).parent
STORE_DIR = BASE_DIR / "doc_store"
STORE_DIR.mkdir(exist_ok=True)

VISION_PROMPT = (
    "You are looking at one page of a document. Describe ONLY the visual "
    "content that a plain text extraction would miss: any tables "
    "(transcribe their data as a markdown table), charts or graphs "
    "(describe the type, axes, trend, and key numbers), diagrams, or "
    "photos/figures. If the page is plain prose with no such visual "
    "content, respond with exactly: NONE. Be precise with numbers, do not "
    "round or approximate values you can read."
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a careful research assistant answering questions about ONE
uploaded document. You must answer using ONLY the context below. Do not
use outside knowledge, and do not guess.

Each context block is labeled with its page number. When you state a fact,
mention the page number it came from inline, like "(Page 4)". If the
context does not contain the answer, say clearly that the document does not
appear to contain that information.

Context:
{context}

Question: {question}

Answer (cite page numbers inline):"""
)


class LocalEmbeddings(Embeddings):
    """Chroma's bundled ONNX model. Downloads about 80 MB the first time it runs."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._embed = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [[float(x) for x in vector] for vector in self._embed(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def config_problems():
    """Everything wrong with the current provider settings, as readable lines."""
    problems = []
    if LLM_PROVIDER not in CHAT_PROVIDERS:
        problems.append(f"LLM_PROVIDER={LLM_PROVIDER!r} is not supported. Use one of: {', '.join(CHAT_PROVIDERS)}.")
    if EMBEDDING_PROVIDER not in EMBEDDING_PROVIDERS:
        problems.append(f"EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r} is not supported. Use one of: {', '.join(EMBEDDING_PROVIDERS)}.")
    if problems:
        return problems

    keys = {CHAT_PROVIDERS[LLM_PROVIDER]["key"], EMBEDDING_PROVIDERS[EMBEDDING_PROVIDER]["key"]}
    for key in sorted(k for k in keys if k):
        if not os.getenv(key):
            problems.append(f"{key} is not set.")
    return problems


def make_chat_model(temperature):
    provider = CHAT_PROVIDERS[LLM_PROVIDER]
    options = {"temperature": temperature} if provider["temperature"] else {}
    return init_chat_model(CHAT_MODEL, model_provider=provider["id"], **options)


def make_embeddings():
    if EMBEDDING_PROVIDER == "local":
        return LocalEmbeddings()
    return init_embeddings(EMBEDDING_MODEL, provider=EMBEDDING_PROVIDERS[EMBEDDING_PROVIDER]["id"])


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    vector_store: Chroma


# Settings that change what ends up in the index, and so have to be part of
# its identity. RETRIEVER_K is deliberately absent: it only decides how many
# chunks a question pulls back, so changing it must not discard the index.
# CHAT_MODEL and PAGE_RENDER_DPI are in because the vision pass writes its
# descriptions into the indexed text, and it reads the rendered page to do it.
INDEX_SETTINGS = {
    "chat_model": CHAT_MODEL,
    "chunk_overlap": CHUNK_OVERLAP,
    "chunk_size": CHUNK_SIZE,
    "embedding_model": EMBEDDING_MODEL,
    "embedding_provider": EMBEDDING_PROVIDER,
    "llm_provider": LLM_PROVIDER,
    "page_render_dpi": PAGE_RENDER_DPI,
    "vision_mode": VISION_MODE,
}


def doc_id_for(file_bytes):
    # Keyed on the settings as well as the bytes. Without this, dropping
    # CHUNK_SIZE or switching embedding model still hit the index built
    # under the old values, and the answers came back as if nothing changed.
    digest = hashlib.md5(file_bytes)
    digest.update(json.dumps(INDEX_SETTINGS, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]


def page_needs_vision(page, text):
    if VISION_MODE == "off":
        return False
    if VISION_MODE == "all":
        return True

    try:
        if len(page.get_images(full=True)) > 0:
            return True
        # Charts and table borders show up here as vector drawings.
        if len(page.get_drawings()) > 8:
            return True
    except Exception:
        pass

    # Barely any text usually means the page is a scan.
    return len(text.strip()) < 40


def describe_page_visuals(image_path, llm):
    image_bytes = image_path.read_bytes()
    mime_type = "image/png"
    if len(image_bytes) > MAX_VISION_IMAGE_BYTES:
        image_bytes = fitz.Pixmap(str(image_path)).tobytes("jpeg", jpg_quality=85)
        mime_type = "image/jpeg"

    # LangChain's own image block, which each provider integration turns into
    # its native format. A bare image_url string only happens to work on Gemini:
    # it crashes the Anthropic converter and OpenAI rejects it.
    message = HumanMessage(
        content=[
            {"type": "text", "text": VISION_PROMPT},
            {
                "type": "image",
                "base64": base64.b64encode(image_bytes).decode("utf-8"),
                "mime_type": mime_type,
            },
        ]
    )

    # .text and not .content - Anthropic and the newer Gemini and OpenAI
    # models return a list of blocks instead of a plain string, and
    # .content.strip() blows up on it.
    description = llm.invoke([message]).text.strip()
    if description.upper() == "NONE":
        return ""
    return description


def read_meta(meta_file):
    # meta.json is written only once embedding has finished, so having it at
    # all is the signal that an index was completed rather than just started.
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text())
    except (ValueError, OSError):
        return None


def load_cached(doc_id, chroma_dir, filename):
    if not chroma_dir.exists() or not any(chroma_dir.iterdir()):
        return None

    meta = read_meta(chroma_dir.parent / "meta.json")

    embeddings = make_embeddings()
    vector_store = Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
        collection_name=doc_id,
    )

    # The directory existing is not proof the index is usable:
    # from_documents creates it before it has finished embedding, so an
    # upload that died partway through leaves a store that looks fine and
    # answers every question with "not in the document". Only trust the
    # cache when meta.json agrees with what is really in the collection.
    try:
        stored = len(vector_store.get(include=[])["ids"])
    except Exception as e:
        print(f"[warn] could not read cached index for {doc_id}: {e}")
        stored = 0

    expected = meta.get("num_chunks") if meta else None

    if meta is None or stored == 0 or stored != expected:
        print(
            f"[warn] cached index for {doc_id} is incomplete "
            f"({stored} chunks stored, {expected} expected), rebuilding it"
        )
        # Empty the collection rather than delete the directory. Re-ingest
        # appends, so leaving a half-built collection would store some
        # chunks twice, and the sqlite file is still open here - on Windows
        # removing the directory under it fails.
        try:
            vector_store.reset_collection()
        except Exception as e:
            print(f"[warn] could not clear stale index for {doc_id}: {e}")
        return None

    return IngestResult(
        doc_id=doc_id,
        filename=meta.get("filename", filename),
        num_pages=meta.get("num_pages", 0),
        num_chunks=meta.get("num_chunks", 0),
        vector_store=vector_store,
    )


def ingest_pdf(file_bytes, filename, progress_callback=None):
    problems = config_problems()
    if problems:
        raise RuntimeError(" ".join(problems) + " Check your .env.")

    doc_id = doc_id_for(file_bytes)
    doc_dir = STORE_DIR / doc_id
    pages_dir = doc_dir / "pages"
    chroma_dir = doc_dir / "chroma"

    cached = load_cached(doc_id, chroma_dir, filename)
    if cached is not None:
        return cached

    pages_dir.mkdir(parents=True, exist_ok=True)

    vision_llm = make_chat_model(temperature=0)
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(pdf)
    page_documents = []
    vision_attempts = 0
    vision_errors = []

    for i, page in enumerate(pdf):
        page_number = i+1

        if progress_callback:
            progress_callback(page_number, total_pages, f"Reading page {page_number}/{total_pages}")

        text = page.get_text("text")

        image_path = pages_dir / f"page_{page_number:04d}.png"
        if not image_path.exists():
            page.get_pixmap(dpi=PAGE_RENDER_DPI).save(str(image_path))

        visual_description = ""
        if page_needs_vision(page, text):
            if progress_callback:
                progress_callback(page_number, total_pages, f"Analyzing visuals on page {page_number}/{total_pages}")
            vision_attempts += 1
            try:
                visual_description = describe_page_visuals(image_path, vision_llm)
            except Exception as e:
                # One bad page shouldn't kill the whole upload.
                vision_errors.append(str(e))
                print(f"[warn] vision failed on page {page_number}: {e}")

        combined = text.strip()
        if visual_description:
            combined += f"\n\n[Visual content on this page]:\n{visual_description}"

        if not combined:
            continue

        page_documents.append(
            Document(
                page_content=combined,
                metadata={
                    "source": filename,
                    "page": page_number,
                    "has_visual": bool(visual_description),
                    "image_path": str(image_path),
                },
            )
        )

    pdf.close()

    # Every page failing is a setup problem (wrong model name, no image
    # support on the account), not a bad page. Carrying on would cache an
    # index with no visual content in it and never say why.
    if vision_attempts and len(vision_errors) == vision_attempts:
        raise RuntimeError(
            f"The vision model failed on all {vision_attempts} page(s) it was asked to read "
            f"({LLM_PROVIDER} / {CHAT_MODEL}). Check the model name and that it accepts images. "
            f"First error: {vision_errors[0]}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(page_documents)

    if progress_callback:
        progress_callback(total_pages, total_pages, f"Embedding {len(chunks)} chunks")

    embeddings = make_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
        collection_name=doc_id,
    )

    meta = {"filename": filename, "num_pages": total_pages, "num_chunks": len(chunks)}
    (doc_dir / "meta.json").write_text(json.dumps(meta))

    return IngestResult(doc_id, filename, total_pages, len(chunks), vector_store)


def format_context(docs):
    blocks = []
    for d in docs:
        page = d.metadata.get("page", "?")
        blocks.append(f"[Page {page}]:\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def ask(vector_store, question, k=RETRIEVER_K):
    # Search once and reuse the same chunks for both the prompt and the
    # sources panel, so the citations can't drift from what the model saw.
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    source_documents = retriever.invoke(question)

    llm = make_chat_model(temperature=0.2)
    chain = ANSWER_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "context": format_context(source_documents),
        "question": question,
    })

    return {"answer": answer, "source_documents": source_documents}
