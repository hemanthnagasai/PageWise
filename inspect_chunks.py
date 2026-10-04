import rag_pipeline as rag

data = open("sample.pdf", "rb").read()
result = rag.ingest_pdf(data, "sample.pdf")

print(f"Pages : {result.num_pages}")
print(f"Chunks : {result.num_chunks}")
print()

stored = result.vector_store.get()

for i in range(len(stored["documents"])):
    text = stored["documents"][i]
    meta = stored["metadatas"][i]
    print(f"--- chunk {i} | page {meta['page']} | has_visual = {meta['has_visual']} | {len(text)} chars ---")
    print(text[:300])
    print()
