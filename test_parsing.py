import fitz
import os

os.makedirs("test_output", exist_ok=True)
pdf = fitz.open("sample.pdf")

print(f"Number of pages: {len(pdf)}")

for i, page in enumerate(pdf):
    page_number = i+1
    text = page.get_text("text")
    pix = page.get_pixmap(dpi=150)
    image_path = f"test_output/page_{page_number:03d}.png"
    pix.save(image_path)

    print(f"Page : {page_number}")
    print(f"Characters extracted : {len(text)}")
    print(f"First 200 chars : {text[:200]!r}")
    print(f"Saved image : {image_path}")

pdf.close()