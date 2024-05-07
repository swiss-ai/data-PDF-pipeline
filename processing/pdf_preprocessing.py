import os
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import io
import pathlib

def draw_blocks_and_words_on_image(pdf_path, output_dir, draw_on_white=True):
    """
    Draws bounding boxes for both text blocks and words on each page of a PDF file,
    then saves the results as PNG images with an optional white background.
    
    Parameters:
    - pdf_path: Path to the PDF file.
    - output_dir: Directory to save the images.
    - draw_on_white: Whether to use a white background.
    """
    doc = fitz.open(pdf_path)
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        if draw_on_white:
            white_bg = Image.new("RGB", img.size, "white")
            #white_bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else img)
            img = white_bg

        draw = ImageDraw.Draw(img)
        blocks = page.get_text("blocks")
        for b in blocks:
            draw.rectangle(b[:4], outline="red", width=2)
        words = page.get_text("words")
        for w in words:
            draw.rectangle(w[:4], outline="blue", width=1)

        output_path = os.path.join(output_dir, f"{doc_name}_page_{page_num + 1}.png")
        img.save(output_path)

    doc.close()

def convert_pdf_to_png(pdf_path, output_dir=None):
    """
    Convert each page of a PDF to PNG images using PyMuPDF.
    
    Parameters:
    - pdf_path: Path to the source PDF file.
    - output_dir: Directory to save the PNG files.
    """
    doc = fitz.open(pdf_path)
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        output_path = os.path.join(output_dir, f"{doc_name}_page_{page_num + 1}.png")
        pix.save(output_path)

    doc.close()

def record_pdf_text(pdf_path, output_path):
    """
    Opens a PDF file and returns the text from the first page.

    Args:
    pdf_path (str): The path to the PDF file.

    Returns:
    str: The text extracted from the first page of the PDF.
    """
    # Open the PDF file
    doc = fitz.open(pdf_path)
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    text = ''
    # Ensure the PDF contains at least one page

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        write_path = pathlib.Path(os.path.join(output_path, f"{doc_name}_page_{page_num + 1}.txt"))
        write_path.write_text(text)   
    return text