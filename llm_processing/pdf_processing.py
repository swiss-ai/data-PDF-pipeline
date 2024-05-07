import pathlib
import fitz
import re
from contextlib import contextmanager

from .text_processing import get_text_from_blocks, blocks_by_id, blocks_to_text, round_coord, get_blocks_rounded, blocks_to_df, prompt_lines

area = lambda box: (box[3]-box[1])*(box[2]-box[0])

@contextmanager
def open_document(path):
    doc = fitz.open(path)
    try:
        yield doc
    finally:
        doc.close()

def get_pdf_page_count(pdf_path):
    """
    Returns the total number of pages in a PDF document.

    :param pdf_path: The path to the PDF file.
    :return: Total number of pages in the PDF document.
    """
    # Open the PDF file
    with fitz.open(pdf_path) as doc:
        # Return the number of pages
        return doc.page_count
        
def draw_bounding_boxes_on_pdf(page,
                               bb_level = "blocks",
                               area_filter=False):
    """
    Draws bounding boxes on the pages of a PDF file according to the given blocks and saves each page as an image.

    Parameters:
    - page: Oened page oby fitz module
    """
    # Open the PDF file

    blocks = page.get_text(bb_level, sort=True)
    
    for block in blocks:
        # print(block)
        if area_filter:
            if area(block)<100:
                continue
        x0, y0, x1, y1, text, block_type, page_number = block
        # Draw a rectangle for the bounding box
        rect = fitz.Rect(x0, y0, x1, y1)
        page.draw_rect(rect, color=(1, 0, 0), width=1.5)  # Red color, 1.5pt line width
    
        


  
def add_extra_info(page, width, height):
    images = page.get_image_info()
    image = [image for image in images if image['height'] == height][0]
    size = image['size']
    res_x = image['xres']
    res_y = image['yres']
    return {"size":size, "res_x":res_x, "res_y":res_y}
    
def get_image_tag(page, text):
    if '<image' in text:
        width, height = extract_width_height(text)
        sup = add_extra_info(page, width, height)
        size, res_x , res_y = sup['size'], sup['res_x'], sup['res_y']
        return f"IMG w:{width} h:{height} size:{size} r_x: {res_x} r_y: {res_y}"
    else:
        return ""
        

def blocks_to_text(blocks, ids, tags, area_filter=False):
    blocks_rounded = get_blocks_rounded(blocks)
    filtered_blocks = blocks_by_id(blocks_rounded, ids)
    if area_filter:
        filtered_blocks = [box for box in filtered_blocks if area(box)>=100]
    represent = ['(' + ','.join(map(str, block)) + ')' for block in filtered_blocks]
    represent = [i[0]+" "+i[1] for i in zip(represent, tags)]
    return '\n'.join(represent)

def extract_width_height(input_str):
    """
    Extracts width and height from a given string.

    Parameters:
    - input_str: String containing width and height information.

    Returns:
    - A tuple (width, height) where both are integers if found, otherwise (None, None).
    """
    # Regular expression to find width and height
    width_height_pattern = r'width:\s*(\d+),\s*height:\s*(\d+)'

    # Search for matches
    match = re.search(width_height_pattern, input_str)

    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height
    else:
        return None, None

def preprocess_font_info(font_info_list):
    """
    Extracts and compacts important font information from a list of font details.
    
    Parameters:
    font_info_list (list of tuples): List of font information tuples extracted from a PDF.
    
    Returns:
    dict: A dictionary with compacted font information, including font names, styles, and sizes.
    """
    compacted_info = {}
    
    for info in font_info_list:
        size, font_type, _, font_name, style_identifier, _ = info
        
        # Extract actual font name and style (if any) from the font_name field
        font_name_parts = font_name.split('+')[-1]  # Remove subset tag (if exists)
        if "-" in font_name_parts:
            font_name, font_style = font_name_parts.rsplit("-", 1)
        else:
            font_name, font_style = font_name_parts, "Regular"
        
        # Update dictionary with font information
        if font_name not in compacted_info:
            compacted_info[font_name] = {'styles': set(), 'sizes': set()}
        
        compacted_info[font_name]['styles'].add(font_style)
        compacted_info[font_name]['sizes'].add(size)
    
    # Convert sets to sorted lists for consistent output
    for font_name in compacted_info:
        compacted_info[font_name]['styles'] = sorted(compacted_info[font_name]['styles'])
        compacted_info[font_name]['sizes'] = sorted(compacted_info[font_name]['sizes'])
    
    return compacted_info

def detect_non_text_blocks(page):
    """
    Detects non-text blocks (images, drawings, potential formulas) in a PDF document.
    
    Args:
    page: Page of document offered by fitz module  .
    
    Returns:
    list: A list of dictionaries with details about detected non-text blocks.
    """
    non_text_blocks = []

    # Get all image instances on the current page
    images = page.get_images(full=True)
    
    # Get drawings and other vector graphics (paths)
    drawings = page.get_drawings()
    
    # Text instances could also be analyzed here to detect formulas by patterns,
    # but we focus on images and drawings for this example.
    
    for img in images:
        # Each image entry provides details like the image's xref, and position on the page
        img_dict = {
            # 'page': page_num,
            'type': 'image',
            'xref': img[0],
            'details': img
        }
        non_text_blocks.append(img_dict)
    
    for drawing in drawings:
        # Drawings include paths and shapes that could be parts of formulas or diagrams
        drawing_dict = {
            # 'page': page_num,
            'type': 'drawing',
            'details': drawing
        }
        non_text_blocks.append(drawing_dict)

    return non_text_blocks


def summarize_non_text_blocks(non_text_blocks):
    """
    Creates a compact summary of non-text blocks in a document for input to an LLM.
    
    Args:
    non_text_blocks (list): A list of dictionaries with details about detected non-text blocks.
    
    Returns:
    str: A compact summary suitable for LLM input.
    """
    image_count = 0
    drawing_count = 0

    for block in non_text_blocks:
        if block['type'] == 'image':
            image_count += 1
        elif block['type'] == 'drawing':
            drawing_count += 1
            
    return image_count, drawing_count
