import os
import fitz
import pathlib
import sys
import pandas as pd
from tqdm import tqdm

from pdf_processing import *


class SimpleTextRetreiver:
    
    def __init__(self, margin_threshold=0.05):
        self.margin_threshold = margin_threshold
        
    def retreive_text_remove_formatting(self, page):
        
        formatting_lines = []
        content = []
        width, height = page.rect[2], page.rect[3]
        blocks = page.get_text('blocks')
        for block in blocks:
            left, top, right, bottom, text, *metadata = block
            if metadata[1]==0:
                # Check if the block is within the margin threshold from page edges
                if top/height < self.margin_threshold or (height - bottom)/height < self.margin_threshold:
                    # print(f"Non-essential (formatting): {text.strip()}")
                    formatting_lines.append(text.strip())
                    continue
                content.append(text)
                
        return ''.join(content), '\n'.join(formatting_lines)        


class PDFLLMParser:
    
    def __init__(self, directory, write_path, visualize_path, llm_callback, retreiver = SimpleTextRetreiver()):
        """  
        Args:
            directory (str): The path to the directory containing PDF files to be processed. This directory is scanned
                recursively for PDF files.
            write_path (str): The path to the directory where the analysis results, such as complexity scores and
                filtered text, will be written.
            visualize_path (str): The path to the directory where visualizations, such as PDFs with bounding boxes,
                will be stored.
            llm_callback (callable): A callback function that is called with the text extracted from PDF files. This
                function should take a single string argument (the text input) and return an analysis result used
                in the complexity analysis and other processing steps.
            retriever (object, optional): An instance of a class used for text retrieval and processing within PDF
                files. This object must have a method for retrieving text and optionally removing formatting. If not
                provided, a `SimpleTextRetriever` instance is used by default. This allows for customization of text
                extraction and processing strategies.
        """
        self.directory = pathlib.Path(directory)
        self.write_path = pathlib.Path(write_path)
        self.visualize_path = pathlib.Path(visualize_path)
        self.pdf_files = self._scan_directory_for_pdfs()
        self.llm_callback = llm_callback
        self.retreiver = retreiver
        self._assure_path_exist()

    def _assure_path_exist(self):
        
        assert self.directory.exists(), "Data directory doesn't exist"
        assert self.write_path.exists(), "Write directory doesn't exist"
        assert self.visualize_path.exists(), "Visualization directory doesn't exist"
        
    def _scan_directory_for_pdfs(self):
        """Scans the specified directory for PDF files and returns a list of their paths."""
        pdf_files = self.directory.rglob('*.pdf')
        return pdf_files

    def _plot_bbs(self, file_path, bb_level='blocks'):
        """
        Plots bounding boxes on pages of a given PDF file, saving the visual output to a specified path.
        bb_level: (words, blocks), specifying granularity of bounding boxes
        """
        
        path = pathlib.Path(file_path)
        name = path.name
        namespace = path.parts[-2]
        save_file = self.visualize_path/f"{namespace}_{name}"    
        with open_document(file_path) as doc:
            for page in tqdm(doc):
                draw_bounding_boxes_on_pdf(page,
                                           bb_level=bb_level)
            doc.save(save_file)
        return save_file

    def _get_input_for_llm(self, 
                       path,
                       bb_level='blocks',
                       include_font_info=False,
                       add_drawing_info=True):
        
        """Generates formatted input for the LLM from a PDF file, including optional font and drawing information."""
        
        with open_document(path) as doc:

            for page_i in range(doc.page_count):
                page = doc[page_i]
                tags = []
                blocks = page.get_text(bb_level, sort=True)
                # Assuming get_image_tag, preprocess_font_info, summarize_non_text_blocks,
                # detect_non_text_blocks, and blocks_to_text are defined elsewhere
                for block in blocks:
                    tags.append(get_image_tag(page, block[4]))
                
                fonts_info = preprocess_font_info(page.get_fonts())
                num_image, num_drawings = summarize_non_text_blocks(detect_non_text_blocks(page))
        
                input_formatted = blocks_to_text(blocks, [0,1,2,3], tags)
                if include_font_info:
                    input_formatted += f"\nfonts: {fonts_info}"
                if add_drawing_info:
                    input_formatted+= f"\n IMG: {num_image} DRAW: {num_drawings}"
                yield (page_i, input_formatted)                 
    
    def analyze_files(self, pdf_path, append_namespace=False):
        """
        Call LLM inference on  specified PDF file (iterating over pages), writing the ourput to a designated directory.
        append_namespace: to take two last parts in the file_path in order to infer the file_name (for cases like book_name/page_10.pdf)
        """
        
        path = pathlib.Path(pdf_path)
        name = path.name.split('.')[0]
        namespace = path.parts[-2]
        if append_namespace:
            file_namespace = f"{namespace}_{name}"
        else: 
            file_namespace = name
        write_path = (self.write_path/file_namespace)/"scores"
        write_path.mkdir(exist_ok=True, parents=True)
        print(f"Writing llm output to {write_path}")
        for i, input in tqdm(self._get_input_for_llm(pdf_path), total=get_pdf_page_count(path)):
            meta_file = write_path/f"{i+1}.txt"
            if not meta_file.exists():
                try:
                    output = self.llm_callback(input)
                except Exception as e:
                    print(e)
                    output = None 
                if output:
                    meta_file.write_text(output)
        print(f"Finish doing llm inderence")
        return write_path
        
    def aggregate_metadata(self, write_directory):
        """Aggregates metadata from analyzed files into a metadata CSV file in the specified directory."""
    
        write_path = pathlib.Path(write_directory)
        metadata_file = write_path/"metadata.csv"
        if not metadata_file.exists():
            # data = pd.DataFrame(columns=["file_path", "score"])
            array = []
            files_parsed = write_path.rglob('./*.txt')
            for file in files_parsed:
                try:
                     file_name = file.parts[-3]
                     page_number = file.parts[-1].split('.')[0]
                     score = float(file.read_text())
                     array.append([file, file_name, page_number, score])
                except Exception as e:
                    print(f"Fail extracting score for {file}")
            data = pd.DataFrame(array, columns=["file_path", "file_name", "page_number", "score"])
            data.to_csv(metadata_file, index=False)
        return metadata_file 

    def parse_pdfs(self, metadata, threshold = 0.5, is_greater=False):
        """Filters and parses PDFs based on complexity scores, organizing output into directories based on complexity level."""
        
        data = pd.read_csv(metadata)
        flag = "complex" if is_greater else "easy" 
        if is_greater:
            filtered_pds = data[data.score > threshold]
        else: 
            filtered_pds = data[data.score <= threshold]

        for i, row in tqdm(filtered_pds.iterrows(), total = len(filtered_pds)):
            file_name = self.directory/(str(row.file_name+'.pdf'))
            assert(file_name.exists())
            page_number = row.page_number
            with open_document(file_name) as file:
                page = file[page_number-1]
                text, formatting_lines = self.retreiver.retreive_text_remove_formatting(page)
                write_dir = (((self.write_path/row.file_name)/"parsed_text")/flag)
                write_dir.mkdir(exist_ok=True, parents=True)
                write_path = write_dir/f"page_{page_number}.txt"
                write_path.exists()
                text_to_write = text + "\n\n\n ### Formatting lines ### \n\n\n"+ formatting_lines
                write_path.write_text(text_to_write)
        return write_dir 
