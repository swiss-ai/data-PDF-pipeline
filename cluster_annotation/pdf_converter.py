import concurrent.futures
import os
import queue
import threading
import time
import traceback
import logging
from tqdm import tqdm
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimized_batch_conversion(pdf_files,
                               output_base_dir,
                               format="jpg",
                               dpi=150,
                               quality=90,
                               max_workers=None,
                               chunk_size=35,
                               page_threshold=30):
    """
    Optimized batch conversion that handles varying PDF sizes efficiently.
    Uses page-based work distribution instead of file-based.
    
    Parameters:
    - pdf_files: List of PDF file paths
    - output_base_dir: Base directory for outputs
    - format: Output format (jpg or png)
    - dpi: Resolution in dots per inch
    - quality: JPEG quality (0-100)
    - max_workers: Maximum worker threads (None = auto)
    - chunk_size: Number of pages to process in each chunk for large PDFs
    - page_threshold: Threshold for small/large PDFs
    """
    # First, analyze all files to get page counts
    pdf_info = []
    total_pages = 0
    small_pdfs = []  # PDFs with few pages (process as whole files)
    large_pdfs = []  # PDFs with many pages (process page by page)
    
    PAGE_THRESHOLD = page_threshold  # PDFs with more pages than this will be processed page by page
    
    print("Analyzing PDFs...")
    for pdf_path in tqdm(pdf_files, desc="Counting pages", unit="file"):
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            
            pdf_info.append({
                'path': pdf_path,
                'pages': page_count,
                'name': os.path.splitext(os.path.basename(pdf_path))[0]
            })
            
            total_pages += page_count
            
            # Sort PDFs by size
            if page_count <= PAGE_THRESHOLD:
                small_pdfs.append(pdf_path)
            else:
                large_pdfs.append((pdf_path, page_count))
                
        except Exception as e:
            logger.error(f"Error analyzing {pdf_path}: {str(e)}")
    
    print(f"Found {len(pdf_files)} PDFs with {total_pages} total pages")
    print(f"  - {len(small_pdfs)} small PDFs (≤{PAGE_THRESHOLD} pages)")
    print(f"  - {len(large_pdfs)} large PDFs (>{PAGE_THRESHOLD} pages)")
    
    # Create a task queue for better load balancing
    task_queue = queue.Queue()
    results_dict = {}
    results_lock = threading.Lock()
    
    # Add small PDF tasks (entire files)
    for pdf_path in small_pdfs:
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(output_base_dir, doc_name)
        os.makedirs(output_dir, exist_ok=True)
        
        task_queue.put({
            'type': 'whole_file',
            'pdf_path': pdf_path,
            'output_dir': output_dir,
            'format': format,
            'dpi': dpi,
            'quality': quality
        })
    
    # Add large PDF tasks (page by page)
    for pdf_path, page_count in large_pdfs:
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(output_base_dir, doc_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a task for each chunk of pages (for better granularity)
        CHUNK_SIZE = chunk_size  # Process n pages at a time
        for start_page in range(0, page_count, CHUNK_SIZE):
            end_page = min(start_page + CHUNK_SIZE, page_count)
            
            task_queue.put({
                'type': 'page_range',
                'pdf_path': pdf_path,
                'output_dir': output_dir,
                'start_page': start_page,
                'end_page': end_page,
                'format': format,
                'dpi': dpi,
                'quality': quality
            })
    
    # Define worker function
    def worker():
        while True:
            try:
                # Get task with timeout to allow for clean shutdown
                task = task_queue.get(timeout=1)
                
                if task['type'] == 'whole_file':
                    # Process entire file
                    success = convert_pdf_with_ghostscript(
                        pdf_path=task['pdf_path'],
                        output_dir=task['output_dir'],
                        format=task['format'],
                        dpi=task['dpi'],
                        quality=task['quality'],
                        show_progress=False
                    )
                    
                    # Record result
                    with results_lock:
                        results_dict[task['pdf_path']] = success
                        
                elif task['type'] == 'page_range':
                    # Process specific page range
                    success = convert_pdf_pages_with_ghostscript(
                        pdf_path=task['pdf_path'],
                        output_dir=task['output_dir'],
                        start_page=task['start_page'],
                        end_page=task['end_page'],
                        format=task['format'],
                        dpi=task['dpi'],
                        quality=task['quality']
                    )
                    
                    # For page ranges, we only record final success when all chunks complete
                    with results_lock:
                        if task['pdf_path'] not in results_dict:
                            results_dict[task['pdf_path']] = True
                        if not success:
                            results_dict[task['pdf_path']] = False
                
                # Update progress bar
                pbar.update(1)
                task_queue.task_done()
                
            except queue.Empty:
                # Queue is empty, exit worker
                return
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                pbar.update(1)
                task_queue.task_done()
    
    # Determine total tasks
    total_tasks = len(small_pdfs) + sum(
        (page_count + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
        for _, page_count in large_pdfs
    )
    
    # Auto-determine worker count if not specified
    if max_workers is None:
        import multiprocessing
        max_workers = min(64, multiprocessing.cpu_count() * 4)  # Cap at 64
    
    # Start worker threads
    threads = []
    
    # Create and start progress bar
    with tqdm(total=total_tasks, desc="Converting", unit="task") as pbar:
        # Start workers
        for _ in range(min(max_workers, total_tasks)):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            threads.append(t)
        
        # Wait for all tasks to complete
        try:
            # Monitor for all tasks to be done
            while not task_queue.empty():
                time.sleep(0.1)
                
            # Wait for worker threads to finish
            for t in threads:
                t.join(timeout=5)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            return results_dict
    
    # Print summary
    successful = sum(1 for success in results_dict.values() if success)
    print(f"\nConversion complete: {successful}/{len(pdf_files)} PDFs successfully converted")
    
    return results_dict

def convert_pdf_with_ghostscript(pdf_path, output_dir, format="jpg", dpi=150, quality=90, show_progress=True):
    """
    Convert an entire PDF to images using Ghostscript.
    This function is a wrapper that calls convert_pdf_pages_with_ghostscript for the entire PDF.
    """
    try:
        # Get page count
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        
        # Call the page-based conversion function for all pages
        return convert_pdf_pages_with_ghostscript(
            pdf_path=pdf_path,
            output_dir=output_dir,
            start_page=0,
            end_page=page_count,
            format=format,
            dpi=dpi,
            quality=quality
        )
    except Exception as e:
        logger.error(f"Error converting {pdf_path}: {str(e)}")
        return False

def convert_pdf_pages_with_ghostscript(pdf_path, output_dir, start_page, end_page, format="jpg", dpi=150, quality=90):
    """
    Convert a specific page range of a PDF using Ghostscript.
    
    Parameters:
    - pdf_path: Path to PDF file
    - output_dir: Directory to save images
    - start_page: First page to convert (0-based)
    - end_page: Last page to convert (exclusive)
    - format: Output format (jpg or png)
    - dpi: Resolution in dots per inch
    - quality: JPEG quality (0-100)
    """
    import subprocess
    import os
    
    try:
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Determine output format and device
        if format.lower() in ("jpg", "jpeg"):
            device = "jpeg"
            ext = "jpg"
        else:
            device = "png16m"
            ext = "png"
        
        # Build the output pattern
        output_pattern = os.path.join(output_dir, f"{doc_name}_page_%d.{ext}")
        
        # Build the Ghostscript command with page range
        args = [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            f"-r{dpi}",
            f"-sDEVICE={device}",
            # Specify page range (Ghostscript uses 1-based page numbers)
            f"-dFirstPage={start_page + 1}",
            f"-dLastPage={end_page}"
        ]
        
        # Add quality setting for JPEG
        if device == "jpeg":
            args.append(f"-dJPEGQ={quality}")
        
        args.extend([
            "-dTextAlphaBits=4",
            "-dGraphicsAlphaBits=4",
            f"-sOutputFile={output_pattern}",
            pdf_path
        ])
        
        # Run Ghostscript
        process = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Ghostscript error on pages {start_page}-{end_page}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error converting pages {start_page}-{end_page} for {pdf_path}: {str(e)}")
        traceback.print_exc()
        return False