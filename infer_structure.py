import os
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pathlib
import fasttext
import numpy as np
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import langdetect

from processing.data_preparation import prepare_pdf_data, PairedImageDataset
from models.backbones import CustomMobileNetV3


def prepare_dataset(pdf_files, output_dir, num_workers=None):
    """
    Prepare the dataset for processing by converting PDF files to data suitable for ML models.
    
    Args:
    pdf_files (list): A list of paths to PDF files to be processed.
    output_dir (str): The directory where processed files will be saved.
    num_workers (int, optional): Number of worker processes to use. Defaults to the number of CPU cores available.

    Returns:
    list: A list of directories containing processed data for each PDF.
    """
    if num_workers is None:
        num_workers = cpu_count()

    processing_func = partial(prepare_pdf_data, output_dir=output_dir)
    
    with Pool(num_workers) as pool:
        results = pool.map(processing_func, pdf_files)
    
    results = [item for item in results if item]  # Filter out any None results

    return results

def predict_lang(text):
    """
    Detect the language of the given text using the langdetect library.
    
    Args:
    text (str): Text to detect the language of.
    
    Returns:
    str: Detected language or an empty string if detection fails.
    """
    try:
        return langdetect.detect(text)
    except Exception as e:
        return ""

def post_process_fast_text(predictions, pos_label = '__label__has_latex'):
    """
    Post-process predictions from the fastText model.
    
    Args:
    predictions (list of tuples): List of tuples containing labels and scores from fastText.
    pos_label (str): The label corresponding to positive class predictions.
    
    Returns:
    numpy.array: Processed scores as a numpy array.
    """
    score_ft = [float(item[1]) if item[0] == pos_label else 1-float(item[1]) for item in predictions]
    return np.array(score_ft)

def load_model(model_path, device):
    """
    Load a PyTorch model from the specified path and transfer it to the given device.
    
    Args:
    model_path (str): Path to the model's state dictionary.
    device (torch.device): The device to load the model onto (e.g., CPU or GPU).
    
    Returns:
    torch.nn.Module: The loaded and initialized PyTorch model.
    """
    model = CustomMobileNetV3(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()
    model = model.to(device)
    return model

def main(input_dir, output_dir, batch_size, model1_path, model2_path, fast_text_path, output_file):
    """
    Main function to run the machine learning pipeline on PDF files.
    
    Args:
    input_dir (str): Directory containing PDF files to process.
    output_dir (str): Directory to save output files.
    batch_size (int): Batch size for data loading.
    model1_path (str): Path to the first model (latex visual classifier).
    model2_path (str): Path to the second model (table classifier).
    fast_text_path (str): Path to the fastText model (text-based latex classifier).
    output_file (str): Name of the output CSV file where results will be stored.
    """
    print("Preparing pdf dataset\n")
    all_files = list(pathlib.Path(input_dir).glob("*.pdf"))
    results = prepare_dataset(all_files, output_dir)
    dataset = PairedImageDataset(results)
    print(f"{len(results)} file preprocessed")
    print(f"{len(dataset)} pages are found")
    data_loader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models onto the specified device
    model1 = load_model(model1_path, device)
    model2 = load_model(model2_path, device)
    fast_text_model = fasttext.load_model(fast_text_path)

    # Initialize lists to store results
    latex_scores_vis = []
    tab_scores_vis = []
    latex_scores_text= []
    languages = []
    file_names = []
    text_length = []

    # Perform inference without gradient calculation
    with torch.no_grad():
        # Iterate over data batches using tqdm for progress visualization
        for file_name, image1, image2, text in tqdm(data_loader):
            image1, image2 = image1.to(device), image2.to(device)
            outputs1 = model1(image1)
            outputs2 = model2(image2)
            outputs3 = [fast_text_model.predict(item.replace('\n', ' ')) for item in text]
            lang = [predict_lang(item) for item in text]
            scores_latex = F.softmax(outputs1, dim=1)
            scores_tab = F.softmax(outputs2, dim=1)
            length_stats = [len(item) for item in text]
            file_names.extend(file_name)
            latex_scores_vis.extend(scores_latex.cpu().numpy())
            tab_scores_vis.extend(scores_tab.cpu().numpy())
            latex_scores_text.extend(outputs3)
            languages.extend(lang)
            text_length.extend(length_stats)
    
    # Post-process fastText results
    latex_scores_text_processed = post_process_fast_text(np.array(latex_scores_text).squeeze())
    latex_scores_vis = np.array(latex_scores_vis)[..., 0]
    tab_scores_vis = np.array(tab_scores_vis)[..., 0]
    
    # Combine all results into a DataFrame and save as CSV
    df = pd.DataFrame(zip(file_names, latex_scores_vis, latex_scores_text_processed, tab_scores_vis, languages, text_length))
    df.columns = ['page_name', 'latex_visual_score', 'latex_textual_score', 'table_bbs_score', 'language_detected', 'text_length']        
    
    df.to_csv(os.path.join(output_dir, output_file), index=False)
    print(f"Results saved to {os.path.join(output_dir, output_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML pipeline on PDF files.")
    parser.add_argument("input_dir", type=str, help="Directory with PDF files")
    parser.add_argument("output_dir", type=str, help="Directory to save preprocessed files for ML pipeline")
    parser.add_argument("batch_size", type=int, help="Batch size for the pipeline")
    parser.add_argument("model1_path", type=str, help="Path to the latex visual classifier model")
    parser.add_argument("model2_path", type=str, help="Path to the table classifier model")
    parser.add_argument("fast_text_path", type=str, help="Path to the fast_text model (text-based latex classifier)")
    parser.add_argument("output_file", type=str, help="Name of the file (with .csv extension) to store the inference results")

    args = parser.parse_args()
    print(args)

    main(args.input_dir, args.output_dir, args.batch_size, args.model1_path, args.model2_path, args.fast_text_path, args.output_file)
