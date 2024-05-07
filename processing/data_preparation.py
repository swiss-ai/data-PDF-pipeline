import pathlib
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
import os

from .pdf_preprocessing import *


transform_table = transforms.Compose([
    transforms.Resize((792, 612)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.9476, 0.9188, 0.9712], std=[0.2228, 0.2731, 0.1673]), # tables
    #transforms.Normalize(mean=[0.9389, 0.9424, 0.9421], std=[0.1816, 0.1679, 0.1702]), # latex
])


transform_latex = transforms.Compose([
    transforms.Resize((792, 612)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.9476, 0.9188, 0.9712], std=[0.2228, 0.2731, 0.1673]), # tables
    transforms.Normalize(mean=[0.9389, 0.9424, 0.9421], std=[0.1816, 0.1679, 0.1702]), # latex
])


def prepare_pdf_data(pdf_file, output_dir=None, print_info = False):
    file_name = pathlib.Path(pdf_file).name.split('.')[0]
    nspace = pathlib.Path(output_dir) 
    write_name = nspace/file_name
    if not write_name.exists(): # Assuming that it has been already processed
        try:
            image_dir = (write_name)/"images_raw"
            boxes_dir = (write_name)/"boxes_raw"
            text_dir = (write_name)/"text_raw"
            image_dir.mkdir(exist_ok=True, parents=True)
            boxes_dir.mkdir(exist_ok=True, parents = True)
            text_dir.mkdir(exist_ok=True, parents= True)
            convert_pdf_to_png(pdf_file, image_dir)
            draw_blocks_and_words_on_image(pdf_file, boxes_dir)
            record_pdf_text(pdf_file, text_dir)
        except Exception as e:
            print(f"Error opening {pdf_file}")
            return None
    if print_info:
            print(f"Prepared {pdf_file} file")
    return nspace/file_name
    

class PairedImageDataset(Dataset):
    
    def __init__(self, directories, transform_latex=transform_latex, transform_table=transform_table):
        """
        Initializes the dataset.
        
        :param directories: List of directories, each corresponds to processed PDF data.
        :param transform_latex: Transform to apply to latex images.
        :param transform_table: Transform to apply to table images.
        """
        self.transform_latex = transform_latex
        self.transform_table = transform_table

        # Flatten the list of images, boxes, and texts from all directories
        self.data = []
        for directory in directories:
            img_dir = pathlib.Path(directory) / "images_raw"
            box_dir = pathlib.Path(directory) / "boxes_raw"
            text_dir = pathlib.Path(directory) / "text_raw"

            # Assuming filenames without extension are the same across images and texts
            page_names = [file.stem for file in img_dir.glob('*.png')]
            
            for page_name in page_names:
                img_path = img_dir / f"{page_name}.png"
                box_path = box_dir / f"{page_name}.png"
                text_path = text_dir / f"{page_name}.txt"

                self.data.append((page_name, img_path, box_path, text_path))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        page_name, img_path, box_path, text_path = self.data[idx]
        
        # Load and transform images
        image = Image.open(img_path).convert("RGB")
        boxed_image = Image.open(box_path).convert("RGB")
        
        if self.transform_latex:
            image = self.transform_latex(image)
        if self.transform_table:
            boxed_image = self.transform_table(boxed_image)
        
        # Load text
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        return page_name, image, boxed_image, text

# Old version which didn't support providing multiple directories
# class PairedImageDataset(Dataset):
    
#     def __init__(self, pdf_dir,
#      transform_latex=transform_latex,
#      transform_table=transform_table):
#         self.img_dir = pathlib.Path(pdf_dir)/"images_raw"
#         self.box_dir = pathlib.Path(pdf_dir)/"boxes_raw"
#         self.text_dir = pathlib.Path(pdf_dir)/"text_raw"
#         self.transform_latex = transform_latex
#         self.transform_table = transform_table

#         self.page_names = self._get_page_names()
        
#     def _get_page_names(self):
#         return [file.name for file in list(self.img_dir.rglob('*.png'))]
        
#     def __len__(self):
#         return len(self.page_names)
    
#     def __getitem__(self, idx):
#         page_name = self.page_names[idx]
#         img_path = self.img_dir/page_name
#         box_path = self.box_dir/page_name
#         text_path = (self.text_dir/page_name).with_suffix('.txt')
        
#         image = Image.open(img_path).convert("RGB")
#         boxed_image = Image.open(box_path).convert("RGB")
#         text = text_path.read_text()
        
        
#         image = self.transform_latex(image)
#         boxed_image = self.transform_table(boxed_image)
           
#         return image, boxed_image, text
    
    