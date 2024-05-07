# Repo for PDF parsing pipeline

Suggested methodology: estimate presence of scientific content (LaTeX) or tables on the pdf page (as these are the major reasons why simple parsers might result in a corrupted data), and route the parsing to the appropriate tool. 
Suggested tools for simple layouts:
- PyMuPDF, etc. (40 ms processing per page)
Suggested tools for complex layouts:
- Nougat (https://github.com/facebookresearch/nougat), most optimal open source model, still hallucinates sometimes and quite slow
- Mathpix (https://mathpix.com/) best-in-quality, quite fast, but commercial

Currently there are two ways implemented to estimate complexity of the pdf document 
- LLM based (based on api from GPT-4 or Claude 3) (around 1 page per second)
- Light-weight visual/textual models (trained for latex and table detection from visual/textual signal) (whole pipeline around 30 pages per second on GPU)

To perform llm-based inference consult `notebooks/Example_llm.ipynb`
\nFor the light-weight, scalable annotation, download checkpoints of the models from https://drive.google.com/file/d/1cQCvW4JdETfO55zVvq6m5vEnDPaTwWzn/
and run script `infer_structure.py`

