# RAG-Augmented Automotive Policy Decision Support

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19003899.svg)](https://doi.org/10.5281/zenodo.19003899)

This repository contains the codebase for a hybrid framework that combines machine learning sales forecasting (XGBoost and LSTM) with a Retrieval-Augmented Generation (RAG) pipeline to detect and explain anomalies in the Chinese automotive market.

## Project Overview
The framework detects anomalous sales deviations and automatically explains the underlying real-world policy causes by retrieving and synthesizing context from a corpus of automotive policy documents.
- **Forecasting Models**: XGBoost and stacked LSTM with lag features.
- **RAG Pipeline**: Built with ChromaDB (for vector indexing) and Ollama running `qwen3.5:4b` (for local LLM inference), engineered without LangChain for maximal portability.

## Dataset Requirements
This code requires the **SRNI-CAR Dataset** (Ding et al., 2023). Before running the code, ensure the following two files are placed in this directory alongside the notebook:
- `Sales.csv` (contains the monthly market sales data)
- `Information.csv` (contains the corpus of news and policy documents)

## Setup & Dependencies
1. Install Python 3.8+
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ollama Setup**: The RAG generation relies on a local Ollama instance running the Qwen 3.5 4B model.
   - Install Ollama from [ollama.com](https://ollama.com)
   - Pull the Chinese-proficient language model:
     ```bash
     ollama pull qwen3.5:4b
     ```

## How to Run
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `training.ipynb` and execute the cells sequentially. The notebook will automatically:
   - Load and preprocess the SRNI-CAR dataset
   - Train the XGBoost and LSTM time-series forecasting models
   - Identify anomalous sales months based on residual thresholding
   - Initialize the local ChromaDB vector store with chunked policy documents
   - Retrieve relevant policy context and generate grounded explanations using Ollama

## Citation
If you use this code or the accompanying paper, please cite:

```bibtex
@misc{turja2026rag,
  author    = {Turja, Md Tanvir Hasan},
  title     = {A RAG-Augmented Hybrid Forecasting Framework for Policy-Driven 
               Sales Anomaly Explanation: Evidence from the Chinese Automotive 
               Market (2016--2022)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19003899},
  url       = {https://doi.org/10.5281/zenodo.19003899}
}
```

## License
This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
