# Multimodal RAG Learning Project

A Retrieval-Augmented Generation (RAG) system that processes PDF documents with both text and images, enabling question-answering over multimodal content using CLIP embeddings and Google's Gemini model.

## Overview

This project implements a multimodal RAG pipeline that:
- Extracts both text and images from PDF documents
- Uses CLIP to create unified embeddings for text and images
- Stores embeddings in a FAISS vector database for efficient similarity search
- Retrieves relevant context (both text and images) based on user queries
- Generates answers using Google's Gemini model with multimodal context




## Requirements

- Python 3.8+
- Google API Key (for Gemini model)

## Installation

1. Clone this repository or navigate to the project directory:
```bash
cd rag_learning
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

1. Place your PDF file in the project directory (or update the `pdf_path` variable in the notebook)

2. Open and run the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

3. Execute all cells to:
   - Load the CLIP model
   - Process the PDF (extract text and images)
   - Create embeddings and build the vector store
   - Initialize the Gemini model

4. Query the system using the `multimodal_pdf_rag_pipeline()` function:
```python
answer = multimodal_pdf_rag_pipeline("Your question here")
print(answer)
```

## Project Structure

```
rag_learning/
├── main.ipynb              # Main Jupyter notebook with the RAG pipeline
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (create this)
└── mypdf.pdf # Example PDF file
```

## Technologies Used

- **PyMuPDF (fitz)**: PDF processing
- **CLIP (OpenAI)**: Multimodal embeddings
- **FAISS**: Vector similarity search
- **LangChain**: RAG framework and document processing
- **Google Gemini**: Multimodal LLM
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library


## License

This project is for educational purposes.

## Author

Hossein Zolfaghari

