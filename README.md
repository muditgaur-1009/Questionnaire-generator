# RAG Questionnaire Generator

## Overview

The RAG (Retrieval-Augmented Generation) Questionnaire Generator is a Streamlit-based web application that automatically generates diverse and thought-provoking questions from uploaded PDF documents. It uses Google's Generative AI models and the LangChain framework to create high-quality questions across various cognitive levels.

## Features

- Upload and process multiple PDF documents
- Generate 10 questions based on the content of the uploaded PDFs
- Questions cover different types: Factual Recall, Conceptual Understanding, Critical Thinking, Application, and Problem-Solving
- User-friendly web interface built with Streamlit

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/rag-questionnaire-generator.git
   cd rag-questionnaire-generator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Google API key:
   - Create a `.env` file in the project root directory
   - Add your Google API key to the file:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`)

3. Use the sidebar to upload at least 10 PDF files

4. Click "Submit & Process" to process the PDFs

5. Click "Generate Questionnaire" to create and display the questions

## How It Works

1. The app extracts text from the uploaded PDF files
2. The text is split into chunks and embedded using Google's embedding model
3. The embeddings are stored in a FAISS index for efficient retrieval
4. When generating questions, the app retrieves relevant text chunks
5. The retrieved text is sent to a Gemini model to generate diverse questions
6. The generated questions are displayed in the Streamlit interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Google Generative AI](https://ai.google/discover/generative-ai/)
- [FAISS](https://github.com/facebookresearch/faiss)
