import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables and configure API
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an expert questionnaire generator. Your task is to generate thought-provoking questions based on the provided context. 
    Create a diverse set of questions that cover various aspects of the subject matter, including factual recall, conceptual understanding, 
    critical thinking, application, and problem-solving. Ensure the questions are clear, concise, and relevant to the context.

    For each question, specify the question type in parentheses (e.g., Factual Recall, Conceptual Understanding, etc.).

    Context: {context}

    Generate 10 questions based on the above context, formatted as follows:
    1. Question 1 (Question Type): [Your question here]
    2. Question 2 (Question Type): [Your question here]
    ...and so on.
    """    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_questionnaire(num_questions=10):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    
    all_questions = []
    for _ in range(2):  # We'll generate 2 sets of 5 questions each
        sample_doc = new_db.similarity_search("", k=1)[0]
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": [sample_doc], "question": "Generate questions"},
            return_only_outputs=True
        )
        
        questions = response["output_text"].split("\n")
        all_questions.extend([q.strip() for q in questions if q.strip() and q[0].isdigit()])
    
    return all_questions[:num_questions]

def main():
    st.set_page_config("RAG Questionnaire Generator")
    st.header("RAG Questionnaire Generator using Gemini💁")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload at least 10 PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if len(pdf_docs) < 10:
                st.error("Please upload at least 10 PDF files.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")

    if st.button("Generate Questionnaire"):
        if not os.path.exists("faiss_index"):
            st.error("Please process PDFs first before generating the questionnaire.")
        else:
            with st.spinner("Generating questionnaire..."):
                questions = generate_questionnaire(num_questions=10)
                st.subheader("Generated Questionnaire:")
                for question in questions:
                    # Split the question into number, type, and content
                    parts = question.split(":", 1)
                    if len(parts) == 2:
                        number_type, content = parts
                        number, q_type = number_type.split(" ", 1)
                        st.markdown(f"**{number}** {q_type}:")
                        st.write(content.strip())
                        st.write("")  # Add a blank line for spacing

if __name__ == "__main__":
    main()