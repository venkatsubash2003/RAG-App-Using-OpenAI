import openai
import langchain
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
import streamlit as st
from langchain.chains.summarize import load_summarize_chain 
from dotenv import load_dotenv
load_dotenv()

import os
st.set_page_config(page_title="Document Querying")
st.header("Document Querying Using LangChain")
os.environ["OPENAI_API_KEY"] = ""
def read_pdf(file):
    text = ""
    try:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading the PDF file: {e}")
    return text

final_docs = None
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None:
        st.write("File Name:", uploaded_file.name)
        pdf_content = read_pdf(uploaded_file)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap = 50,
            length_function = len,
            is_separator_regex = False
        )
        docs = text_splitter.create_documents([pdf_content])
        final_docs = docs
        st.write(f"The document is divided into {len(docs)} chunks")
        st.write("Sample Chunk for the given document:")
        st.write(docs[0])



llm = OpenAI(temperature=0.5,model_name="gpt-3.5-turbo-instruct")

    


# def read_doc(directory):
#     file_loader = PyPDFDirectoryLoader(directory)
#     documents = file_loader.load()
#     return documents
# docs = read_doc("documents/")
# def chunk_data(docs,chunk_size=800,chunk_overlap = 50):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
#     docs = text_splitter.split_documents(docs)
#     return docs
# documents = chunk_data(docs=docs)
embeddings = OpenAIEmbeddings()
# vectors = embeddings.embed_query("How are you?")







if final_docs:
    db = FAISS.from_documents(final_docs,embeddings)
    from langchain.chains.question_answering import load_qa_chain
    from langchain import OpenAI
    def retrieve_query(query,k=2):
        matching_results = db.similarity_search(query)
        return matching_results
    chain = load_qa_chain(llm,chain_type="stuff")
    def retrieve_answer(query):
        doc_search = retrieve_query(query)
        with get_openai_callback() as cb:

            response = chain.run(input_documents=doc_search,question=query)
            return [response,cb]
    input = st.text_input("Query :",key="input")
    result = retrieve_answer(input)
    submit = st.button("Submit the Query")

    if submit:
        st.subheader("The Query Response is:")
        st.write(result)

    
else:
    st.write("Load the document...")


    

