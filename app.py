import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle
import os

st.header("Chat with PDF...")

pdf = st.file_uploader("Upload your PDF here",type="pdf")
load_dotenv()

if pdf is not None:
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text=text)

    # embeddings
    # store_name = pdf.name[:-4]
    # if os.path.exists(f"{store_name}.pkl"):
    #     with open(f"{store_name}.pkl","rb") as f:
    #         vectorstore = pickle.load(f)
    # else:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks,embedding=embeddings)
    #     with open(f"{store_name}.pkl","wb") as f:
    #         pickle.dump(vectorstore,f)

    query = st.text_input("Ask question about your PDF file:")

    if query: 
            docs = vectorstore.similarity_search(query=query,k=3)

            llm = OpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)



