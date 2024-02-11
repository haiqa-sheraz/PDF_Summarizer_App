import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# model and tokenization
checkpoint = "LaMini-FlanT5-248M"
model = T5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = T5Tokenizer.from_pretrained(checkpoint, legacy=False)

def llm_pipeline(filepath, checkpoint="LaMini-FlanT5-248M"):
    # File loader and preprocessing
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    # LM pipeline
    tokenizer = T5Tokenizer.from_pretrained(checkpoint, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    pipe_sum = pipeline(
        'summarization',
        model=model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )
    result = pipe_sum(final_texts)
    summary_text = result[0]['summary_text']
    return summary_text

# function to display pdf of given file
def displayPDF(upl_file):
    
    # Read file as bytes:
    bytes_data = upl_file.getvalue()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")

    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="650" height="600" type="application/pdf"></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


    
# streamlit code 
st.set_page_config(layout="wide", page_title="PDF Summarization App")

def main():
    st.title('Document Summarization App Using LaMini-FlanT5-248M')
    
    uploaded_file = st.file_uploader(
        "Upload file",
        type=["pdf"],
        help="Only PDF files are supported"
    )
    
    if uploaded_file is not None:
        if st.button('Summarize'):
            col1, col2 = st.columns(2)
            filepath = "data/" + uploaded_file.name
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info('Uploaded PDF')
                displayPDF(uploaded_file)
                
            with col2:
                st.info('Document Summary')
                summary = llm_pipeline(filepath)
                st.success(summary)
                
if __name__ == '__main__':
    main()
