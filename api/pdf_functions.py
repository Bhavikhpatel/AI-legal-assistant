from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFFunctions:
    def __init__(self):
        pass

    def pdf_to_chunks(self, pdf_paths):
        """Convert PDF to chunks"""
        documents = []
        for x in pdf_paths:
            loader = PyPDFLoader(x)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300
        )
        chunks = splitter.split_documents(documents)

        return chunks
