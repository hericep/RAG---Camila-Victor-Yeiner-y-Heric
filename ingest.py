import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

# Cargar las variables del archivo .env
load_dotenv()

# Configuración de conexión a PGVector en Docker
CONNECTION_STRING = "postgresql+psycopg2://rag_user:rag_password@localhost:5432/rag_db"
COLLECTION_NAME = "my_rag_docs"

def ingest_documents():
    # 1. Cargar el documento de texto local
    loader = TextLoader("sample.txt")
    docs = loader.load()

    # 2. Dividir el texto en fragmentos (Chunking) para evitar pérdida de contexto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Documento dividido en {len(chunks)} fragmentos.")

    # 3. Inicializar el modelo de embeddings oficial y actualizado de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # 4. Generar embeddings y almacenar en PostgreSQL con PGVector
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=chunks,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
    )
    print("¡Documentos vectorizados con Google y guardados en PostgreSQL!")

if __name__ == "__main__":
    ingest_documents()