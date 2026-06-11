import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate

# Cargar la key de la API de Google Generative AI desde el archivo .env
load_dotenv()

# Configuración de conexión a PGVector en Docker
CONNECTION_STRING = "postgresql+psycopg2://rag_user:rag_password@localhost:5432/rag_db"
COLLECTION_NAME = "my_rag_docs"

# 1. Definir la estructura del Estado (User States) que viajará por el Grafo
class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str

# 2. Configurar el cargador de vectores (Retriever)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = PGVector(
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    embeddings=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Configurar el modelo de lenguaje, como tengo sólo acceso a Gemini ese uso
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 4. Nodo 1: Recuperación de Contexto (retrieving the most relevant context vectors to answer a user's prompt)
def retrieve_node(state: RAGState):
    print("--- RECUPERANDO EL CONTEXTO DEL SAMPLE ---")
    docs = retriever.invoke(state["question"])
    return {"context": [doc.page_content for doc in docs]}

# 5. Nodo 2: Generación de Respuesta
def generate_node(state: RAGState):
    print("--- GENERANDO RESPUESTA CON GEMINI LIMITADO AL CONTEXTO DEL SAMPLE ---")
    context_str = "\n".join(state["context"])
    
    # Prompt estricto para evitar alucinaciones
    prompt = PromptTemplate.from_template(
        "Responde a la pregunta usando estrictamente el siguiente contexto.\n\n"
        "Contexto: {context}\n\nPregunta: {question}\n\nRespuesta:"
    )
    
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": state["question"]})
    return {"answer": response.content}

# 6. Construcción del Grafo de Flujo Modular
workflow = StateGraph(RAGState)

# Registrar los módulos (nodos)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Trazar las conexiones de control
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compilar la aplicación final
app = workflow.compile()

if __name__ == "__main__":
    print("=" * 50)
    print("RAG de Heric, Camila, Yeiner y Victor")
    print("Escribe tu pregunta y presiona Enter.")
    print("Para salir del chat, escribe 'salir', 'exit' o 'quit'.")
    print("=" * 50 + "\n")
    
    while True:
        # Captura la pregunta
        test_question = input("\n Tu pregunta: ")
        
        # Condición de salida para que dejen de preguntar
        if test_question.strip().lower() in ["salir", "exit", "quit"]:
            print("\n Gracias por usar nuestro RAG UwU.")
            break
            
        # Si el usuario presiona enter por error sin escribir nada, ignorarlo
        if not test_question.strip():
            continue
            
        try:
            # Ejecutar el flujo del grafo enviando la pregunta de la consola
            result = app.invoke({"question": test_question, "context": [], "answer": ""})
            
            print("\n--- RESULTADO FINAL ---")
            print(f"Respuesta de nuestro RAG: {result['answer']}")
            print("-" * 50)
            
        except Exception as e:
            print(f"\n Ocurrió un error al procesar la pregunta: {e}")