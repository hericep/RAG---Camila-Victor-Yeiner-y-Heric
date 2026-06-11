# PGVector RAG System with LangGraph

Este repositorio implementa un sistema RAG modular de 3ra generación utilizando LangGraph, LangChain y PostgreSQL con PGVector.

## Requisitos
1. Levantar la base de datos: `docker-compose up -d`
2. Instalar dependencias: `pip install -r requirements.txt`
3. Crear un archivo `.env` con tu `OPENAI_API_KEY=tu_clave_aqui`.

## Fase 1: Ingestión (Ingestion Process)
El script `ingest.py` lee un archivo local, lo divide en fragmentos de 500 caracteres (evitando pérdida de contexto) y los guarda como vectores (embeddings) en PGVector.
**Output de ejemplo:**
> Documento dividido en 12 fragmentos.
> ¡Documentos vectorizados y guardados en PostgreSQL!

## Fase 2: Ejecución de Consulta (Query Process)
El script `query.py` inicializa un grafo de estado (StateGraph). Recibe el prompt del usuario, recupera los fragmentos más relevantes y genera una respuesta fundamentada.
**Output de ejemplo:**
> --- RECUPERANDO CONTEXTO ---
> --- GENERANDO RESPUESTA ---
> --- RESULTADO FINAL ---
> Pregunta: ¿Cuál es la arquitectura RAG?
> Respuesta de la IA: RAG (Retrieval-Augmented Generation) es una arquitectura que busca conocimiento relevante al momento de la consulta y lo inyecta en el contexto del LLM.