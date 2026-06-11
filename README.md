# PGVector RAG System with LangGraph

Este repositorio implementa un sistema RAG utilizando LangGraph, LangChain y PostgreSQL con PGVector.
Es una herramienta creada por Yeiner Arwawingumu Zapata Vallejo, Heric Francisco Vargas Cabiativa, Maria Camila Castro Porras y Victor Daniel Diaz Reyes para el curso "Introduction to Intelligent Systems".

## Pasos para probarlo
1. Levantar la base de datos: `docker-compose up -d`
2. Instalar dependencias: `pip install -r requirements.txt`
3. Crear un archivo `.env` con tu `GOOGLE_API_KEY=la clave de Google Studio AI`.
4. Seguir las instrucciones en consola.

## Fase 1: Ingestión (Ingestion Process)
El script `ingest.py` lee un archivo local, lo divide en fragmentos de 500 caracteres (evitando pérdida de contexto) y los guarda como vectores (embeddings) en PGVector.

## Fase 2: Ejecución de Consulta (Query Process)
El script `query.py` inicializa un grafo de estado (StateGraph). Recibe el prompt del usuario, recupera los fragmentos más relevantes y genera una respuesta fundamentada en el sample.
