# 🍩 Simpsons Quote Search Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

> *"Pregúntale a Springfield"* - Un motor de búsqueda semántico para frases icónicas de Los Simpsons con RAG, evaluación automática y observabilidad de producción.

![Simpsons Quote Search Demo](docs/demo.gif)

## 📋 Tabla de Contenidos

- [🎯 Descripción](#-descripción)
- [✨ Características](#-características)
- [🏗️ Arquitectura](#️-arquitectura)
- [🚀 Inicio Rápido](#-inicio-rápido)
- [📖 Uso](#-uso)
- [📊 Evaluación](#-evaluación)
- [🔍 Observabilidad](#-observabilidad)
- [🧪 Testing](#-testing)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)
- [🛠️ Tecnologías](#️-tecnologías)
- [📈 Roadmap](#-roadmap)
- [👤 Autor](#-autor)
- [📄 Licencia](#-licencia)

## 🎯 Descripción

Este proyecto implementa un **sistema RAG (Retrieval-Augmented Generation) completo** para buscar y responder preguntas usando un corpus de diálogos de Los Simpsons. No es solo un chatbot básico - es un sistema de producción con:

- **Búsqueda híbrida**: Combina BM25 (léxica) con embeddings semánticos
- **Respuestas citadas**: Cada respuesta incluye la quote exacta y metadatos
- **Evaluación rigurosa**: Métricas de retrieval y evaluación de faithfulness
- **Observabilidad completa**: Trazabilidad, logging estructurado y monitoreo

### Ejemplo de uso:

```
Pregunta: "¿Qué frase resume mejor a Homero cuando le hablan de dieta?"

Respuesta: "You don't win friends with salad!"
- Personaje: Homer Simpson
- Episodio: S07E05 "Lisa the Vegetarian"
- Contexto: Homer burlándose de la decisión de Lisa de ser vegetariana

Métricas:
- Latencia: 245ms
- Tokens utilizados: 847
- Documentos recuperados: 5
- Score de confianza: 0.89
```

## ✨ Características

### 🔍 Retrieval Híbrido
- **BM25**: Búsqueda léxica eficiente con rank_bm25
- **Embeddings**: Vectores semánticos con sentence-transformers
- **Fusión**: Reciprocal Rank Fusion (RRF) para combinar resultados

### 🤖 RAG Avanzado
- **Respuestas contextuales**: Usa LLM para generar respuestas naturales
- **Citación precisa**: Siempre incluye la quote original y fuente
- **Metadatos ricos**: Personaje, episodio, temporada, contexto

### 📊 Evaluación Automática
- **Retrieval**: Recall@k, MRR, NDCG
- **Generación**: Faithfulness, Groundedness, Relevance
- **Experimentos**: MLflow para tracking de métricas

### 🔭 Observabilidad de Producción
- **OpenTelemetry**: Traces distribuidos
- **Logging estructurado**: Con structlog
- **Métricas**: Prometheus-compatible
- **Dashboard**: Monitoreo en tiempo real

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Gateway                          │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   BM25 Index  │     │  Vector Store   │     │   Observability │
│   (Whoosh)    │     │  (PostgreSQL +  │     │  (OpenTelemetry │
│               │     │   pgvector)     │     │   + Structlog)  │
└───────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        └───────────┬───────────┘                       │
                    ▼                                   │
            ┌───────────────┐                           │
            │  RRF Fusion   │                           │
            │  (Hybrid)     │                           │
            └───────────────┘                           │
                    │                                   │
                    ▼                                   │
            ┌───────────────┐                           │
            │    LLM        │ ◄─────────────────────────┘
            │  Generation   │
            └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   Response    │
            │  + Citations  │
            └───────────────┘
```

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.10+
- Docker & Docker Compose (opcional pero recomendado)
- API Key de OpenAI (para generación LLM)

### Instalación con Docker (Recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/cesar530/simpsons-quote-search.git
cd simpsons-quote-search

# Copiar y configurar variables de entorno
cp .env.example .env
# Editar .env con tu API key de OpenAI

# Levantar servicios
docker-compose up -d

# Ejecutar ingesta de datos
docker-compose exec api python -m ingestion.ingest

# La API estará disponible en http://localhost:8000
```

### Instalación Manual

```bash
# Clonar el repositorio
git clone https://github.com/cesar530/simpsons-quote-search.git
cd simpsons-quote-search

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o en Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# Ejecutar ingesta de datos
python -m ingestion.ingest

# Iniciar servidor de desarrollo
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📖 Uso

### API REST

```bash
# Búsqueda simple
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "frases de Homero sobre cerveza"}'

# Búsqueda con filtros
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "frases sobre el trabajo",
    "character": "Homer Simpson",
    "season_range": [1, 10],
    "top_k": 5
  }'

# Pregunta con respuesta generada (RAG)
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál es la filosofía de vida de Homero?"}'
```

### Python SDK

```python
from simpsons_search import SimpsonsSearchClient

client = SimpsonsSearchClient(base_url="http://localhost:8000")

# Búsqueda semántica
results = client.search("mejores frases de Bart")
for quote in results:
    print(f"{quote.character}: '{quote.text}'")
    print(f"  - Episodio: {quote.episode}")

# Pregunta con RAG
response = client.ask("¿Por qué Homero siempre dice D'oh?")
print(response.answer)
print(f"Fuentes: {response.citations}")
```

## 📊 Evaluación

### Métricas de Retrieval

```python
from eval.retrieval_metrics import evaluate_retrieval

metrics = evaluate_retrieval(
    queries=test_queries,
    ground_truth=relevance_judgments,
    k_values=[1, 3, 5, 10]
)

print(f"Recall@5: {metrics['recall@5']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")
print(f"NDCG@10: {metrics['ndcg@10']:.3f}")
```

### Evaluación de Respuestas

```python
from eval.response_evaluator import ResponseEvaluator

evaluator = ResponseEvaluator()

# Evaluar faithfulness (basado en documentos recuperados)
score = evaluator.evaluate_faithfulness(
    response=generated_response,
    retrieved_docs=documents
)

# Evaluar groundedness (qué tan bien fundamentada está)
groundedness = evaluator.evaluate_groundedness(
    response=generated_response,
    sources=citations
)
```

### MLflow Tracking

```bash
# Iniciar UI de MLflow
mlflow ui --port 5000

# Los experimentos se registran automáticamente
# Acceder a http://localhost:5000
```

## 🔍 Observabilidad

### Logging Estructurado

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "search_completed",
    query=query,
    num_results=len(results),
    latency_ms=latency,
    retrieval_method="hybrid"
)
```

### Métricas Prometheus

```
# Disponibles en /metrics
simpsons_search_requests_total{method="search", status="success"}
simpsons_search_latency_seconds{quantile="0.95"}
simpsons_search_tokens_used_total
simpsons_search_retrieval_docs_count
```

### OpenTelemetry Traces

Los traces se exportan automáticamente y pueden visualizarse en:
- Jaeger: `http://localhost:16686`
- O cualquier backend compatible con OTLP

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Tests con cobertura
pytest --cov=. --cov-report=html

# Tests de carga con Locust
locust -f tests/load_test.py --host=http://localhost:8000
```

## 📁 Estructura del Proyecto

```
simpsons-quote-search/
├── api/                      # API FastAPI
│   ├── __init__.py
│   ├── main.py              # Punto de entrada de la API
│   ├── routes/              # Endpoints
│   │   ├── search.py
│   │   └── health.py
│   ├── models/              # Schemas Pydantic
│   │   ├── request.py
│   │   └── response.py
│   └── dependencies.py      # Inyección de dependencias
│
├── ingestion/               # Pipeline de ingesta
│   ├── __init__.py
│   ├── ingest.py           # Script principal de ingesta
│   ├── loaders.py          # Cargadores de datos
│   └── preprocessors.py    # Preprocesamiento de texto
│
├── retrieval/               # Módulo de búsqueda
│   ├── __init__.py
│   ├── hybrid.py           # Retrieval híbrido
│   ├── bm25.py             # Índice BM25
│   ├── embeddings.py       # Embeddings semánticos
│   └── reranker.py         # Re-ranking de resultados
│
├── eval/                    # Evaluación
│   ├── __init__.py
│   ├── retrieval_metrics.py # Métricas de retrieval
│   ├── response_evaluator.py # Evaluación de respuestas
│   └── experiments.py       # Tracking de experimentos
│
├── observability/           # Observabilidad
│   ├── __init__.py
│   ├── logging.py          # Configuración de logging
│   ├── metrics.py          # Métricas Prometheus
│   └── tracing.py          # OpenTelemetry tracing
│
├── data/                    # Datos (no versionados)
│   └── .gitkeep
│
├── tests/                   # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_retrieval.py
│   └── load_test.py
│
├── simpsons_search.py       # Cliente Python
├── utils.py                 # Utilidades compartidas
├── config.py                # Configuración
├── docker-compose.yml       # Orquestación Docker
├── Dockerfile               # Imagen de la API
├── requirements.txt         # Dependencias Python
├── .env.example             # Variables de entorno ejemplo
├── .gitignore
├── LICENSE
└── README.md
```

## 🛠️ Tecnologías

| Categoría | Tecnología | Propósito |
|-----------|------------|-----------|
| **API** | FastAPI | Framework web asíncrono |
| **Vector Store** | PostgreSQL + pgvector | Almacenamiento de embeddings |
| **BM25** | rank_bm25 / Whoosh | Búsqueda léxica |
| **Embeddings** | sentence-transformers | Modelos de embeddings |
| **LLM** | OpenAI GPT | Generación de respuestas |
| **Experiment Tracking** | MLflow | Registro de experimentos |
| **Observabilidad** | OpenTelemetry + structlog | Traces y logs |
| **Métricas** | Prometheus | Monitoreo de métricas |
| **Testing** | pytest + Locust | Unit tests y load testing |
| **Containerización** | Docker + docker-compose | Despliegue |

## 📈 Roadmap

- [x] Ingesta básica de dataset
- [x] Búsqueda BM25
- [x] Embeddings semánticos
- [x] Retrieval híbrido (RRF)
- [x] API FastAPI
- [x] Evaluación de retrieval
- [x] Logging estructurado
- [ ] UI web con Streamlit
- [ ] Evaluación LLM-as-a-judge
- [ ] Fine-tuning de embeddings
- [ ] Despliegue en cloud
- [ ] Caché de embeddings
- [ ] Soporte multi-idioma

## 👤 Autor

**César Adrián Delgado Díaz**

- 🌐 Portfolio: [tu-portfolio.com](https://tu-portfolio.com)
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](https://www.linkedin.com/in/cesar-delgado-diaz)
- 🐙 GitHub: [github.com/cesar530](https://github.com/cesar530)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">
  <i>"D'oh!" - Homer Simpson</i>
</p>

<p align="center">
  ⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub ⭐
</p>
