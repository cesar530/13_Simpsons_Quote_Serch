"""
Simpsons Quote Search Engine - Utility Functions
=================================================
Author: César Adrián Delgado Díaz
License: MIT

Funciones utilitarias compartidas para todo el proyecto.
"""

import os
import re
import json
import hashlib
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

import numpy as np


# =============================================================================
# Text Processing Utilities
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normaliza texto: minúsculas, elimina acentos, espacios extra.
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto normalizado
    """
    if not text:
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar acentos
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Eliminar espacios extra
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_quote(quote: str) -> str:
    """
    Limpia una quote: elimina comillas innecesarias, corrige puntuación.
    
    Args:
        quote: Quote original
        
    Returns:
        Quote limpia
    """
    if not quote:
        return ""
    
    # Eliminar comillas al inicio y final
    quote = quote.strip('"\'""''')
    
    # Corregir espacios antes de puntuación
    quote = re.sub(r'\s+([.,!?;:])', r'\1', quote)
    
    # Corregir espacios dobles
    quote = re.sub(r'\s+', ' ', quote).strip()
    
    return quote


def tokenize_simple(text: str) -> List[str]:
    """
    Tokenización simple para BM25.
    
    Args:
        text: Texto a tokenizar
        
    Returns:
        Lista de tokens
    """
    # Normalizar
    text = normalize_text(text)
    
    # Tokenizar por espacios y puntuación
    tokens = re.findall(r'\b\w+\b', text)
    
    # Filtrar tokens muy cortos
    tokens = [t for t in tokens if len(t) > 1]
    
    return tokens


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Divide texto en chunks con overlap.
    
    Args:
        text: Texto a dividir
        chunk_size: Tamaño máximo de cada chunk
        overlap: Caracteres de solapamiento entre chunks
        
    Returns:
        Lista de chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Intentar cortar en un espacio
        if end < len(text):
            space_idx = text.rfind(' ', start, end)
            if space_idx > start:
                end = space_idx
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks


# =============================================================================
# Vector Utilities
# =============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula similitud coseno entre dos vectores.
    
    Args:
        vec1: Primer vector
        vec2: Segundo vector
        
    Returns:
        Similitud coseno (0-1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector a norma unitaria.
    
    Args:
        vec: Vector a normalizar
        
    Returns:
        Vector normalizado
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Implementa Reciprocal Rank Fusion para combinar múltiples rankings.
    
    Args:
        rankings: Lista de rankings, cada uno es una lista de (doc_id, score)
        k: Parámetro de suavizado (default 60)
        
    Returns:
        Ranking fusionado ordenado por score
    """
    rrf_scores: Dict[str, float] = {}
    
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)
    
    # Ordenar por score descendente
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results


# =============================================================================
# Data Utilities
# =============================================================================

def generate_doc_id(content: str, metadata: Optional[Dict] = None) -> str:
    """
    Genera un ID único para un documento.
    
    Args:
        content: Contenido del documento
        metadata: Metadatos opcionales
        
    Returns:
        ID hexadecimal de 16 caracteres
    """
    hash_input = content
    if metadata:
        hash_input += json.dumps(metadata, sort_keys=True)
    
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def format_episode_code(season: int, episode: int) -> str:
    """
    Formatea código de episodio (ej: S01E05).
    
    Args:
        season: Número de temporada
        episode: Número de episodio
        
    Returns:
        Código formateado
    """
    return f"S{season:02d}E{episode:02d}"


def parse_episode_code(code: str) -> Optional[Tuple[int, int]]:
    """
    Parsea código de episodio.
    
    Args:
        code: Código en formato S01E05
        
    Returns:
        Tupla (temporada, episodio) o None si no es válido
    """
    match = re.match(r'S(\d+)E(\d+)', code, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_json_file(filepath: str) -> Optional[Dict]:
    """
    Carga un archivo JSON de forma segura.
    
    Args:
        filepath: Ruta al archivo
        
    Returns:
        Contenido del archivo o None si hay error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file {filepath}: {e}")
        return None


def save_json_file(data: Any, filepath: str, indent: int = 2) -> bool:
    """
    Guarda datos a un archivo JSON.
    
    Args:
        data: Datos a guardar
        filepath: Ruta del archivo
        indent: Indentación del JSON
        
    Returns:
        True si se guardó correctamente
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON file {filepath}: {e}")
        return False


# =============================================================================
# Metrics Utilities
# =============================================================================

def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Calcula Recall@k.
    
    Args:
        retrieved_ids: IDs de documentos recuperados
        relevant_ids: IDs de documentos relevantes
        k: Número de resultados a considerar
        
    Returns:
        Recall@k (0-1)
    """
    if not relevant_ids:
        return 0.0
    
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = len(retrieved_at_k & relevant_set)
    return hits / len(relevant_set)


def calculate_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Calcula Mean Reciprocal Rank.
    
    Args:
        retrieved_ids: IDs de documentos recuperados
        relevant_ids: IDs de documentos relevantes
        
    Returns:
        MRR (0-1)
    """
    relevant_set = set(relevant_ids)
    
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    
    return 0.0


def calculate_ndcg_at_k(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """
    Calcula NDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        retrieved_ids: IDs de documentos recuperados
        relevance_scores: Diccionario de doc_id -> relevancia
        k: Número de resultados a considerar
        
    Returns:
        NDCG@k (0-1)
    """
    def dcg(scores: List[float]) -> float:
        return sum(
            (2**rel - 1) / np.log2(rank + 2)
            for rank, rel in enumerate(scores)
        )
    
    # DCG de los resultados recuperados
    retrieved_scores = [
        relevance_scores.get(doc_id, 0.0)
        for doc_id in retrieved_ids[:k]
    ]
    actual_dcg = dcg(retrieved_scores)
    
    # IDCG (DCG ideal)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


# =============================================================================
# Timing Utilities
# =============================================================================

class Timer:
    """Context manager para medir tiempo de ejecución."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def __enter__(self) -> 'Timer':
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args) -> None:
        self.end_time = datetime.now()
    
    @property
    def elapsed_ms(self) -> float:
        """Tiempo transcurrido en milisegundos."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return 0.0
    
    @property
    def elapsed_seconds(self) -> float:
        """Tiempo transcurrido en segundos."""
        return self.elapsed_ms / 1000


# =============================================================================
# Environment Utilities
# =============================================================================

def get_env_var(name: str, default: str = "", required: bool = False) -> str:
    """
    Obtiene una variable de entorno.
    
    Args:
        name: Nombre de la variable
        default: Valor por defecto
        required: Si True, lanza excepción si no existe
        
    Returns:
        Valor de la variable
        
    Raises:
        ValueError: Si required=True y la variable no existe
    """
    value = os.getenv(name, default)
    
    if required and not value:
        raise ValueError(f"Required environment variable '{name}' is not set")
    
    return value


def setup_logging(level: str = "INFO", format_json: bool = False) -> None:
    """
    Configura logging básico.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        format_json: Si True, usa formato JSON
    """
    log_format = (
        '{"time": "%(asctime)s", "level": "%(levelname)s", '
        '"module": "%(module)s", "message": "%(message)s"}'
        if format_json
        else '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# =============================================================================
# Simpsons-Specific Utilities
# =============================================================================

# Personajes principales para filtrado
MAIN_CHARACTERS = [
    "Homer Simpson",
    "Marge Simpson", 
    "Bart Simpson",
    "Lisa Simpson",
    "Maggie Simpson",
    "Ned Flanders",
    "Mr. Burns",
    "Moe Szyslak",
    "Apu Nahasapeemapetilon",
    "Krusty the Clown",
    "Chief Wiggum",
    "Comic Book Guy",
    "Milhouse Van Houten",
    "Ralph Wiggum",
    "Nelson Muntz",
    "Groundskeeper Willie",
    "Principal Skinner",
    "Patty Bouvier",
    "Selma Bouvier",
    "Grampa Simpson"
]


def normalize_character_name(name: str) -> str:
    """
    Normaliza nombre de personaje.
    
    Args:
        name: Nombre original
        
    Returns:
        Nombre normalizado
    """
    if not name:
        return "Unknown"
    
    # Limpiar espacios
    name = name.strip()
    
    # Mapeo de variaciones comunes
    name_mapping = {
        "homer": "Homer Simpson",
        "homer_simpson": "Homer Simpson",
        "marge": "Marge Simpson",
        "marge_simpson": "Marge Simpson",
        "bart": "Bart Simpson",
        "bart_simpson": "Bart Simpson",
        "lisa": "Lisa Simpson",
        "lisa_simpson": "Lisa Simpson",
        "maggie": "Maggie Simpson",
        "burns": "Mr. Burns",
        "c. montgomery burns": "Mr. Burns",
        "monty burns": "Mr. Burns",
        "ned": "Ned Flanders",
        "flanders": "Ned Flanders",
        "moe": "Moe Szyslak",
        "apu": "Apu Nahasapeemapetilon",
        "krusty": "Krusty the Clown",
        "wiggum": "Chief Wiggum",
        "cbg": "Comic Book Guy",
        "milhouse": "Milhouse Van Houten",
        "ralph": "Ralph Wiggum",
        "nelson": "Nelson Muntz",
        "willie": "Groundskeeper Willie",
        "skinner": "Principal Skinner",
        "grampa": "Grampa Simpson",
        "abe": "Grampa Simpson",
        "abe simpson": "Grampa Simpson",
    }
    
    normalized = name_mapping.get(name.lower(), name)
    
    return normalized


def is_main_character(name: str) -> bool:
    """
    Verifica si es un personaje principal.
    
    Args:
        name: Nombre del personaje
        
    Returns:
        True si es personaje principal
    """
    normalized = normalize_character_name(name)
    return normalized in MAIN_CHARACTERS


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Text processing
    'normalize_text',
    'clean_quote',
    'tokenize_simple',
    'chunk_text',
    
    # Vector utilities
    'cosine_similarity',
    'normalize_vector',
    'reciprocal_rank_fusion',
    
    # Data utilities
    'generate_doc_id',
    'format_episode_code',
    'parse_episode_code',
    'load_json_file',
    'save_json_file',
    
    # Metrics
    'calculate_recall_at_k',
    'calculate_mrr',
    'calculate_ndcg_at_k',
    
    # Timing
    'Timer',
    
    # Environment
    'get_env_var',
    'setup_logging',
    
    # Simpsons-specific
    'MAIN_CHARACTERS',
    'normalize_character_name',
    'is_main_character',
]
