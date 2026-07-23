"""Hybrid retrieval using Chroma, HuggingFace embeddings, BM25, and RRF."""

import logging
from functools import lru_cache
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import Config
from embeddings import get_embeddings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_vectorstore() -> Chroma:
    """Load the Chroma vector store and embedding model once per app process."""

    logger.info("Loading ChromaDB...")
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=Config.DB_PATH,
        collection_name=Config.COLLECTION_NAME,
        embedding_function=embeddings,
    )


@lru_cache(maxsize=1)
def build_retrievers(_vectorstore: Chroma):
    """Build independent BM25 and vector retrievers."""

    logger.info("Building Independent Retrievers for RRF...")
    data = _vectorstore.get()
    docs = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(data["documents"], data["metadatas"])
    ]
    logger.info("Loaded %s documents for BM25 Index", len(docs))

    pool_size = Config.TOP_K * 3
    vector_retriever = _vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": Config.SIMILARITY_THRESHOLD,
            "k": pool_size,
        },
    )

    if not docs:
        logger.warning("Database is empty. BM25 will be skipped.")
        return None, vector_retriever

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = pool_size
    return bm25_retriever, vector_retriever


class HybridRetriever:
    """Coordinates vector retrieval, BM25 retrieval, RRF, and filtering."""

    def __init__(self) -> None:
        self.vectorstore = load_vectorstore()
        self.bm25_retriever, self.vector_retriever = build_retrievers(self.vectorstore)

    def reciprocal_rank_fusion(self, doc_lists: list[list[Document]], c: int = 60) -> list[Document]:
        """Fuse retrieved documents using the original RRF algorithm."""

        fused_scores = {}
        doc_map = {}

        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list):
                doc_id = doc.page_content
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
                    fused_scores[doc_id] = 0.0
                fused_scores[doc_id] += 1 / (rank + c)

        reranked_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_map[doc_id] for doc_id, score in reranked_docs]

    def retrieve(self, query: str, semester_filter: Optional[int] = None) -> list[Document]:
        """Retrieve top documents for a query with optional semester filtering."""

        try:
            try:
                vector_docs = self.vector_retriever.invoke(query)
            except Exception as exc:
                logger.error("Vector retrieval error: %s", exc, exc_info=True)
                vector_docs = []

            try:
                bm25_docs = self.bm25_retriever.invoke(query) if self.bm25_retriever else []
            except Exception as exc:
                logger.error("BM25 retrieval error: %s", exc, exc_info=True)
                bm25_docs = []

            fused_docs = self.reciprocal_rank_fusion([vector_docs, bm25_docs])

            if semester_filter is not None:
                fused_docs = [
                    doc
                    for doc in fused_docs
                    if str(doc.metadata.get("semester")) == str(semester_filter)
                ]

            return fused_docs[: Config.TOP_K]
        except Exception as exc:
            logger.error("Retrieval error: %s", exc, exc_info=True)
            return []
