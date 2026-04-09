import json
from pathlib import Path
from typing import List, Dict, Any

from rag_engine import ProvenanceGraphRAG, RetrievalResult, RAGConfig


def load_builtin_corpus() -> tuple[list[str], list[dict]]:
    """Завантажує вбудований тестовий корпус, якщо файл test_data.txt існує поряд зі скриптом."""
    corpus_path = Path(__file__).with_name("test_data.txt")
    if not corpus_path.exists():
        return [], []
    text = corpus_path.read_text(encoding="utf-8")
    return [text], [{"source": corpus_path.name}]


def build_engine() -> ProvenanceGraphRAG:
    """Створює і готує RAG-двигун для експериментів.

    Використовує вбудований корпус test_data.txt, якщо він є.
    За потреби сюди можна додати завантаження інших документів.
    """
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    builtin_texts, builtin_metadatas = load_builtin_corpus()
    texts.extend(builtin_texts)
    metadatas.extend(builtin_metadatas)

    engine = ProvenanceGraphRAG(config=RAGConfig())
    if texts:
        engine.process_documents(texts, metadatas)
    return engine


def result_to_row(query: str, mode: str, result: RetrievalResult, answer: str) -> Dict[str, Any]:
    """Перетворює RetrievalResult на один рядок з основними метриками для CSV/JSON."""
    return {
        "query": query,
        "mode": mode,
        "confidence_score": round(result.confidence_score, 4),
        "confidence_band": result.confidence_band,
        "context_docs": len(result.context_docs),
        "graph_expanded_count": result.graph_expanded_count,
        "unique_sources": result.unique_sources,
        "avg_graph_centrality": round(result.avg_graph_centrality, 4),
        "query_alignment_score": round(result.query_alignment_score, 4),
        "top_match_score": round(result.top_match_score, 4),
        "answer_evidence_score": round(result.answer_evidence_score, 4),
        "answer_quality_score": round(result.answer_quality_score, 4),
        "answer_len_chars": len(answer or ""),
    }


def snapshot_provenance(query: str, mode: str, result: RetrievalResult, answer: str) -> Dict[str, Any]:
    """Готує компактний снапшот provenance та контексту для аналізу у дипломі."""
    docs_payload = []
    for doc in result.context_docs:
        docs_payload.append(
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "source": doc.metadata.get("source"),
                "metadata": {k: v for k, v in doc.metadata.items()},
                "content_preview": (doc.page_content[:600] + "...") if len(doc.page_content) > 600 else doc.page_content,
            }
        )

    return {
        "query": query,
        "mode": mode,
        "answer": answer,
        "confidence_score": result.confidence_score,
        "confidence_band": result.confidence_band,
        "metrics": {
            "context_docs": len(result.context_docs),
            "graph_expanded_count": result.graph_expanded_count,
            "unique_sources": result.unique_sources,
            "avg_graph_centrality": result.avg_graph_centrality,
            "query_alignment_score": result.query_alignment_score,
            "top_match_score": result.top_match_score,
            "answer_evidence_score": result.answer_evidence_score,
            "answer_quality_score": result.answer_quality_score,
        },
        "context_docs": docs_payload,
        "provenance_records": result.provenance_records,
        "answer_citations": result.answer_citations,
        "answer_supporting_chunk_ids": result.answer_supporting_chunk_ids,
    }


def main() -> None:
    # Набір запитів для експериментів. Додайте/змініть їх під власний корпус.
    queries: List[str] = [
        "Що таке штучний інтелект?",
        "Які види машинного навчання існують?",
        "Скільки літер в українській абетці?",
    ]

    k = 3
    hop_limit = 1

    engine = build_engine()
    if not engine.documents:
        print("[WARN] Engine has no documents. Додайте корпус або test_data.txt для більш репрезентативних тестів.")

    rows: List[Dict[str, Any]] = []
    provenance_snapshots: List[Dict[str, Any]] = []

    for query in queries:
        # З графом
        with_graph = engine.retrieve_with_graph(query, k=k, hop_limit=hop_limit)
        answer_with = engine.generate_answer_with_provenance(query, with_graph)
        engine.calibrate_confidence_with_answer(query, with_graph, answer_with)

        # Без графа
        without_graph = engine.retrieve_without_graph(query, k=k)
        answer_without = engine.generate_answer_with_provenance(query, without_graph)
        engine.calibrate_confidence_with_answer(query, without_graph, answer_without)

        rows.append(result_to_row(query, "with_graph", with_graph, answer_with))
        rows.append(result_to_row(query, "without_graph", without_graph, answer_without))

        provenance_snapshots.append(snapshot_provenance(query, "with_graph", with_graph, answer_with))
        provenance_snapshots.append(snapshot_provenance(query, "without_graph", without_graph, answer_without))

    # Записуємо результати в JSONL для подальшої обробки (наприклад, у ноутбуці)
    results_path = Path("experiments_metrics.jsonl")
    with results_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] Метрики збережено в {results_path.resolve()}")

    provenance_path = Path("experiments_provenance.json")
    with provenance_path.open("w", encoding="utf-8") as f:
        json.dump(provenance_snapshots, f, ensure_ascii=False, indent=2)
    print(f"[OK] Снапшоти provenance збережено в {provenance_path.resolve()}")


if __name__ == "__main__":
    main()
