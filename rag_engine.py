import networkx as nx
import numpy as np
import re
import os
from typing import List, Dict, Any
from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from nlp_utils import normalize_tokens


@dataclass
class RetrievalResult:
    context_docs: List[Document] = field(default_factory=list)
    confidence_score: float = 0.0
    vector_scores: List[float] = field(default_factory=list)
    graph_expanded_count: int = 0
    unique_sources: int = 0
    avg_graph_centrality: float = 0.0
    query_alignment_score: float = 0.0
    top_match_score: float = 0.0
    answer_evidence_score: float = 0.0
    answer_quality_score: float = 0.0
    provenance_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confidence_factors: List[Dict[str, Any]] = field(default_factory=list)
    confidence_band: str = ""
    answer_citations: List[str] = field(default_factory=list)
    answer_supporting_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    with_graph: RetrievalResult = field(default_factory=RetrievalResult)
    without_graph: RetrievalResult = field(default_factory=RetrievalResult)
    answer_with_graph: str = ""
    answer_without_graph: str = ""


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_context_docs: int = 8
    max_graph_expansion: int = 5
    rerank_pool_size: int = 12
    min_graph_edge_weight: float = 0.6
    graph_neighbor_min_score: float = 0.32
    list_answer_min_score: float = 0.6
    extractive_min_score_general: float = 0.4
    extractive_min_score_definition: float = 0.5
    extractive_min_score_list: float = 0.55
    extractive_min_score_yes_no: float = 0.45
    extractive_min_score_factoid: float = 0.45
    max_context_chars_for_llm: int = 2200
    llm_backend: str = "local"
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0


@dataclass
class ConfidenceFactorSpec:
    name: str
    label: str
    weight: float
    description: str


class ProvenanceGraphRAG:
    """Ядро RAG-системи з графом походження та оцінкою довіри.

    Відповідає за індексацію документів, побудову графа, пошук
    з/без графового розширення, формування відповіді та обчислення
    багатофакторної оцінки довіри до результату.
    """

    CONFIDENCE_FACTORS: List[ConfidenceFactorSpec] = [
        ConfidenceFactorSpec(
            name="retrieval_similarity",
            label="Середня релевантність retrieval",
            weight=0.20,
            description="Наскільки сильними в середньому були збіги між запитом і вибраними фрагментами.",
        ),
        ConfidenceFactorSpec(
            name="score_stability",
            label="Стабільність score",
            weight=0.10,
            description="Менший розкид між score фрагментів означає більш узгоджений контекст.",
        ),
        ConfidenceFactorSpec(
            name="context_coverage",
            label="Покриття контексту",
            weight=0.04,
            description="Невелика надбавка за достатню кількість фрагментів у фінальному контексті.",
        ),
        ConfidenceFactorSpec(
            name="graph_precision",
            label="Обережність графового розширення",
            weight=0.06,
            description="Якщо контекст не надто роздутий графом, довіра вища.",
        ),
        ConfidenceFactorSpec(
            name="graph_centrality",
            label="Центральність вузлів графа",
            weight=0.02,
            description="Слабкий графовий сигнал про пов'язаність вибраних вузлів.",
        ),
        ConfidenceFactorSpec(
            name="query_alignment",
            label="Відповідність запиту",
            weight=0.18,
            description="Наскільки фінальний контекст узгоджений із запитом.",
        ),
        ConfidenceFactorSpec(
            name="top_match",
            label="Найсильніший доказ",
            weight=0.16,
            description="Наявність хоча б одного дуже релевантного фрагмента суттєво підвищує довіру.",
        ),
        ConfidenceFactorSpec(
            name="answer_evidence",
            label="Підкріплення відповіді доказами",
            weight=0.24,
            description="Чи є в контексті речення, що прямо підтримують відповідь.",
        ),
    ]

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", config: RAGConfig | None = None, llm_backend: str | None = None):
        self.config = config or RAGConfig()
        if llm_backend is not None:
            self.config.llm_backend = llm_backend
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if self.config.llm_backend == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Будь ласка, задай ключ OpenAI у змінній середовища."
                )
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise ImportError(
                    "Для використання OpenAI встанови пакет 'langchain-openai' "
                    "та клієнт 'openai', наприклад: pip install langchain-openai openai."
                ) from exc

            self.llm = ChatOpenAI(
                model=self.config.openai_model,
                temperature=self.config.openai_temperature,
            )
            print(f"Using OpenAI LLM model: {self.config.openai_model}")
        else:
            print(f"Loading local LLM model: {model_name} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=220,
                truncation=True,
                repetition_penalty=1.1,
                do_sample=False,
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.vector_store = None
        self.graph = nx.Graph()
        self.documents: List[Document] = []

    @staticmethod
    def _distance_to_similarity(score: float) -> float:
        return max(0.0, min(1.0, float(1.0 / (1.0 + max(score, 0.0)))))

    def _get_document_by_chunk_id(self, chunk_id: str) -> Document | None:
        return next((d for d in self.documents if d.metadata["chunk_id"] == chunk_id), None)

    def _select_graph_neighbors(self, chunk_id: str, hop_limit: int) -> List[tuple[str, int, float, str]]:
        if chunk_id not in self.graph or hop_limit <= 0:
            return []

        visited = {chunk_id}
        frontier = [(chunk_id, 0)]
        selected: List[tuple[str, int, float, str]] = []

        while frontier and len(selected) < self.config.max_graph_expansion:
            current_id, depth = frontier.pop(0)
            if depth >= hop_limit:
                continue

            neighbors = []
            for neighbor_id in self.graph.neighbors(current_id):
                if neighbor_id in visited:
                    continue
                edge_data = self.graph.get_edge_data(current_id, neighbor_id) or {}
                edge_type = edge_data.get("type", "semantic")
                edge_weight = float(edge_data.get("weight", 0.0))
                edge_priority = 2.0 if edge_type == "sequential" else 1.0
                neighbors.append((neighbor_id, depth + 1, edge_weight, edge_priority, edge_type))

            neighbors.sort(key=lambda item: (item[3], item[2]), reverse=True)

            for neighbor_id, next_depth, edge_weight, _, edge_type in neighbors[:2]:
                visited.add(neighbor_id)
                frontier.append((neighbor_id, next_depth))
                selected.append((neighbor_id, next_depth, edge_weight, edge_type))
                if len(selected) >= self.config.max_graph_expansion:
                    break

        return selected

    def _trim_context(self, docs: List[Document], scores: List[float]) -> tuple[List[Document], List[float]]:
        ranked = list(zip(docs, scores))
        ranked.sort(key=lambda item: item[1], reverse=True)
        ranked = ranked[: self.config.max_context_docs]
        return [doc for doc, _ in ranked], [score for _, score in ranked]

    @staticmethod
    def _normalize_tokens(text: str) -> List[str]:
        return normalize_tokens(text)

    def _query_tokens(self, query: str) -> set[str]:
        return {
            token for token in self._normalize_tokens(query)
            if len(token) > 2 and token not in {"це", "що", "яка", "який", "яке", "таке", "про", "для"}
        }

    def _is_definition_query(self, query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in (" це", "що таке", "визначення"))

    def _query_mode(self, query: str) -> str:
        lowered = query.lower().strip()
        if self._is_definition_query(query):
            return "definition"
        if lowered.startswith("скільки") or lowered.startswith("хто") or lowered.startswith("коли"):
            return "factoid"
        if lowered.startswith("які") or lowered.startswith("яка") or lowered.startswith("які є"):
            return "list"
        if lowered.startswith("чи "):
            return "yes_no"
        if lowered.startswith("як "):
            return "process"
        return "general"

    def _lexical_overlap_score(self, query: str, doc: Document) -> float:
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return 0.0
        doc_tokens = set(self._normalize_tokens(doc.page_content))
        overlap = len(query_tokens & doc_tokens)
        return overlap / len(query_tokens)

    def _phrase_match_bonus(self, query: str, doc: Document) -> float:
        query_tokens = list(self._query_tokens(query))
        if len(query_tokens) < 2:
            return 0.0
        phrase = " ".join(query_tokens)
        content = doc.page_content.lower()
        if phrase in content:
            return 0.2
        return 0.0

    def _sentence_evidence_score(self, query: str, sentence: str) -> float:
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return 0.0
        sentence_lower = sentence.lower()
        sentence_tokens = set(self._normalize_tokens(sentence))
        overlap = len(query_tokens & sentence_tokens)
        lexical_score = overlap / len(query_tokens)
        phrase_match = " ".join(query_tokens) in sentence_lower if len(query_tokens) >= 2 else False
        mode = self._query_mode(query)

        evidence_bonus = 0.0
        if mode == "definition":
            if " — це " in sentence_lower or " це " in sentence_lower or "що таке" in sentence_lower:
                evidence_bonus += 0.25
            if any(token in sentence_tokens for token in query_tokens):
                evidence_bonus += 0.1
            if len(sentence.split()) <= 28 and overlap >= max(1, len(query_tokens) - 1):
                evidence_bonus += 0.12
        elif mode == "list":
            if any(marker in sentence_lower for marker in ("типи", "види", "тополог", "класиф", ":")):
                evidence_bonus += 0.2
            if sentence_lower.count(",") >= 2:
                evidence_bonus += 0.1
            if any(marker in sentence_lower for marker in ("перш", "друг", "трет", "1)", "2)", "3)")):
                evidence_bonus += 0.15
        elif mode == "factoid":
            if any(char.isdigit() for char in sentence):
                evidence_bonus += 0.18
            if any(marker in sentence_lower for marker in ("чинний", "обирається", "року", "літер", "букв", "президент", "є ")):
                evidence_bonus += 0.16
        elif mode == "yes_no":
            if any(marker in sentence_lower for marker in ("може", "неможе", "не може", "здат", "дозволя")):
                evidence_bonus += 0.15
        elif mode == "process":
            if any(marker in sentence_lower for marker in ("етап", "крок", "процес", "спочатку", "далі", "потім")):
                evidence_bonus += 0.15

        score = 0.65 * lexical_score + evidence_bonus + (0.15 if phrase_match else 0.0)
        return min(score, 1.0)

    def _is_list_like_sentence(self, sentence: str) -> bool:
        sentence_lower = sentence.lower()
        return (
            sentence_lower.count(",") >= 2
            or any(marker in sentence_lower for marker in ("типи", "види", "тополог", "класиф", "перш", "друг", "трет", ":"))
        )

    @staticmethod
    def _clean_fragment(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip(" -:;,.\n\t")
        return cleaned

    def _extract_list_answer_from_doc(self, query: str, doc: Document) -> str | None:
        if self._query_mode(query) != "list":
            return None
        content = re.sub(r"\s+", " ", doc.page_content).strip()
        if not content:
            return None

        patterns = [
            r"[Тт]ипи[^:]{0,80}:\s*(.+)",
            r"[Вв]иди[^:]{0,80}:\s*(.+)",
            r"[Кк]ласиф[^:]{0,80}:\s*(.+)",
            r"[Тт]ополог[^:]{0,80}:\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if not match:
                continue
            tail = match.group(1)
            stop_match = re.search(r"(?:Перехід:|Слайд \d+|$)", tail)
            if stop_match:
                tail = tail[:stop_match.start()] if stop_match.start() > 0 else tail
            parts = re.split(r"(?:(?<=\.)\s+)|;", tail)
            normalized_parts = []
            for part in parts:
                piece = self._clean_fragment(part)
                if len(piece) < 6:
                    continue
                if piece.lower().startswith("перехід"):
                    continue
                normalized_parts.append(piece)
            if normalized_parts:
                return "; ".join(normalized_parts[:5]) + "."

        labeled_items = re.findall(r"([А-ЯA-ZІЇЄҐ][^:]{2,60}:\s*[^.\n]{5,140})", content)
        filtered_items = []
        for item in labeled_items:
            cleaned = self._clean_fragment(item)
            if cleaned.lower().startswith("слайд") or cleaned.lower().startswith("перехід"):
                continue
            filtered_items.append(cleaned)
        if len(filtered_items) >= 2:
            return "; ".join(filtered_items[:5]) + "."
        return None

    def _doc_evidence_score(self, query: str, doc: Document) -> float:
        if self._query_mode(query) == "list" and self._extract_list_answer_from_doc(query, doc):
            return 0.95
        sentences = re.split(r"(?<=[.!?])\s+|\n+", doc.page_content)
        best_score = 0.0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            best_score = max(best_score, self._sentence_evidence_score(query, sentence))
        return best_score

    def _meets_definition_match_threshold(self, query: str, doc: Document) -> bool:
        if not self._is_definition_query(query):
            return True
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return True
        lexical_score = self._lexical_overlap_score(query, doc)
        phrase_bonus = self._phrase_match_bonus(query, doc)
        if len(query_tokens) <= 3:
            return lexical_score >= 0.99 or phrase_bonus > 0.0
        return lexical_score >= 0.6 or phrase_bonus > 0.0

    def _score_doc_against_query(self, query: str, doc: Document, raw_score: float) -> float:
        base_similarity = self._distance_to_similarity(raw_score)
        lexical_score = self._lexical_overlap_score(query, doc)
        phrase_bonus = self._phrase_match_bonus(query, doc)
        evidence_score = self._doc_evidence_score(query, doc)
        final_score = 0.35 * base_similarity + 0.25 * lexical_score + 0.25 * evidence_score + phrase_bonus
        if self._is_definition_query(query) and lexical_score < 0.5:
            final_score *= 0.55
        return min(final_score, 1.0)

    def _definition_candidate_score(self, query: str, doc: Document) -> float:
        lexical_score = self._lexical_overlap_score(query, doc)
        phrase_bonus = self._phrase_match_bonus(query, doc)
        content = doc.page_content.lower()
        definitional_bonus = 0.0
        if " — це " in content or " це " in content or "що таке" in content:
            definitional_bonus = 0.2
        score = 0.7 * lexical_score + phrase_bonus + definitional_bonus
        return min(score, 1.0)

    def _retrieve_definition_candidates(self, query: str, k: int) -> List[tuple[Document, float]]:
        if not self.documents:
            return []
        ranked: List[tuple[Document, float]] = []
        for doc in self.documents:
            if not self._meets_definition_match_threshold(query, doc):
                continue
            score = self._definition_candidate_score(query, doc)
            if score >= 0.45:
                ranked.append((doc, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]

    def _deduplicate_docs(self, docs: List[Document], scores: List[float]) -> tuple[List[Document], List[float]]:
        best_by_chunk: Dict[str, tuple[Document, float]] = {}
        for doc, score in zip(docs, scores):
            chunk_id = doc.metadata["chunk_id"]
            current = best_by_chunk.get(chunk_id)
            if current is None or score > current[1]:
                best_by_chunk[chunk_id] = (doc, score)
        deduped = list(best_by_chunk.values())
        deduped.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in deduped], [score for _, score in deduped]

    def _compute_query_alignment(self, query: str, docs: List[Document], scores: List[float]) -> tuple[float, float]:
        query_tokens = self._query_tokens(query)
        if not docs:
            return 0.0, 0.0
        if not query_tokens:
            top_match = max(scores) if scores else 0.0
            return top_match, top_match

        per_doc_alignments = []
        for doc, score in zip(docs, scores):
            doc_tokens = set(self._normalize_tokens(doc.page_content))
            overlap = len(query_tokens & doc_tokens)
            lexical_score = overlap / len(query_tokens)
            per_doc_alignments.append(0.6 * score + 0.4 * lexical_score)

        top_match = max(per_doc_alignments) if per_doc_alignments else 0.0
        alignment = float(np.mean(per_doc_alignments)) if per_doc_alignments else 0.0
        return alignment, top_match

    def _compute_answer_evidence(self, query: str, docs: List[Document]) -> float:
        if not docs:
            return 0.0
        evidence_scores = [self._doc_evidence_score(query, doc) for doc in docs]
        if not evidence_scores:
            return 0.0
        top_evidence = max(evidence_scores)
        avg_evidence = float(np.mean(evidence_scores))
        evidence = min(1.0, 0.7 * top_evidence + 0.3 * avg_evidence)
        if self._query_mode(query) == "list":
            list_like_hits = 0
            for doc in docs:
                sentences = re.split(r"(?<=[.!?])\s+|\n+", doc.page_content)
                if any(self._is_list_like_sentence(sentence.strip()) for sentence in sentences if sentence.strip()):
                    list_like_hits += 1
            if list_like_hits == 0:
                evidence *= 0.45
            elif list_like_hits == 1:
                evidence *= 0.75
        return evidence

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[tuple[Document, float]],
        top_k: int,
    ) -> List[tuple[Document, float]]:
        if not candidates:
            return []

        reranked: List[tuple[Document, float]] = []
        for doc, raw_score in candidates:
            final_score = self._score_doc_against_query(query, doc, raw_score)
            reranked.append((doc, min(final_score, 1.0)))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_k]

    def _retrieve_candidates(self, query: str, k: int) -> List[tuple[Document, float]]:
        if self._is_definition_query(query):
            definition_candidates = self._retrieve_definition_candidates(query, k=k)
            if definition_candidates:
                return definition_candidates
        pool_k = max(k, self.config.rerank_pool_size)
        initial = self.vector_store.similarity_search_with_score(query, k=pool_k)
        reranked = self._rerank_candidates(query, initial, top_k=pool_k)
        if self._is_definition_query(query):
            strong = [
                (doc, score) for doc, score in reranked
                if self._meets_definition_match_threshold(query, doc)
            ]
            if strong:
                return strong[:k]
        return reranked[:k]

    def _extractive_answer(self, query: str, context_docs: List[Document]) -> str | None:
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return None

        best_sentence = None
        best_score = 0.0
        mode = self._query_mode(query)
        is_definition_query = mode == "definition"

        query_lower = query.lower().strip()
        focus_query = query_lower
        for prefix in ("що таке ", "що таке", "визначення ", "визначення"):
            if focus_query.startswith(prefix):
                focus_query = focus_query[len(prefix):].strip()
        focus_query = re.sub(r"[?!.:,;]+$", "", focus_query).strip()
        focus_tokens = set(self._normalize_tokens(focus_query))

        # Спочатку: для визначень із веб-режиму (Wikipedia) намагаємось
        # напряму взяти перше осмислене речення з тієї статті, чий заголовок
        # найбільше перетинається з фокусом запиту.
        if is_definition_query and focus_tokens:
            best_title_doc: Document | None = None
            best_title_overlap = 0
            for doc in context_docs:
                provider = str(doc.metadata.get("provider", "")).lower()
                if provider != "wikipedia":
                    continue
                doc_source = str(doc.metadata.get("source", "")).lower()
                source_tokens = set(self._normalize_tokens(doc_source))
                overlap = len(focus_tokens & source_tokens)
                if overlap > best_title_overlap:
                    best_title_overlap = overlap
                    best_title_doc = doc
            if best_title_doc is not None and best_title_overlap >= 1:
                sentences = re.split(r"(?<=[.!?])\s+|\n+", best_title_doc.page_content)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) >= 20:
                        return sentence

        if mode == "list":
            best_list_answer = None
            best_list_score = 0.0
            for doc in context_docs:
                list_answer = self._extract_list_answer_from_doc(query, doc)
                if not list_answer:
                    continue
                score = self._doc_evidence_score(query, doc)
                if score > best_list_score:
                    best_list_score = score
                    best_list_answer = list_answer
            if best_list_answer and best_list_score >= self.config.list_answer_min_score:
                return best_list_answer

        for doc in context_docs:
            doc_source = str(doc.metadata.get("source", "")).lower()
            provider = str(doc.metadata.get("provider", "")).lower()
            source_tokens = set(self._normalize_tokens(doc_source))
            doc_title_bias = 0.0
            if is_definition_query and focus_query:
                if f": {focus_query}" in doc_source or doc_source.endswith(focus_query):
                    doc_title_bias += 0.28
                elif f": {focus_query} (" in doc_source:
                    doc_title_bias += 0.2
                elif any(
                    marker in doc_source
                    for marker in (
                        f": велика {focus_query}",
                        f": диференціальна {focus_query}",
                        f": синхронне {focus_query}",
                        f": післяпологова {focus_query}",
                        "вільним стилем",
                    )
                ):
                    doc_title_bias -= 0.2
                if focus_tokens and source_tokens & focus_tokens == focus_tokens:
                    doc_title_bias += 0.08
            sentences = re.split(r"(?<=[.!?])\s+|\n+", doc.page_content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                score = self._sentence_evidence_score(query, sentence) + doc_title_bias
                if score <= 0.0:
                    continue
                sentence_lower = sentence.lower()
                if is_definition_query:
                    has_definition_pattern = " — це " in sentence_lower or " це " in sentence_lower or "що таке" in sentence_lower
                    phrase_match = " ".join(query_tokens) in sentence_lower if len(query_tokens) >= 2 else False
                    lexical_overlap = len(query_tokens & set(self._normalize_tokens(sentence))) / max(len(query_tokens), 1)
                    looks_encyclopedic = lexical_overlap >= 0.6 and len(sentence.split()) <= 32
                    # Для сторінок Wikipedia послаблюємо умови: достатньо
                    # помірної відповідності та наявності шаблону визначення
                    # або хоч якоїсь лексичної схожості.
                    if provider == "wikipedia":
                        if not (score >= 0.4 and (has_definition_pattern or lexical_overlap >= 0.4)):
                            continue
                    else:
                        if not (score >= 0.99 or phrase_match or (score >= 0.6 and has_definition_pattern) or looks_encyclopedic):
                            continue
                if mode == "list" and not self._is_list_like_sentence(sentence):
                    continue
                if score > best_score:
                    best_score = score
                    best_sentence = sentence

        if is_definition_query:
            if best_sentence and best_score >= self.config.extractive_min_score_definition:
                return best_sentence
            return "Не вдалося знайти коректне визначення у наданому контексті."

        if mode == "list":
            if best_sentence and best_score >= self.config.extractive_min_score_list:
                return best_sentence
            return "Не вдалося знайти прямий перелік або класифікацію у наданому контексті."

        if mode == "yes_no":
            if best_sentence and best_score >= self.config.extractive_min_score_yes_no:
                return best_sentence
            return "У наданому контексті немає достатньо прямої відповіді на це запитання."

        if mode == "factoid":
            if best_sentence and best_score >= self.config.extractive_min_score_factoid:
                return best_sentence
            return "У наданому контексті немає достатньо прямої відповіді на це фактологічне запитання."

        if best_sentence and best_score >= self.config.extractive_min_score_general:
            return best_sentence
        return None

    @staticmethod
    def _doc_reference(doc: Document) -> str:
        return f"{doc.metadata.get('source', '?')} [{doc.metadata.get('chunk_id', '?')}]"

    def _record_direct_provenance(
        self,
        result: RetrievalResult,
        doc: Document,
        score: float,
        retrieval_mode: str,
        reason: str,
    ) -> None:
        chunk_id = doc.metadata["chunk_id"]
        current = result.provenance_records.get(chunk_id)
        if current is not None and float(current.get("retrieval_score", 0.0)) >= float(score):
            return
        result.provenance_records[chunk_id] = {
            "chunk_id": chunk_id,
            "source": doc.metadata.get("source", "unknown"),
            "retrieval_mode": retrieval_mode,
            "retrieval_score": float(score),
            "reason": reason,
            "path": [
                {
                    "from": "query",
                    "to": chunk_id,
                    "relation": retrieval_mode,
                    "score": round(float(score), 3),
                    "reason": reason,
                }
            ],
        }

    def _record_graph_provenance(
        self,
        result: RetrievalResult,
        seed_doc: Document,
        doc: Document,
        score: float,
        hop_distance: int,
        edge_type: str,
        edge_weight: float,
    ) -> None:
        chunk_id = doc.metadata["chunk_id"]
        seed_chunk_id = seed_doc.metadata["chunk_id"]
        current = result.provenance_records.get(chunk_id)
        if current is not None and float(current.get("retrieval_score", 0.0)) >= float(score):
            return
        seed_record = result.provenance_records.get(seed_chunk_id)
        seed_path = list(seed_record.get("path", [])) if seed_record else []
        result.provenance_records[chunk_id] = {
            "chunk_id": chunk_id,
            "source": doc.metadata.get("source", "unknown"),
            "retrieval_mode": "graph_expansion",
            "retrieval_score": float(score),
            "reason": f"Expanded from {seed_chunk_id} via {edge_type}",
            "seed_chunk_id": seed_chunk_id,
            "hop_distance": hop_distance,
            "edge_type": edge_type,
            "edge_weight": float(edge_weight),
            "path": seed_path + [
                {
                    "from": seed_chunk_id,
                    "to": chunk_id,
                    "relation": f"graph_{edge_type}",
                    "score": round(float(score), 3),
                    "edge_weight": round(float(edge_weight), 3),
                    "hop_distance": hop_distance,
                    "reason": f"Graph expansion over {edge_type} edge",
                }
            ],
        }

    def _prune_provenance_records(self, result: RetrievalResult) -> None:
        selected = {doc.metadata["chunk_id"] for doc in result.context_docs}
        result.provenance_records = {
            chunk_id: record
            for chunk_id, record in result.provenance_records.items()
            if chunk_id in selected
        }
        score_by_chunk = {
            doc.metadata["chunk_id"]: float(score)
            for doc, score in zip(result.context_docs, result.vector_scores)
        }
        for rank, doc in enumerate(result.context_docs, start=1):
            chunk_id = doc.metadata["chunk_id"]
            record = result.provenance_records.get(chunk_id)
            if record is None:
                self._record_direct_provenance(
                    result,
                    doc,
                    score_by_chunk.get(chunk_id, 0.0),
                    retrieval_mode="retrieval",
                    reason="Selected for final context",
                )
                record = result.provenance_records.get(chunk_id)
            record["rank"] = rank
            record["retrieval_score"] = score_by_chunk.get(chunk_id, 0.0)

    def _select_supporting_docs(self, query: str, answer: str, context_docs: List[Document]) -> List[Document]:
        ranked_support = []
        answer_tokens = set(self._normalize_tokens(answer))
        for doc in context_docs:
            support = self._doc_evidence_score(query, doc)
            if answer and answer.lower() in doc.page_content.lower():
                support += 0.25
            overlap = len(answer_tokens & set(self._normalize_tokens(doc.page_content)))
            support += min(0.15, 0.03 * overlap)
            ranked_support.append((support, doc))
        ranked_support.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in ranked_support[:3] if score >= 0.35]

    def _format_answer_with_citations(self, answer: str, citations: List[str]) -> str:
        if not citations or "\n\nДжерела:" in answer:
            return answer
        return f"{answer}\n\nДжерела: {', '.join(citations)}"

    def _attach_answer_provenance(self, query: str, result: RetrievalResult, answer: str) -> str:
        answer_body = answer.split("\n\nДжерела:", 1)[0].strip()
        fallback_answers = {
            "Недостатньо контексту для відповіді.",
            "У наданому контексті немає достатньо інформації для відповіді.",
            "Не вдалося знайти коректне визначення у наданому контексті.",
            "Не вдалося знайти прямий перелік або класифікацію у наданому контексті.",
            "У наданому контексті немає достатньо прямої відповіді на це запитання.",
            "У наданому контексті немає достатньо прямої відповіді на це фактологічне запитання.",
        }
        if answer_body in fallback_answers:
            result.answer_citations = []
            result.answer_supporting_chunk_ids = []
            return answer_body
        supporting_docs = self._select_supporting_docs(query, answer_body, result.context_docs)
        result.answer_supporting_chunk_ids = [doc.metadata.get("chunk_id") for doc in supporting_docs]
        result.answer_citations = [self._doc_reference(doc) for doc in supporting_docs]
        return self._format_answer_with_citations(answer_body, result.answer_citations)

    def generate_answer_with_provenance(self, query: str, result: RetrievalResult) -> str:
        answer = self.generate_answer(query, result.context_docs)
        return self._attach_answer_provenance(query, result, answer)

    def _answer_quality_score(self, query: str, answer: str, context_docs: List[Document]) -> float:
        if not answer:
            return 0.0
        answer = answer.split("\n\nДжерела:", 1)[0].strip()
        answer_lower = answer.lower().strip()
        if answer_lower in {
            "недостатньо контексту для відповіді.",
            "у наданому контексті немає достатньо інформації для відповіді.",
            "не вдалося знайти коректне визначення у наданому контексті.",
            "не вдалося знайти прямий перелік або класифікацію у наданому контексті.",
            "у наданому контексті немає достатньо прямої відповіді на це запитання.",
            "у наданому контексті немає достатньо прямої відповіді на це фактологічне запитання.",
        }:
            return 0.0

        mode = self._query_mode(query)
        answer_tokens = set(self._normalize_tokens(answer))
        query_tokens = self._query_tokens(query)
        lexical = len(query_tokens & answer_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
        evidence = 0.0
        for doc in context_docs:
            if answer_lower in doc.page_content.lower():
                evidence = 1.0
                break
            evidence = max(evidence, self._sentence_evidence_score(query, answer))

        quality = 0.45 * lexical + 0.55 * evidence

        # Якщо відповідь пропускає суттєву частину ключових токенів запиту
        # (наприклад, не містить слова "трансформер" у запиті про трансформер),
        # вважаємо таку відповідь підозрілою і зменшуємо її якість.
        if query_tokens:
            missing = query_tokens - answer_tokens
            missing_ratio = len(missing) / len(query_tokens)
            if missing_ratio >= 0.4:
                quality *= 0.4
        if mode == "definition":
            if len(answer.split()) < 4:
                quality *= 0.3
            if " — " not in answer and " це " not in answer_lower and lexical < 0.5:
                quality *= 0.45
        elif mode == "list":
            has_list_shape = ";" in answer or "," in answer or any(marker in answer_lower for marker in ("типи", "види", "класиф"))
            if not has_list_shape:
                quality *= 0.35
        elif mode == "factoid":
            if len(answer.split()) < 2:
                quality *= 0.35
            if not any(char.isdigit() for char in answer) and not any(marker in answer_lower for marker in ("президент", "чинний", "літер", "букв", "року")):
                quality *= 0.45
        elif mode == "general":
            if len(answer.split()) < 3:
                quality *= 0.5
        return max(0.0, min(1.0, quality))

    def calibrate_confidence_with_answer(self, query: str, result: RetrievalResult, answer: str) -> float:
        answer_quality = self._answer_quality_score(query, answer, result.context_docs)
        result.answer_quality_score = answer_quality
        confidence = result.confidence_score
        base_confidence = confidence
        confidence = min(confidence, 0.15 + 0.85 * answer_quality)
        if answer_quality < 0.2:
            confidence = min(confidence, 0.18)
        elif answer_quality < 0.4:
            confidence = min(confidence, 0.35)
        elif answer_quality < 0.55:
            confidence = min(confidence, 0.52)
        result.confidence_score = max(0.0, min(1.0, confidence))
        result.confidence_factors = [
            factor for factor in result.confidence_factors
            if factor.get("name") != "answer_quality_calibration"
        ]
        result.confidence_factors.append(
            {
                "name": "answer_quality_calibration",
                "label": "Якість сформованої відповіді",
                "value": round(answer_quality, 3),
                "weight": 0.0,
                "contribution": round(result.confidence_score - base_confidence, 3),
                "description": "Після генерації confidence обмежується, якщо відповідь слабо підкріплена контекстом.",
            }
        )
        result.confidence_band = self._confidence_band(result.confidence_score)
        return result.confidence_score

    def process_documents(self, texts: List[str], metadatas=None):
        """Індексує сирі тексти, розбиває на фрагменти та будує граф.

        На виході створюється векторний індекс (FAISS) і граф
        послідовних та семантичних зв'язків між chunk'ами.
        """
        if not texts:
            return
        raw_docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {"source": f"doc_{i}"}
            raw_docs.append(Document(page_content=text, metadata=meta))
        self.documents = self.text_splitter.split_documents(raw_docs)
        for i, doc in enumerate(self.documents):
            doc.metadata["chunk_id"] = f"chunk_{i}"
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self._build_graph()

    def _build_graph(self):
        self.graph.clear()
        for doc in self.documents:
            assert "chunk_id" in doc.metadata, "Document is missing 'chunk_id' in metadata during graph build"
            self.graph.add_node(
                doc.metadata["chunk_id"],
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
            )
        for i in range(len(self.documents) - 1):
            d1, d2 = self.documents[i], self.documents[i + 1]
            if d1.metadata.get("source") == d2.metadata.get("source"):
                self.graph.add_edge(
                    d1.metadata["chunk_id"],
                    d2.metadata["chunk_id"],
                    type="sequential",
                    weight=1.0,
                )
        for doc in self.documents:
            similar = self.vector_store.similarity_search_with_score(doc.page_content, k=4)
            for sim_doc, score in similar:
                if sim_doc.metadata["chunk_id"] != doc.metadata["chunk_id"]:
                    weight = float(1.0 / (1.0 + score))
                    if weight > 0.45:
                        self.graph.add_edge(
                            doc.metadata["chunk_id"],
                            sim_doc.metadata["chunk_id"],
                            type="semantic",
                            weight=weight,
                        )

    def retrieve_with_graph(self, query: str, k: int = 3, hop_limit: int = 1) -> RetrievalResult:
        """Пошук релевантних фрагментів з графовим розширенням контексту.

        Спочатку обирає seed-фрагменти з векторного індексу, потім
        додає сусідів у графі за обмеженнями на вагу ребер і кількість
        кроків, формуючи розширений контекст та метрики довіри.
        """
        result = RetrievalResult()
        if not self.vector_store:
            return result
        initial = self._retrieve_candidates(query, k=k)
        if not initial:
            return result
        retrieved_ids = set()
        direct_scores = []
        for doc, score in initial:
            cid = doc.metadata["chunk_id"]
            retrieved_ids.add(cid)
            result.context_docs.append(doc)
            base_similarity = float(score)
            direct_scores.append(base_similarity)
            self._record_direct_provenance(
                result,
                doc,
                base_similarity,
                retrieval_mode="vector_seed",
                reason="Direct retrieval from vector index",
            )
            if not self._meets_definition_match_threshold(query, doc):
                continue
            ranked_neighbors = []
            for nbr_id, dist, edge_weight, edge_type in self._select_graph_neighbors(cid, hop_limit):
                if edge_weight < self.config.min_graph_edge_weight or nbr_id in retrieved_ids:
                    continue
                nbr_doc = self._get_document_by_chunk_id(nbr_id)
                if not nbr_doc:
                    continue
                neighbor_raw_score = max(0.0, (1.0 / max(edge_weight, 1e-6)) - 1.0)
                query_neighbor_score = self._score_doc_against_query(query, nbr_doc, neighbor_raw_score)
                graph_score = min(1.0, query_neighbor_score * (0.96 ** dist) * max(edge_weight, 0.5))
                if graph_score < self.config.graph_neighbor_min_score:
                    continue
                ranked_neighbors.append((nbr_doc, nbr_id, graph_score, dist, edge_type, edge_weight, doc))

            ranked_neighbors.sort(key=lambda item: item[2], reverse=True)
            for nbr_doc, nbr_id, graph_score, dist, edge_type, edge_weight, seed_doc in ranked_neighbors[:2]:
                if nbr_id in retrieved_ids:
                    continue
                retrieved_ids.add(nbr_id)
                result.context_docs.append(nbr_doc)
                direct_scores.append(graph_score)
                result.graph_expanded_count += 1
                self._record_graph_provenance(
                    result,
                    seed_doc,
                    nbr_doc,
                    graph_score,
                    hop_distance=dist,
                    edge_type=edge_type,
                    edge_weight=edge_weight,
                )

        result.context_docs, direct_scores = self._deduplicate_docs(result.context_docs, direct_scores)
        result.context_docs, direct_scores = self._trim_context(result.context_docs, direct_scores)
        retrieved_ids = {doc.metadata["chunk_id"] for doc in result.context_docs}
        result.vector_scores = direct_scores
        self._prune_provenance_records(result)
        result.unique_sources = len({d.metadata.get("source") for d in result.context_docs})
        result.query_alignment_score, result.top_match_score = self._compute_query_alignment(query, result.context_docs, direct_scores)
        result.answer_evidence_score = self._compute_answer_evidence(query, result.context_docs)
        if self.graph.nodes:
            centrality = nx.degree_centrality(self.graph)
            c_vals = [centrality.get(cid, 0.0) for cid in retrieved_ids]
            result.avg_graph_centrality = float(np.mean(c_vals)) if c_vals else 0.0
        result.confidence_score = self._compute_confidence(result)
        return result

    def retrieve_without_graph(self, query: str, k: int = 3) -> RetrievalResult:
        """Базовий RAG-пошук без використання графа походження.

        Використовує лише векторний індекс для відбору фрагментів
        і обчислює метрики довіри без графового розширення.
        """
        result = RetrievalResult()
        if not self.vector_store:
            return result
        initial = self._retrieve_candidates(query, k=k)
        if not initial:
            return result
        for doc, score in initial:
            result.context_docs.append(doc)
            result.vector_scores.append(float(score))
            self._record_direct_provenance(
                result,
                doc,
                float(score),
                retrieval_mode="vector",
                reason="Direct retrieval without graph expansion",
            )
        result.context_docs, result.vector_scores = self._deduplicate_docs(result.context_docs, result.vector_scores)
        result.context_docs, result.vector_scores = self._trim_context(result.context_docs, result.vector_scores)
        self._prune_provenance_records(result)
        result.unique_sources = len({d.metadata.get("source") for d in result.context_docs})
        result.query_alignment_score, result.top_match_score = self._compute_query_alignment(query, result.context_docs, result.vector_scores)
        result.answer_evidence_score = self._compute_answer_evidence(query, result.context_docs)
        result.confidence_score = self._compute_confidence(result)
        return result

    def compare(self, query: str, k: int = 3, hop_limit: int = 1) -> ComparisonResult:
        """Порівнює RAG з графом походження та базовий RAG.

        Виконує два запити (з графом і без), генерує відповіді та
        повертає структуру для аналізу різниці у контексті й довірі.
        """
        comp = ComparisonResult()
        comp.with_graph = self.retrieve_with_graph(query, k=k, hop_limit=hop_limit)
        comp.without_graph = self.retrieve_without_graph(query, k=k)
        comp.answer_with_graph = self.generate_answer_with_provenance(query, comp.with_graph)
        comp.answer_without_graph = self.generate_answer_with_provenance(query, comp.without_graph)
        return comp

    @staticmethod
    def _confidence_band(score: float) -> str:
        if score >= 0.8:
            return "висока"
        if score >= 0.6:
            return "помірна"
        if score >= 0.4:
            return "обережна"
        return "низька"

    def _compute_confidence(self, result: RetrievalResult) -> float:
        scores = result.vector_scores
        f1 = float(np.mean(scores)) if scores else 0.0
        total = len(result.context_docs) if result.context_docs else 1
        f2 = 1.0 - min(float(np.std(scores)) if len(scores) > 1 else 0.0, 1.0)
        f3 = min(len(result.context_docs) / 6.0, 1.0)
        f4 = 1.0 - min(result.graph_expanded_count / max(3, total), 1.0)
        f5 = min(result.avg_graph_centrality * 2.0, 1.0)
        f6 = result.query_alignment_score
        f7 = result.top_match_score
        f8 = result.answer_evidence_score

        value_by_name: Dict[str, float] = {
            "retrieval_similarity": f1,
            "score_stability": f2,
            "context_coverage": f3,
            "graph_precision": f4,
            "graph_centrality": f5,
            "query_alignment": f6,
            "top_match": f7,
            "answer_evidence": f8,
        }

        factors: List[Dict[str, Any]] = []
        for spec in self.CONFIDENCE_FACTORS:
            raw_value = value_by_name.get(spec.name, 0.0)
            contribution = spec.weight * raw_value
            factors.append(
                {
                    "name": spec.name,
                    "label": spec.label,
                    "value": round(raw_value, 3),
                    "weight": spec.weight,
                    "contribution": round(contribution, 3),
                    "description": spec.description,
                }
            )

        confidence = sum(factor["contribution"] for factor in factors)

        # Для локальних/одномовних сценаріїв зберігаємо жорстке обрізання
        # довіри за низького answer_evidence. Для веб-режиму з OpenAI, де
        # запит може бути українською, а джерела англійською, лексична
        # схожість між запитом і реченнями знижується, тому evidence може
        # бути штучно низьким. У такому випадку не обрізаємо довіру так
        # агресивно й покладаємося більше на якість відповіді.
        backend = getattr(self.config, "llm_backend", "local")
        providers = {str(doc.metadata.get("provider", "")).lower() for doc in result.context_docs}
        is_web_mixed = any(p in {"wikipedia", "openalex", "arxiv", "semantic scholar", "duckduckgo", "bing"} for p in providers)

        if not (backend == "openai" and is_web_mixed):
            if f8 < 0.35:
                confidence = min(confidence, 0.34)
            elif f8 < 0.5:
                confidence = min(confidence, 0.49)
        result.confidence_factors = factors
        result.confidence_band = self._confidence_band(confidence)
        return max(0.0, min(1.0, confidence))

    def _build_prompt_text(self, context: str, question: str) -> str:
        if getattr(self.config, "llm_backend", "local") == "openai":
            return (
                "Ти помічник з питань штучного інтелекту та суміжних тем. "
                "Тобі дається запит користувача та додатковий контекст (уривки з джерел). "
                "Використовуй як свої загальні знання, так і цей контекст, щоб дати найкращу, коректну та стислу відповідь. "
                "Якщо контекст суперечить твоїм знанням, надавай перевагу контексту. "
                "Якщо ти справді не знаєш точної відповіді, чесно напиши, що не знаєш. "
                "Відповідай лише українською мовою, чітко й по суті, без зайвих пояснень.\n\n"
                f"Контекст (може бути неповним):\n{context}\n\n"
                f"Питання користувача: {question}\n"
            )
        SYS = chr(60) + "|system|" + chr(62)
        USR = chr(60) + "|user|" + chr(62)
        AST = chr(60) + "|assistant|" + chr(62)
        EOS = chr(60) + "/s" + chr(62)
        return (
            f"{SYS}\n"
            "Ти помічник для системи RAG. Відповідай ТІЛЬКИ на основі наданого контексту. "
            "Не використовуй зовнішні знання. Якщо відповіді немає в контексті, напиши: "
            f"'У наданому контексті немає достатньо інформації для відповіді.' Відповідай лише українською мовою.{EOS}\n"
            f"{USR}\n"
            f"Контекст:\n{context}\n\n"
            f"Питання: {question}{EOS}\n"
            f"{AST}\n"
        )

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "Недостатньо контексту для відповіді."

        backend = getattr(self.config, "llm_backend", "local")

        extractive_answer = self._extractive_answer(query, context_docs)
        if extractive_answer:
            # Якщо екстрактивна відповідь є одним із стандартних fallback-повідомлень
            # про відсутність інформації, даю шанс генеративній моделі
            # (особливо для OpenAI-бекенду) спробувати побудувати відповідь
            # на основі ширшого контексту.
            fallback_answers = {
                "Недостатньо контексту для відповіді.",
                "У наданому контексті немає достатньо інформації для відповіді.",
                "Не вдалося знайти коректне визначення у наданому контексті.",
                "Не вдалося знайти прямий перелік або класифікацію у наданому контексті.",
                "У наданому контексті немає достатньо прямої відповіді на це запитання.",
                "У наданому контексті немає достатньо прямої відповіді на це фактологічне запитання.",
            }
            if backend != "openai" and extractive_answer not in fallback_answers:
                # Для локальних моделей, як і раніше, можу повертати хорошу
                # екстрактивну відповідь напряму.
                return extractive_answer
            # Для OpenAI-бекенду завжди даю шанс генеративній моделі, навіть
            # якщо екстрактивний шар знайшов речення з високим score.
            # Якщо екстрактивна відповідь була fallback-повідомленням, також
            # продовжуємо до генеративного етапу.

        answer_evidence = self._compute_answer_evidence(query, context_docs)
        backend = getattr(self.config, "llm_backend", "local")
        if backend != "openai" and answer_evidence < 0.3:
            return "У наданому контексті немає достатньо інформації для відповіді."

        context_text = "\n\n".join(
            f"[{doc.metadata.get('source', '?')}] {doc.page_content}"
            for doc in context_docs
        )
        if len(context_text) > self.config.max_context_chars_for_llm:
            context_text = context_text[: self.config.max_context_chars_for_llm] + "\n...[контекст скорочено]..."
        prompt_text = self._build_prompt_text(context_text, query)
        tpl = PromptTemplate(template="{full_prompt}", input_variables=["full_prompt"])
        chain = tpl | self.llm | StrOutputParser()
        response = chain.invoke({"full_prompt": prompt_text})
        AST = chr(60) + "|assistant|" + chr(62)
        if AST in response:
            response = response.split(AST)[-1].strip()
        response = response.strip()

        # Фільтр латиниця/кирилиця залишаємо лише для локальних моделей,
        # щоб захищатися від англомовних відповідей TinyLlama тощо.
        backend = getattr(self.config, "llm_backend", "local")
        if backend != "openai":
            latin_letters = sum(1 for char in response if "a" <= char.lower() <= "z")
            cyrillic_letters = sum(
                1
                for char in response
                if "а" <= char.lower() <= "я" or char.lower() in {"є", "і", "ї", "ґ"}
            )
            if latin_letters > cyrillic_letters:
                return "У наданому контексті немає достатньо інформації для відповіді."

        return response

    def get_subgraph_for_visualization(self, active_node_ids: List[str]) -> nx.Graph:
        if not self.graph.nodes:
            return nx.Graph()
        nodes_to_include = set(active_node_ids)
        for node_id in active_node_ids:
            if node_id in self.graph:
                nodes_to_include.update(self.graph.neighbors(node_id))
        return self.graph.subgraph(nodes_to_include)
