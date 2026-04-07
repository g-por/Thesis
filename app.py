import streamlit as st
import streamlit.components.v1 as components
import tempfile
import networkx as nx
from io import BytesIO
from pathlib import Path
from statistics import mean
from pyvis.network import Network
from pypdf import PdfReader
from docx import Document as DocxDocument
from rag_engine import ProvenanceGraphRAG, RetrievalResult
from web_retrieval import WebSearchClient

st.set_page_config(page_title="Provenance Graph RAG", layout="wide")

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

WEB_PROVIDER_OPTIONS = ["Wikipedia", "OpenAlex", "arXiv", "Semantic Scholar", "DuckDuckGo"]


def load_builtin_corpus() -> tuple[list[str], list[dict]]:
    corpus_path = Path(__file__).with_name("test_data.txt")
    if not corpus_path.exists():
        return [], []

    text = corpus_path.read_text(encoding="utf-8")
    return [text], [{"source": corpus_path.name}]


def read_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.getvalue()

    if suffix in {".txt", ".md"}:
        return raw_bytes.decode("utf-8")

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        doc = DocxDocument(BytesIO(raw_bytes))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    raise ValueError(f"Непідтримуваний формат: {suffix}")


def render_graph(graph: nx.Graph, active_nodes: list = None):
    if not graph or len(graph.nodes) == 0:
        st.info("Граф порожній. Використайте вбудований корпус або завантажте документи.")
        return

    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    net.toggle_physics(True)

    for node_id in graph.nodes:
        node_data = graph.nodes[node_id]
        color = "#00ff1e" if active_nodes and node_id in active_nodes else "#97c2fc"
        size = 20 if active_nodes and node_id in active_nodes else 15
        title = (
            f"ID: {node_id}\n"
            f"Source: {node_data.get('source', 'unknown')}\n\n"
            f"Content:\n{node_data.get('content', '')[:120]}..."
        )
        net.add_node(
            node_id,
            label=f"Chunk {node_id.split('_')[-1]}",
            title=title,
            color=color,
            size=size,
        )

    for source, target, edge_data in graph.edges(data=True):
        edge_type = edge_data.get("type", "unknown")
        color = "#e74c3c" if edge_type == "semantic" else "#95a5a6"
        weight = float(edge_data.get("weight", 1.0))
        net.add_edge(source, target, value=weight, color=color, title=f"Type: {edge_type}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, "r", encoding="utf-8") as f:
            html_source = f.read()
        components.html(html_source, height=520, scrolling=True)


def build_web_graph(docs: list) -> nx.Graph:
    graph = nx.Graph()
    if not docs:
        return graph

    provider_groups: dict[str, list[str]] = {}
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        provider = doc.metadata.get("provider", "Веб")
        source = doc.metadata.get("source", "unknown")
        graph.add_node(
            chunk_id,
            content=doc.page_content,
            source=source,
        )
        provider_groups.setdefault(provider, []).append(chunk_id)

    for provider, node_ids in provider_groups.items():
        provider_node = f"provider::{provider}"
        graph.add_node(
            provider_node,
            content=f"Провайдер {provider}",
            source=provider,
        )
        for node_id in node_ids:
            graph.add_edge(provider_node, node_id, type="provider", weight=1.0)

    docs_by_title = []
    for doc in docs:
        tokens = set(str(doc.metadata.get("source", "")).lower().replace(":", " ").split())
        docs_by_title.append((doc.metadata.get("chunk_id", "unknown"), tokens))

    for idx, (left_id, left_tokens) in enumerate(docs_by_title):
        for right_id, right_tokens in docs_by_title[idx + 1:]:
            overlap = left_tokens & right_tokens
            if len(overlap) >= 1:
                graph.add_edge(left_id, right_id, type="semantic", weight=min(1.0, 0.35 + 0.15 * len(overlap)))
    return graph


def get_engine(model_name: str) -> ProvenanceGraphRAG:
    engine = st.session_state.rag_engine
    if engine is None:
        engine = ProvenanceGraphRAG(model_name=model_name)
        st.session_state.rag_engine = engine
    return engine


def build_web_result(engine: ProvenanceGraphRAG, query: str, providers: list[str], limit_per_provider: int) -> RetrievalResult:
    client = WebSearchClient()
    web_results = client.search_many(query, providers, limit_per_provider=limit_per_provider)
    result = RetrievalResult()
    if not web_results:
        return result
    docs = [item.to_document() for item in web_results]
    for idx, doc in enumerate(docs):
        doc.metadata["chunk_id"] = f"web_{idx}"
    scores = [item.score for item in web_results]
    docs, scores = engine._deduplicate_docs(docs, scores)
    docs, scores = engine._trim_context(docs, scores)
    result.context_docs = docs
    result.vector_scores = scores
    result.unique_sources = len({d.metadata.get("source") for d in docs})
    result.query_alignment_score, result.top_match_score = engine._compute_query_alignment(query, docs, scores)
    result.answer_evidence_score = engine._compute_answer_evidence(query, docs)
    result.confidence_score = engine._compute_confidence(result)
    return result


def merge_results(engine: ProvenanceGraphRAG, query: str, local_result: RetrievalResult, web_result: RetrievalResult) -> RetrievalResult:
    merged = RetrievalResult()
    merged.context_docs = list(local_result.context_docs) + list(web_result.context_docs)
    merged.vector_scores = list(local_result.vector_scores) + list(web_result.vector_scores)
    merged.graph_expanded_count = local_result.graph_expanded_count
    merged.context_docs, merged.vector_scores = engine._deduplicate_docs(merged.context_docs, merged.vector_scores)
    merged.context_docs, merged.vector_scores = engine._trim_context(merged.context_docs, merged.vector_scores)
    merged.unique_sources = len({d.metadata.get("source") for d in merged.context_docs})
    merged.avg_graph_centrality = local_result.avg_graph_centrality
    merged.query_alignment_score, merged.top_match_score = engine._compute_query_alignment(query, merged.context_docs, merged.vector_scores)
    merged.answer_evidence_score = engine._compute_answer_evidence(query, merged.context_docs)
    base_confidences = [score for score in [local_result.confidence_score, web_result.confidence_score] if score > 0.0]
    merged.confidence_score = engine._compute_confidence(merged)
    if base_confidences:
        merged.confidence_score = min(1.0, 0.65 * merged.confidence_score + 0.35 * mean(base_confidences))
    return merged


def build_answer(engine: ProvenanceGraphRAG, query: str, result: RetrievalResult) -> str:
    return engine.generate_answer(query, result.context_docs)


def render_context_block(title: str, result: RetrievalResult, answer: str):
    st.markdown(f"**{title}**")
    st.markdown(f"Відповідь: {answer}")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Confidence", f"{result.confidence_score:.3f}")
    mc2.metric("Фрагментів", len(result.context_docs))
    mc3.metric("Додано графом", result.graph_expanded_count)
    mc4.metric("Джерел", result.unique_sources)
    for doc in result.context_docs:
        extra = ""
        if doc.metadata.get("url"):
            extra = f"\n{doc.metadata.get('url')}"
        st.info(
            f"[{doc.metadata.get('source','?')} - {doc.metadata.get('chunk_id')} ]\n"
            f"{doc.page_content[:220]}...{extra}"
        )


# --- Page ---
st.title("RAG з графом походження та оцінкою довіри")
st.markdown(
    "Цей додаток демонструє метод генерації відповідей на основі RAG, "
    "що використовує граф походження для розширення контексту "
    "та багатофакторну оцінку довіри."
)

# --- Sidebar ---
with st.sidebar:
    st.header("Налаштування")

    model_choice = st.selectbox(
        "Локальна модель",
        ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-0.5B-Instruct"],
    )
    st.info("Модель завантажується локально. Перше завантаження може зайняти певний час.")

    st.subheader("Режим пошуку")
    search_mode = st.radio(
        "Де шукати відповідь",
        ["Локальні файли", "Веб", "Локальні файли + Веб"],
        index=0,
    )
    selected_web_providers = st.multiselect(
        "Веб-провайдери",
        WEB_PROVIDER_OPTIONS,
        default=["Wikipedia", "OpenAlex"],
    )
    web_limit = st.slider("Результатів з кожного веб-провайдера", 1, 5, 2)

    st.subheader("Джерела знань")
    use_builtin_corpus = st.checkbox("Додати вбудований тестовий корпус", value=False)
    uploaded_files = st.file_uploader(
        "Додатково завантажте документи (.txt, .md, .pdf, .docx)",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )

    if st.button("Побудувати базу знань"):
        needs_local_corpus = search_mode in {"Локальні файли", "Локальні файли + Веб"}
        if needs_local_corpus and not use_builtin_corpus and not uploaded_files:
            st.warning("Оберіть вбудований корпус або завантажте хоча б один документ.")
        elif search_mode in {"Веб", "Локальні файли + Веб"} and not selected_web_providers:
            st.warning("Оберіть хоча б один веб-провайдер.")
        else:
            with st.spinner("Завантаження моделі та підготовка джерел..."):
                try:
                    texts = []
                    metadatas = []

                    for file in uploaded_files:
                        text = read_uploaded_file(file)
                        if not text.strip():
                            continue
                        texts.append(text)
                        metadatas.append({"source": file.name})

                    if use_builtin_corpus:
                        builtin_texts, builtin_metadatas = load_builtin_corpus()
                        texts.extend(builtin_texts)
                        metadatas.extend(builtin_metadatas)

                    if needs_local_corpus and not texts:
                        st.warning("Не вдалося отримати текст із вибраних джерел.")
                        st.stop()

                    engine = ProvenanceGraphRAG(model_name=model_choice)
                    if texts:
                        engine.process_documents(texts, metadatas)
                    st.session_state.rag_engine = engine
                    st.session_state.processed = True
                    st.session_state.chat_history = []
                    if texts:
                        st.success(
                            f"Успішно оброблено {len(texts)} локальних джерел. "
                            f"Граф: {len(engine.graph.nodes)} вузлів, "
                            f"{len(engine.graph.edges)} зв'язків."
                        )
                    else:
                        st.success("Локальну модель підготовлено для веб-пошуку та синтезу відповіді.")
                except Exception as e:
                    st.error(f"Помилка: {e}")

# --- Main area ---
tab_query, tab_compare = st.tabs(["Запит", "Порівняння: з графом vs без графа"])

# ==================== TAB 1: Query ====================
with tab_query:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Запит")
        user_query = st.text_input("Введіть запит:", key="q1")
        k_retrieval = st.slider("Кількість базових фрагментів (k)", 1, 10, 3, key="k1")
        hop_limit = st.slider("Глибина обходу графа (hops)", 0, 3, 1, key="h1")

        if st.button("Надіслати", type="primary", key="btn1"):
            if not st.session_state.processed:
                st.error("Спочатку побудуйте базу знань.")
            elif not user_query:
                st.warning("Введіть запит.")
            elif search_mode in {"Веб", "Локальні файли + Веб"} and not selected_web_providers:
                st.warning("Оберіть хоча б один веб-провайдер.")
            else:
                with st.spinner("Пошук та генерація відповіді..."):
                    engine = get_engine(model_choice)
                    if search_mode == "Локальні файли":
                        res = engine.retrieve_with_graph(user_query, k=k_retrieval, hop_limit=hop_limit)
                        answer = build_answer(engine, user_query, res)
                        engine.calibrate_confidence_with_answer(user_query, res, answer)
                        retrieved_ids = [d.metadata["chunk_id"] for d in res.context_docs]
                        st.session_state.chat_history.append({
                            "query": user_query,
                            "answer": answer,
                            "result": res,
                            "active_nodes": retrieved_ids,
                            "search_mode": search_mode,
                        })
                    elif search_mode == "Веб":
                        res = build_web_result(engine, user_query, selected_web_providers, web_limit)
                        answer = build_answer(engine, user_query, res)
                        engine.calibrate_confidence_with_answer(user_query, res, answer)
                        retrieved_ids = [d.metadata["chunk_id"] for d in res.context_docs]
                        st.session_state.chat_history.append({
                            "query": user_query,
                            "answer": answer,
                            "result": res,
                            "active_nodes": retrieved_ids,
                            "search_mode": search_mode,
                        })
                    else:
                        local_res = engine.retrieve_with_graph(user_query, k=k_retrieval, hop_limit=hop_limit)
                        web_res = build_web_result(engine, user_query, selected_web_providers, web_limit)
                        merged = merge_results(engine, user_query, local_res, web_res)
                        local_answer = build_answer(engine, user_query, local_res)
                        web_answer = build_answer(engine, user_query, web_res)
                        engine.calibrate_confidence_with_answer(user_query, local_res, local_answer)
                        engine.calibrate_confidence_with_answer(user_query, web_res, web_answer)
                        merged.confidence_score = max(local_res.confidence_score, web_res.confidence_score)
                        st.session_state.chat_history.append({
                            "query": user_query,
                            "answer": local_answer if local_res.confidence_score >= web_res.confidence_score else web_answer,
                            "result": merged,
                            "local_result": local_res,
                            "web_result": web_res,
                            "local_answer": local_answer,
                            "web_answer": web_answer,
                            "active_nodes": [d.metadata["chunk_id"] for d in local_res.context_docs],
                            "search_mode": search_mode,
                        })

        st.subheader("Історія")
        for chat in reversed(st.session_state.chat_history):
            res = chat["result"]
            with st.expander(f"Q: {chat['query']}  |  Довіра: {res.confidence_score:.2f} | Режим: {chat.get('search_mode', 'Локальні файли')}"):
                if chat.get("search_mode") == "Локальні файли + Веб":
                    st.markdown("**Комбінований режим:** локальні та веб-результати показані окремо.")
                    render_context_block("Локальні файли", chat["local_result"], chat["local_answer"])
                    render_context_block("Веб", chat["web_result"], chat["web_answer"])
                else:
                    render_context_block("Результат", res, chat["answer"])

    with col2:
        st.subheader("Візуалізація графа походження")
        if st.session_state.processed and search_mode == "Локальні файли":
            st.markdown(
                "- **Сині вузли** - усі фрагменти документів.\n"
                "- **Зелені вузли** - фрагменти, використані для генерації останньої відповіді.\n"
                "- **Сірі зв'язки** - послідовність у документі.\n"
                "- **Червоні зв'язки** - семантична схожість."
            )
            engine = st.session_state.rag_engine
            active_nodes = []
            if st.session_state.chat_history:
                active_nodes = st.session_state.chat_history[-1]["active_nodes"]
            show_sub = st.checkbox("Показати лише активний підграф", value=False, key="sub1")
            graph_to_render = engine.graph
            if show_sub and active_nodes:
                graph_to_render = engine.get_subgraph_for_visualization(active_nodes)
            render_graph(graph_to_render, active_nodes)
        elif st.session_state.processed and search_mode == "Веб":
            st.markdown(
                "- **Сині вузли** - знайдені веб-джерела.\n"
                "- **Зелені вузли** - джерела, використані у поточній відповіді.\n"
                "- **Сірі зв'язки** - належність до провайдера.\n"
                "- **Червоні зв'язки** - слабка тематична схожість між результатами."
            )
            active_nodes = []
            web_docs = []
            if st.session_state.chat_history:
                active_nodes = st.session_state.chat_history[-1].get("active_nodes", [])
                web_docs = st.session_state.chat_history[-1].get("result", RetrievalResult()).context_docs
            render_graph(build_web_graph(web_docs), active_nodes)
        elif st.session_state.processed and search_mode == "Локальні файли + Веб":
            st.markdown(
                "- **Сині вузли** - локальні або веб-фрагменти.\n"
                "- **Зелені вузли** - локальні фрагменти, використані для локальної відповіді.\n"
                "- **Сірі зв'язки** - структура документа або належність до провайдера.\n"
                "- **Червоні зв'язки** - семантична схожість."
            )
            active_nodes = []
            local_graph = st.session_state.rag_engine.graph
            web_docs = []
            if st.session_state.chat_history:
                active_nodes = st.session_state.chat_history[-1].get("active_nodes", [])
                web_docs = st.session_state.chat_history[-1].get("web_result", RetrievalResult()).context_docs
            graph_choice = st.radio(
                "Що візуалізувати",
                ["Локальний граф", "Веб-граф"],
                horizontal=True,
                key="combined_graph_choice",
            )
            if graph_choice == "Локальний граф":
                show_sub = st.checkbox("Показати лише активний підграф", value=False, key="sub_combined")
                graph_to_render = local_graph
                if show_sub and active_nodes:
                    graph_to_render = st.session_state.rag_engine.get_subgraph_for_visualization(active_nodes)
                render_graph(graph_to_render, active_nodes)
            else:
                web_active_nodes = [doc.metadata.get("chunk_id") for doc in web_docs]
                render_graph(build_web_graph(web_docs), web_active_nodes)
        elif st.session_state.processed:
            st.info("Граф буде доступний після отримання результатів пошуку.")
        else:
            st.info("Граф буде відображено після побудови бази знань.")

# ==================== TAB 2: Comparison ====================
with tab_compare:
    st.subheader("Порівняння: RAG з графом vs RAG без графа")
    if search_mode != "Локальні файли":
        st.info("Порівняння графового та базового RAG доступне лише для локальних файлів.")
    elif not st.session_state.processed:
        st.info("Спочатку побудуйте базу знань.")
    else:
        cmp_query = st.text_input("Запит для порівняння:", key="q2")
        cmp_k = st.slider("k", 1, 10, 3, key="k2")
        cmp_hop = st.slider("hops", 0, 3, 1, key="h2")

        if st.button("Порівняти", type="primary", key="btn2"):
            if not cmp_query:
                st.warning("Введіть запит.")
            else:
                with st.spinner("Генерація двох відповідей..."):
                    engine = st.session_state.rag_engine
                    comp = engine.compare(cmp_query, k=cmp_k, hop_limit=cmp_hop)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### З графом походження")
                        st.success(comp.answer_with_graph)
                        r = comp.with_graph
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Confidence", f"{r.confidence_score:.3f}")
                        m2.metric("Фрагментів", len(r.context_docs))
                        m3.metric("Додано графом", r.graph_expanded_count)
                        m4.metric("Джерел", r.unique_sources)

                    with c2:
                        st.markdown("### Без графа (baseline)")
                        st.warning(comp.answer_without_graph)
                        r2 = comp.without_graph
                        m5, m6, m7, m8 = st.columns(4)
                        m5.metric("Confidence", f"{r2.confidence_score:.3f}")
                        m6.metric("Фрагментів", len(r2.context_docs))
                        m7.metric("Додано графом", r2.graph_expanded_count)
                        m8.metric("Джерел", r2.unique_sources)

                    st.markdown("---")
                    delta = comp.with_graph.confidence_score - comp.without_graph.confidence_score
                    if delta > 0:
                        st.success(f"Граф походження підвищив довіру на +{delta:.3f}")
                    elif delta < 0:
                        st.warning(f"Граф походження знизив довіру на {delta:.3f}")
                    else:
                        st.info("Рівень довіри однаковий.")
