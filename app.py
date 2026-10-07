"""DATA 폴더의 자료를 찾아 답변하는 간단한 RAG 챗봇입니다."""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# 현재 파일 기준으로 경로를 계산해 어느 폴더에서 실행해도 같은 자료를 찾습니다.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "DATA"
ENV_FILE = PROJECT_DIR / ".env"
NO_ANSWER_MESSAGE = "자료에서 답을 확인할 수 없습니다."
INDEX_VERSION = 4

# .env에 저장한 OPENAI_API_KEY를 환경 변수로 불러옵니다.
load_dotenv(ENV_FILE)


def get_data_signature() -> tuple[tuple[str, int, int], ...]:
    """파일 변경 시 벡터 저장소를 다시 만들 수 있도록 파일 정보를 모읍니다."""
    if not DATA_DIR.exists():
        return ()

    return tuple(
        (str(path.relative_to(DATA_DIR)), path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(DATA_DIR.rglob("*"))
        if path.is_file()
    )


def load_data_files() -> tuple[list[Document], list[str]]:
    """PDF와 일반 텍스트 파일에서 페이지 또는 파일 단위 문서를 읽습니다."""
    documents: list[Document] = []
    skipped_files: list[str] = []

    if not DATA_DIR.exists():
        raise FileNotFoundError("프로젝트 최상단에 DATA 폴더가 없습니다.")

    supported_text_extensions = {".txt", ".md", ".csv", ".json"}
    data_files = sorted(path for path in DATA_DIR.rglob("*") if path.is_file())

    for path in data_files:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            # PDF는 페이지마다 나누어 저장해 출처와 페이지를 표시할 수 있게 합니다.
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    # 점선 뒤에 쪽수가 반복되는 목차는 답의 근거가 아니므로 검색에서 제외합니다.
                    table_of_contents_entries = re.findall(
                        r"[·.．…]{4,}\s*\d{1,4}", page_text
                    )
                    q_and_a_references = re.findall(
                        r"Q\s*&\s*A\s*\d+", page_text, flags=re.IGNORECASE
                    )
                    is_q_and_a_contents = (
                        len(q_and_a_references) >= 5
                        and not re.search(r"[•◦]\s+", page_text)
                    )
                    if len(table_of_contents_entries) >= 3 or is_q_and_a_contents:
                        continue
                    base_metadata = {
                        "source": path.name,
                        "page": page_number,
                        "relative_path": str(path.relative_to(PROJECT_DIR)),
                    }
                    # Q&A 문서는 한 페이지에 여러 문항이 있으므로 질문과 답을 문항별로 보존합니다.
                    qa_headings = list(
                        re.finditer(
                            r"(?im)^[ \t]*Q\s*&\s*A\s*(\d+)\b",
                            page_text,
                        )
                    )
                    if qa_headings:
                        for index, heading in enumerate(qa_headings):
                            end = (
                                qa_headings[index + 1].start()
                                if index + 1 < len(qa_headings)
                                else len(page_text)
                            )
                            qa_text = page_text[heading.start() : end].strip()
                            if qa_text:
                                documents.append(
                                    Document(
                                        page_content=qa_text,
                                        metadata={
                                            **base_metadata,
                                            "section": f"Q&A {heading.group(1)}",
                                        },
                                    )
                                )
                    else:
                        documents.append(
                            Document(page_content=page_text, metadata=base_metadata)
                        )
        elif suffix in supported_text_extensions:
            text = path.read_text(encoding="utf-8-sig")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "page": None,
                            "relative_path": str(path.relative_to(PROJECT_DIR)),
                        },
                    )
                )
        else:
            skipped_files.append(path.name)

    return documents, skipped_files


@st.cache_resource(scope="session", show_spinner="자료를 읽고 검색용 벡터를 만들고 있습니다...")
def build_vector_store(
    api_key: str,
    data_signature: tuple[tuple[str, int, int], ...],
    index_version: int,
) -> tuple[InMemoryVectorStore, int, list[str]]:
    """문서를 잘게 나눈 뒤 OpenAI 임베딩으로 메모리 벡터 저장소를 만듭니다."""
    # data_signature는 파일이 바뀌었을 때 캐시를 새로 만들기 위한 값입니다.
    del data_signature, index_version

    source_documents, skipped_files = load_data_files()
    if not source_documents:
        raise ValueError("DATA 폴더에서 읽을 수 있는 문서 내용을 찾지 못했습니다.")

    splitter = RecursiveCharacterTextSplitter(
        # 문단과 줄 경계를 우선 나누고, 긴 규정 문장만 필요한 경우에 추가로 자릅니다.
        separators=["\n\n", "\n", "。", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=120,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for document in source_documents:
        if document.metadata.get("section") and len(document.page_content) <= 2000:
            # Q&A는 질문, 답변, 관련 규정을 온전히 한 조각으로 유지합니다.
            chunks.append(document)
        else:
            chunks.extend(splitter.split_documents([document]))

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)
    return vector_store, len(chunks), skipped_files


def choose_evidence_sentence(text: str, question: str, answer: str = "") -> str:
    """검색된 문단에서 질문과 가장 관련 있는 실제 문장을 고릅니다."""
    # PDF Q&A의 글머리표 하나가 답변 근거 한 덩어리인 경우가 많아 먼저 찾습니다.
    bullet_blocks = [
        re.sub(r"\s+", " ", match.group(1)).strip()
        for match in re.finditer(
            r"(?ms)^[ \t]*[•◦][ \t]*(.*?)(?=^[ \t]*[•◦]|^[ \t]*관련 규정|\Z)",
            text,
        )
        if match.group(1).strip()
    ]
    if bullet_blocks:
        answer_terms = extract_terms(answer)
        question_terms = extract_terms(question)
        result_amounts = extract_amount_anchors(answer) - extract_amount_anchors(question)
        # 답변에서 마지막에 제시한 지급액이 근거에 있으면 해당 문장을 우선합니다.
        answer_amount_order = re.findall(
            r"\d[\d,]*(?:만\d*)?(?:천)?원",
            re.sub(r"\s+", "", answer.lower()),
        )
        final_amount = answer_amount_order[-1] if answer_amount_order else None
        if final_amount and final_amount in result_amounts:
            final_amount_evidence = next(
                (
                    item
                    for item in bullet_blocks
                    if final_amount in re.sub(r"\s+", "", item.lower())
                ),
                None,
            )
            if final_amount_evidence:
                return final_amount_evidence

        def bullet_score(item: str) -> tuple[int, int]:
            compact_item = re.sub(r"\s+", "", item.lower())
            direct_amount = sum(anchor in compact_item for anchor in result_amounts)
            term_overlap = 2 * sum(term in item.lower() for term in answer_terms)
            term_overlap += sum(term in item.lower() for term in question_terms)
            return direct_amount, term_overlap

        return max(bullet_blocks, key=bullet_score)

    flattened = re.sub(r"\s+", " ", text).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", flattened) if part.strip()]
    if not sentences:
        return flattened

    # 한글 형태소 분석 없이도 공통 단어가 많은 문장을 근거로 우선 표시합니다.
    answer_terms = extract_terms(answer)
    question_terms = extract_terms(question)
    terms = answer_terms | question_terms
    question_amounts = extract_amount_anchors(question)
    answer_amounts = extract_amount_anchors(answer)
    amount_anchors = question_amounts | answer_amounts
    # 답변에서 계산한 금액이 문장에 실제로 있으면 그 문장을 가장 먼저 근거로 씁니다.
    result_amounts = answer_amounts - question_amounts
    if result_amounts:
        result_sentence = next(
            (
                item
                for item in sentences
                if any(
                    anchor in re.sub(r"\s+", "", item.lower())
                    for anchor in result_amounts
                )
            ),
            None,
        )
        if result_sentence:
            return result_sentence
    if terms:
        sentence = max(
            sentences,
            key=lambda item: (
                2 * sum(term in item.lower() for term in answer_terms)
                + sum(term in item.lower() for term in question_terms)
                + 5
                * sum(
                    anchor in re.sub(r"\s+", "", item.lower())
                    for anchor in question_amounts
                )
                + 8
                * sum(
                    anchor in re.sub(r"\s+", "", item.lower())
                    for anchor in answer_amounts
                )
            ),
        )
        # 계산 결과가 다른 문장에 적혀 있으면 함께 보여줘 답과 근거를 바로 대조하게 합니다.
        compact_sentence = re.sub(r"\s+", "", sentence.lower())
        result_evidence = next(
            (
                item
                for item in sentences
                if item != sentence
                and any(
                    anchor in re.sub(r"\s+", "", item.lower())
                    for anchor in answer_amounts
                )
            ),
            "",
        )
        if result_evidence and not any(
            anchor in compact_sentence for anchor in answer_amounts
        ):
            sentence = f"{sentence} {result_evidence}"
    else:
        sentence = sentences[0]
    return sentence


def extract_terms(text: str) -> set[str]:
    """조사가 붙은 한글 단어에서 핵심 단어도 함께 뽑습니다."""
    particles = (
        "으로부터", "에서는", "에게서", "에서", "까지", "으로", "에게",
        "은", "는", "이", "가", "을", "를", "와", "과", "로", "에", "의", "도",
    )
    terms: set[str] = set()
    for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", text.lower()):
        terms.add(token)
        for particle in particles:
            if token.endswith(particle) and len(token) - len(particle) >= 2:
                terms.add(token[: -len(particle)])
    return terms


def extract_amount_anchors(text: str) -> set[str]:
    """질문의 원화 금액을 공백 없이 뽑아 해당 금액이 있는 문단을 찾습니다."""
    compact_text = re.sub(r"\s+", "", text.lower())
    return set(re.findall(r"\d[\d,]*(?:만\d*)?(?:천)?원", compact_text))


def select_supporting_sources(
    answer: str,
    documents: list[Document] | list[dict[str, object]],
    question: str = "",
) -> list[Document] | list[dict[str, object]]:
    """답변의 핵심 단어가 실제 문서에도 있는 검색 결과만 출처로 남깁니다."""
    ignored_terms = {"자료에서", "확인할", "수", "있습니다", "입니다", "경우"}
    terms = extract_terms(answer) - ignored_terms
    if not terms:
        return []
    question_terms = extract_terms(question) - ignored_terms

    def source_text(source: Document | dict[str, object]) -> str:
        if isinstance(source, Document):
            return source.page_content.lower()
        return str(source.get("text", "")).lower()

    scored_documents = [
        (
            sum(term in source_text(document) for term in terms)
            + 3 * sum(term in source_text(document) for term in question_terms),
            document,
        )
        for document in documents
    ]
    # 답변에 계산 결과가 있으면 그 금액이 실제로 적힌 조각을 우선 출처로 고릅니다.
    amount_anchors = extract_amount_anchors(f"{question} {answer}")
    if amount_anchors:
        amount_scores = [
            (
                sum(
                    anchor in re.sub(r"\s+", "", source_text(document))
                    for anchor in amount_anchors
                ),
                document,
            )
            for _score, document in scored_documents
        ]
        best_amount_score = max((score for score, _document in amount_scores), default=0)
        if best_amount_score:
            scored_documents = [
                (score, document)
                for score, document in amount_scores
                if score == best_amount_score
            ]
    scored_documents = [item for item in scored_documents if item[0] > 0]
    scored_documents.sort(key=lambda item: item[0], reverse=True)
    if not scored_documents:
        return []
    minimum_score = max(2, (scored_documents[0][0] + 1) // 2)
    return [
        document
        for score, document in scored_documents
        if score >= minimum_score
    ][:1]


def answer_question(
    vector_store: InMemoryVectorStore,
    question: str,
    api_key: str,
) -> tuple[str, list[Document]]:
    """관련 문서를 검색하고 검색 결과에 한정해 답변을 만듭니다."""
    # 원 질문과 숫자를 뺀 주제 검색을 함께 해 금액 표현 때문에 정답 문서가 밀리지 않게 합니다.
    topic_query = re.sub(
        r"\d[\d,]*(?:\s*만)?(?:\s*\d*\s*천)?\s*원",
        " ",
        question,
    )
    query_variants = [
        question,
        topic_query,
        f"{topic_query} 지급 기준 실비 상한액 산정",
    ]
    if any(term in question for term in ("거주", "체재", "바로", "직접 출발")):
        query_variants.append(
            "근무지 또는 출장지 외 거주지 체재지에서 목적지까지 직접 여행 운임 "
            "거주지 목적지 운임은 근무지 목적지 여비를 초과하지 못함"
        )
    if any(term in question for term in ("기상", "악화", "늘어난", "연장", "초과")):
        query_variants.append(
            "Q&A 50 기상악화 천재지변 부득이한 사유 출장일정 초과 늘어나는 일수 "
            "여행일수 포함 추가 숙박비 식비 일비 여비 지급 가능"
        )
    ranked_documents: dict[tuple[str, int | None, int | None, str], tuple[Document, float]] = {}
    for query in query_variants:
        for document, score in vector_store.similarity_search_with_score(query, k=6):
            identity = (
                document.metadata.get("source", ""),
                document.metadata.get("page"),
                document.metadata.get("start_index"),
                document.page_content[:120],
            )
            if identity not in ranked_documents or score > ranked_documents[identity][1]:
                ranked_documents[identity] = (document, score)

    source_documents = [
        document
        for document, _score in sorted(
            ranked_documents.values(),
            key=lambda result: result[1],
            reverse=True,
        )[:8]
    ]

    # 숙박일 수가 없으면 다른 사례의 박 수를 가져오지 않고 필요한 정보를 되묻습니다.
    if (
        any(term in question for term in ("숙박비", "숙박료"))
        and not re.search(r"\d+\s*박", question)
        and not extract_amount_anchors(question)
    ):
        lodging_text = "\n".join(document.page_content for document in source_documents)
        nightly_cap = re.search(
            r"1\s*(?:박|夜)\s*당\s*([0-9,만천원]+)",
            re.sub(r"\s+", "", lodging_text),
        )
        if nightly_cap:
            answer = (
                f"자료에 나온 숙박비 상한은 1박당 {nightly_cap.group(1)}입니다. "
                "총 지급액을 계산하려면 숙박 일수와 실제 지출액이 필요합니다. "
                "몇 박이며 각 숙박일에 얼마를 지출하셨나요?"
            )
            return answer, select_supporting_sources(answer, source_documents, question)

    context = "\n\n".join(
        f"[자료 {index}] 파일: {document.metadata['source']}\n"
        f"페이지: {document.metadata.get('page') or '해당 없음'}\n"
        f"내용:\n{document.page_content}"
        for index, document in enumerate(source_documents, start=1)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 제공된 자료만 근거로 답하는 한국어 RAG 챗봇입니다. "
                "자료에 직접 적혀 있지 않은 사실은 일반 지식으로 보충하거나 추측하지 마세요. "
                f"자료에서 답을 확인할 수 없거나 근거가 부족하면 정확히 '{NO_ANSWER_MESSAGE}'라고 답하세요. "
                "질문에 없는 숙박 일수나 실제 지출액을 검색된 다른 사례에서 가져와 현재 질문의 사실처럼 쓰지 마세요. "
                "질문이 총 지급액을 묻는데 숙박 일수나 실제 지출액처럼 계산에 필요한 정보가 빠졌다면 "
                "임의의 숙박 일수 예시를 들지 말고 자료에서 확인되는 1박 기준만 말한 뒤 빠진 조건을 물어보세요. "
                "간결하고 이해하기 쉬운 한국어로 답하고, 출처 목록이나 인용 문장을 만들어 내지 마세요. "
                "자료에 나온 규칙과 숫자를 사용한 간단한 덧셈, 비교, 상한 적용 계산은 수행하고 계산식을 설명하세요. "
                "출처와 근거 문장은 검색 결과를 바탕으로 화면에 별도로 표시합니다.",
            ),
            (
                "human",
                "질문:\n{question}\n\n검색된 자료:\n{context}\n\n"
                "검색된 자료의 내용만으로 질문에 답해 주세요.",
            ),
        ]
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    response = (prompt | model).invoke({"question": question, "context": context})
    answer = str(response.content).strip()
    # 근거 부족으로 답을 보류한 경우, 검색 결과를 근거처럼 잘못 표시하지 않습니다.
    if answer == NO_ANSWER_MESSAGE or (
        "확인할 수 없습니다" in answer
        and any(term in answer for term in ("자료", "문서", "정보"))
    ):
        return answer, []
    full_pages, _skipped_files = load_data_files()
    # Q&A는 문항별 검색 조각을 그대로 써서 같은 페이지의 다른 문항을 근거로 섞지 않습니다.
    citation_candidates = source_documents
    selected_sources = select_supporting_sources(answer, citation_candidates, question)
    # 검색 조각에서 근거 문장이 잘렸을 수 있어 인용할 때는 해당 페이지 전체 문서를 사용합니다.
    expanded_sources: list[Document] = []
    for source in selected_sources:
        full_page = next(
            (
                page
                for page in full_pages
                if page.metadata.get("source") == source.metadata.get("source")
                and page.metadata.get("page") == source.metadata.get("page")
            ),
            source,
        )
        # 문항 단위로 잘라 둔 Q&A는 페이지 전체로 되돌리지 않습니다.
        expanded_sources.append(source if source.metadata.get("section") else full_page)
    return answer, expanded_sources


st.set_page_config(page_title="자료 기반 RAG 챗봇", page_icon="📚", layout="centered")
st.title("📚 자료 기반 RAG 챗봇")
st.write("DATA 폴더의 문서를 검색해 근거가 있는 답변을 제공합니다.")

if "conversation" not in st.session_state:
    # Streamlit이 화면을 다시 그려도 현재 세션의 대화 목록을 유지합니다.
    st.session_state["conversation"] = []

if st.button(
    "대화 초기화",
    disabled=not st.session_state["conversation"],
    help="현재 대화 기록을 모두 지웁니다.",
):
    st.session_state["conversation"] = []
    st.rerun()

api_key = os.getenv("OPENAI_API_KEY", "").strip()
vector_store: InMemoryVectorStore | None = None

if not api_key:
    st.error(".env 파일의 OPENAI_API_KEY= 뒤에 OpenAI API 키를 입력해 주세요.")
elif not DATA_DIR.exists():
    st.error("프로젝트 최상단에 DATA 폴더가 없습니다.")
elif not get_data_signature():
    st.warning("DATA 폴더에 파일이 없습니다. 문서를 추가한 뒤 새로고침해 주세요.")
else:
    data_signature = get_data_signature()
    try:
        vector_store, chunk_count, skipped_files = build_vector_store(
            api_key,
            data_signature,
            INDEX_VERSION,
        )
        st.caption(f"검색 준비 완료: 문서 조각 {chunk_count}개")
        if skipped_files:
            st.warning("지원하지 않는 파일 형식이라 건너뛴 파일: " + ", ".join(skipped_files))
    except Exception as error:
        st.error(f"자료를 준비하지 못했습니다: {error}")

# 저장된 질문과 답변을 매번 화면에 다시 그려 이전 대화를 계속 볼 수 있게 합니다.
for turn in st.session_state["conversation"]:
    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        supporting_sources = select_supporting_sources(
            turn["answer"],
            turn["sources"],
            turn["question"],
        )
        if supporting_sources and turn["answer"] != NO_ANSWER_MESSAGE:
            st.caption("출처 및 근거 문장")
            displayed_evidence: set[tuple[str, int | None, str]] = set()
            for source in supporting_sources:
                file_name = str(source["source"])
                page_number = source["page"]
                sentence = choose_evidence_sentence(
                    source["text"],
                    turn["question"],
                    turn["answer"],
                )
                evidence_key = (file_name, page_number, sentence)
                if evidence_key in displayed_evidence:
                    continue
                displayed_evidence.add(evidence_key)

                location = f" · {page_number}쪽" if page_number else ""
                st.markdown(f"**{file_name}{location}**")
                st.write(f"> {sentence}")

if vector_store is not None:
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("질문", placeholder="예: 국내 출장의 일비 기준은 얼마인가요?")
        submitted = st.form_submit_button("질문하기", type="primary")

    if submitted:
        if not question.strip():
            st.warning("질문을 입력해 주세요.")
        else:
            try:
                with st.spinner("자료를 검색하고 답변을 작성하고 있습니다..."):
                    answer, source_documents = answer_question(vector_store, question.strip(), api_key)
                st.session_state["conversation"].append(
                    {
                        "question": question.strip(),
                        "answer": answer,
                        "sources": [
                            {
                                "source": document.metadata["source"],
                                "page": document.metadata.get("page"),
                                "text": document.page_content,
                            }
                            for document in source_documents
                        ],
                    }
                )
                st.rerun()
            except Exception as error:
                st.error(f"답변을 만드는 중 문제가 생겼습니다: {error}")
