import base64
import io
import os
import uuid

import fitz
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from tqdm import tqdm


def init_multimodal_retriever(
    documents_dir, embeddings, llm, force_reload: bool = False
):
    def pdf_page_to_base64(pdf_path: str, page_number: int):
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(page_number - 1)  # input is one-indexed
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 이미지와 관련 질문을 입력 받아 대응하는 답변을 생성하는 함수입니다.
    # 인자를 _dict 하나만 사용하는 이유는, RAG 체인을 구성할 때 추가 Adapter 없이 사용하기 위함입니다.
    def query_about_image(_dict):
        query = _dict["query"]
        base64_image = _dict["base64_image"]

        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        )
        response = llm.invoke([message])
        return response.content

    page_summary_texts = []
    base64_images = []

    is_loading = force_reload or not os.path.exists(documents_dir + "/vectorstore")
    for root, dirs, files in os.walk(documents_dir):
        for file in files:
            if file.endswith(".pdf"):
                page_count = fitz.open(os.path.join(root, file)).page_count
                for page_idx in range(1, page_count + 1):
                    base64_image = pdf_page_to_base64(
                        os.path.join(root, file), page_idx
                    )
                    if is_loading:
                        # [지시사항 2]. Base64 인코딩된 각 페이지에 대한 설명문을 생성하고, page_summary_texts에 추가하세요.
                        # Hint 1. query_about_image 함수를 사용하여 이미지와 관련된 질문을 생성합니다.
                        #################################################
                        query = "Describe the content of this document page in detail for retrieval."
                        response = query_about_image({"query": query, "base64_image": base64_image})
                        page_summary_texts.append(response)
                        #################################################

                    base64_images.append(base64_image)

    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="summaries",
        persist_directory=documents_dir + "/vectorstore",
    )

    store = InMemoryStore()
    id_key = "doc_id"

    # VectorDB에서 바로 Retriever를 생성하는 대신, 별도의 Retriever를 생성합니다.
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    # simialrity score threshold를 0.6으로 설정합니다.
    retriever.search_type = SearchType.similarity_score_threshold
    retriever.search_kwargs = {"score_threshold": 0.6}
    # Add Encoded Images
    if is_loading:
        img_ids = [str(uuid.uuid4()) for _ in base64_images]
        summary_images = [
            Document(page_content=s, metadata={id_key: img_ids[i]})
            for i, s in enumerate(page_summary_texts)
        ]
        retriever.vectorstore.add_documents(summary_images)
    else:
        img_ids = [
            metadata[id_key] for metadata in retriever.vectorstore.get()["metadatas"]
        ]
    retriever.docstore.mset(list(zip(img_ids, base64_images)))

    return retriever


def init_chroma_retriever(documents_dir, embeddings, force_reload: bool = False):
    if force_reload or not os.path.exists(documents_dir + "/vectorstore"):
        print("Building retriever...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)

        doc_list = []
        for root, dirs, files in os.walk(documents_dir):
            for file in files:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path=os.path.join(root, file))
                    doc = loader.load()
                    doc = text_splitter.split_documents(doc)
                    doc_list.extend(doc)

        vectorstore = Chroma.from_documents(
            doc_list,
            embedding=embeddings,
            persist_directory=documents_dir + "/vectorstore",
        )
    else:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=documents_dir + "/vectorstore",
        )
    retriever = vectorstore.as_retriever()
    return retriever
