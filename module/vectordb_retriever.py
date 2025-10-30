# 챗봇의 각 기능을 모듈로 나누어 구현합니다.
from typing import List

from langchain_core.documents import Document
from module import BaseRAGModule, State


# 챗봇의 각 기능을 모듈로 나누어 구현합니다.
class VectorDBRetrieverRAGModule(BaseRAGModule):
    node_id = "vector_db_retriever"

    def __init__(self, db_retriever, reranker):
        self.db_retriever = db_retriever
        self.reranker = reranker

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        return self.reranker.compress_documents(docs, query)

    def run(self, state: State) -> State:
        def get_retrieved_text(docs):
            return "\n".join([doc.page_content for doc in docs])

        print("---Vector DB Search---")
        question = state["question"]

        retrieval_chain = (
            self.db_retriever
            | (lambda docs: self._rerank(question, docs))
            | get_retrieved_text
        )
        state["data"] = retrieval_chain.invoke(question)
        return state


# 챗봇의 각 기능을 모듈로 나누어 구현합니다.
class MultiModalVectorDBRetrieverRAGModule(BaseRAGModule):
    node_id = "vector_db_retriever"

    def __init__(self, db_retriever):
        self.db_retriever = db_retriever

    def run(self, state: State) -> State:
        def get_retrieved_base64_image(images):
            try:
                if len(images) > 0:
                    return images[0]
                else:
                    raise Exception("No Image Detected!")
            except Exception as e:
                print(f"Error: {e}")
            return None

        print("---Multimodal Vector DB Search---")
        question = state["question"]

        retrieval_chain = self.db_retriever | get_retrieved_base64_image
        state["base64_image"] = retrieval_chain.invoke(question)
        return state
