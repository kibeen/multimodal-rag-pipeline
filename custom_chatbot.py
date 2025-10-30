import os

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph
from module import (
    LLMAnswerRAGModule,
    MultiModalLLMAnswerRAGModule,
    MultiModalVectorDBRetrieverRAGModule,
    SelfReflectionRAGModule,
    State,
    VectorDBRetrieverRAGModule,
    WebRetrieverRAGModule,
    init_chroma_retriever,
    init_multimodal_retriever,
)


class ModularRAG:
    def __init__(
        self, documents_dir, documents_description, force_reload: bool = False
    ):
        self.llm = ChatOllama(model="minicpm-v", temperature=0)
        self.route_llm = ChatOllama(model="minicpm-v", format="json", temperature=0)
        self.embeddings = OllamaEmbeddings(model="minicpm-v")

        self.top_n_reranked_docs = 3
        self.cross_encoder = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-v2-m3"
        )
        self.reranker = CrossEncoderReranker(
            model=self.cross_encoder, top_n=self.top_n_reranked_docs
        )

        self.db_retriever = init_multimodal_retriever(
            documents_dir, self.embeddings, self.llm, force_reload
        )
        self.documents_description = documents_description

        self.web_retriever = WebRetrieverRAGModule()

        self.vector_db_retriever = MultiModalVectorDBRetrieverRAGModule(
            self.db_retriever
        )
        self.llm_answer = MultiModalLLMAnswerRAGModule(self.llm)
        self.self_reflection = SelfReflectionRAGModule(self.llm, self.route_llm)

        # Build workflow graph
        self.workflow = StateGraph(State)
        # Add nodes: our modules and a lambda node for plain answer generation
        self.workflow.add_node("web_retriever", self.web_retriever)
        self.workflow.add_node("vector_db_retriever", self.vector_db_retriever)
        self.workflow.add_node("llm_answer", self.llm_answer)
        self.workflow.add_node(
            "plain_answer",
            lambda state: {
                **state,
                "generation": self.llm.invoke(state["question"]).content,
            },
        )
        # Set entry point to start with data retrieval
        self.workflow.set_entry_point("vector_db_retriever")

        # Conditional edge after vector DB retrieval: if data is relevance then go to llm_answer, else try web retrieval
        self.workflow.add_conditional_edges(
            "vector_db_retriever",
            lambda state: "yes" if self._is_data_relevance(state) else "no",
            {"yes": "llm_answer", "no": "web_retriever"},
        )
        # Conditional edge after web retrieval: if relevance is confirmed, go to llm_answer; otherwise use plain answer.
        self.workflow.add_conditional_edges(
            "web_retriever",
            lambda state: "yes" if self._is_data_relevance(state) else "no",
            {"yes": "llm_answer", "no": "plain_answer"},
        )
        # Both answer nodes lead to termination.
        self.workflow.add_edge("llm_answer", END)
        self.workflow.add_edge("plain_answer", END)

        self.workflow = self.workflow.compile()

    # 정보 평가
    def _is_data_relevance(self, state: State) -> bool:
        # LLM이 생성한 텍스트가 문서와 관련이 있는지 확인합니다.
        print("--- is_data_relevance ---")
        question = state["question"]
        data = state["data"]
        base64_image = state["base64_image"]
        if data is not None and len(data) > 0:
            # 사용하는 LLM이 변경됨에 따라, 더 적합한 시스템 메시지를 입력하기 위해 `relevent`를 `relevance`로 변경했습니다.
            system_message = """You are an evaluator assessing the relevance between the retrieved document and the user's question.
    The following is the retrieved document: \n{data}\n.
    If the document is relevant to the user's question, select `yes`, otherwise select `no`.
    Respond with a JSON containing only the key 'relevance', and do not generate any other text or explanation."""
            message_list = [("system", system_message)]
            message_list.append(("human", "{question}"))
            relevance_judge_prompt = ChatPromptTemplate.from_messages(message_list)
            router_chain = relevance_judge_prompt | self.route_llm | JsonOutputParser()
            result = router_chain.invoke({"question": question, "data": data})
            print(result)
            return result.get("relevance", "no") in ["yes", "relevance", True]

        elif base64_image is not None and len(base64_image) > 0:
            query = "Verify given image is relevant to the user's question. If the image is relevant, select 'yes'; otherwise, select 'no'.\
                Respond with a JSON containing only the key 'relevance', and do not generate any other text or explanation."
            message = HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            )
            router_chain = self.route_llm | JsonOutputParser()
            result = router_chain.invoke([message])
            return result.get("relevance", "no") in ["yes", "relevance", True]

    def run(self, question: str) -> str:
        # Create the initial state and invoke the workflow
        init_state = {"question": question, "data": "", "code": "", "generation": ""}
        final_state = self.workflow.invoke(init_state)
        # Evaluate answer quality using the SelfReflection module
        quality = self.self_reflection.judge_answer(final_state)
        if quality not in ["yes", True]:
            print(
                "---Answer did not pass self-reflection. Falling back to plain answer.---"
            )
            final_state["generation"] = self.llm.invoke(question).content
        return final_state["generation"]
