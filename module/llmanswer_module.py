from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from module import BaseRAGModule, State


class MultiModalLLMAnswerRAGModule(BaseRAGModule):
    node_id = "multimodal_llm_answer"

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: State) -> State:
        question = state["question"]
        data = state.get("data", None)
        base64_image = state.get("base64_image", None)

        query = question
        # 이미지가 있다면 이미지 기반 답변을 먼저 생성
        if base64_image:
            # print("---멀티모달 데이터 기반 답변 생성---")
            print("---Multimodal Data-based Answer Generation---")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ]
            )
            #################################################
            response = self.llm.invoke([message])
            generation = response.content

        # 이미지는 없지만 다른 데이터가 있는 경우
        elif data:
            # print("---데이터 기반 답변 생성---")
            print("---Data-based Answer Generation---")

            reasoning_with_data = [
                (
                    "system",
                    "You are a data analyst answering questions based on the data. Answer the question based on the data provided by the user.",
                ),
                ("human", "Data: {data}\n{question}"),
            ]
            chain_data = (
                ChatPromptTemplate.from_messages(reasoning_with_data)
                | self.llm
                | StrOutputParser()
            )
            generation = chain_data.invoke({"data": data, "question": question})
        # 이미지와 데이터가 없는 경우
        else:
            print("---Answer Generation---")
            generation = self.llm.invoke(question).content

        return {
            "question": question,
            "code": state.get("code", ""),
            "data": data,
            "generation": generation,
            "base64_image": base64_image,
        }


class LLMAnswerRAGModule(BaseRAGModule):
    node_id = "llm_answer"

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: State) -> State:
        question = state["question"]
        data = state.get("data", "")

        if data:
            print("--- Data-based answer generation ---")
            # 첫 번째 데이터 기반 체인: answer_with_data 방식
            reasoning_with_data = [
                (
                    "system",
                    "You are a data analyst answering questions based on the data. Answer the question based on the data provided by the user.",
                ),
                ("human", "Data: {data}\n{question}"),
            ]
            chain_data = (
                ChatPromptTemplate.from_messages(reasoning_with_data)
                | self.llm
                | StrOutputParser()
            )
            generation = chain_data.invoke({"data": data, "question": question})

        else:
            # 데이터가 없는 경우 단순히 질문만으로 답변 생성 (plain answer)
            print("--- Answer generation ---")
            generation = self.llm.invoke(question).content

        return {
            "question": question,
            "code": state.get("code", ""),
            "data": data,
            "generation": generation,
        }
