from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from module import BaseConditionalEdgeModule, State


class SelfReflectionRAGModule(BaseConditionalEdgeModule):
    edge_id = "self_reflection"

    def __init__(self, llm, route_llm):
        self.llm = llm
        self.route_llm = route_llm

    def judge_answer(self, state: State) -> str:
        # print("---답변 퀄리티 검증 (Self-RAG)---")
        print("---Quality Check (Self-RAG)---")
        try:
            hallucinated = self._is_hallucinated(state)
        except KeyError:
            hallucinated = False
            print("---The truthfulness of the given answer cannot be determined.---")
        else:
            status = "not true" if hallucinated else "true"
            print(f"---The given answer is {status}.---")

        try:
            supportive = self._is_answer_supportive(state)
        except KeyError:
            supportive = True
            print("---The supportiveness of the given answer cannot be determined.---")
        else:
            status = "supportive" if supportive else "not supportive"
            print(f"---The given answer is {status}.---")

        try:
            useful = self._is_answer_useful(state)
        except KeyError:
            useful = True
            print("---The usefulness of the given answer cannot be determined.---")
        else:
            status = "useful" if useful else "not useful"
            print(f"---The given answer is {status}.---")

        if (supportive or useful) and hallucinated is False:
            return "yes"
        else:
            return "no"

    # 지지성 평가
    def _is_answer_supportive(self, state: State) -> bool:
        # 생성된 텍스트가 질문과 관련이 있는지 확인합니다.
        question = state["question"]
        generation = state["generation"]
        system_message = """You are an evaluator assessing whether the AI's response is supportive of the user's question."""
        # 다른 모듈과 달리, 정보와 AI의 답변을 모두 사용자 프롬프트에 추가합니다. 이는 실험을 통해 더 좋은 결과가 나와서 선택한 방법입니다.
        # 또한 함수 이름과 달리 'supportive'가 아니라 'answer' Key에 답변을 저장하라는 지시가 있습니다.
        # 이는 다른 지시사항에 있는 텍스트와 일관성을 유지하기 위함입니다.

        user_message = """User's Question: {question}
        AI's Response: {generation}
        If the AI's response is supportive of the user's question, select 'yes'; otherwise, select 'no'.
        Respond with a JSON containing only the 'answer' key, and do not generate any other text or explanation."""
        message_list = [("system", system_message)]
        message_list.append(("human", user_message))

        relevant_judge_prompt = ChatPromptTemplate.from_messages(message_list)
        # 로직 선택용 ChatOllama 객체를 생성합니다. format="json" 인자를 적용하여 출력 양식을 json으로 강제합니다.
        # 같은 질문에 항상 같은 대답을 유도하기 위해 temperature를 0으로 설정합니다.
        router_chain = relevant_judge_prompt | self.route_llm | JsonOutputParser()

        result = router_chain.invoke({"question": question, "generation": generation})
        if str(result["answer"]).lower() in ["yes", "true"]:
            return True
        return False

    # 유용성 평가
    def _is_answer_useful(self, state: State) -> bool:
        # 생성된 텍스트가 질문에 대한 해답인지 확인합니다.
        question = state["question"]
        generation = state["generation"]
        system_message = """You are an evaluator assessing the usefulness of the AI's response to the user."""
        user_message = """User's Question: {question}
        AI's Response: {generation}
        If the AI's response is useful to the user, select 'yes'; otherwise, select 'no'.
        Respond with a JSON containing only the 'useful' key, and do not generate any other text or explanation."""
        message_list = [("system", system_message)]
        message_list.append(("human", user_message))

        useful_judge_prompt = ChatPromptTemplate.from_messages(message_list)
        # 로직 선택용 ChatOllama 객체를 생성합니다. format="json" 인자를 적용하여 출력 양식을 json으로 강제합니다.
        # 같은 질문에 항상 같은 대답을 유도하기 위해 temperature를 0으로 설정합니다.
        router_chain = useful_judge_prompt | self.route_llm | JsonOutputParser()

        result = router_chain.invoke({"question": question, "generation": generation})
        if str(result["useful"]).lower() in ["yes", "true"]:
            return True
        return False

    # 할루시네이션 평가
    def _is_hallucinated(self, state: State) -> bool:
        # 생성된 텍스트가 질문에 대한 해답인지 확인합니다.
        generation = state["generation"]
        docs = state.get("data", None)
        base64_image = state.get("base64_image", None)
        # docs = state["data"]
        if docs:
            system_message = """You are an evaluator assessing the truthfulness of the AI's response based on the given source document."""
            user_message = """Source Document: {docs}
            AI's Response: {generation}
            If the AI's response is true based on the source document, select True; if not, select False.
            Respond with a JSON containing only the 'answer' key, and do not generate any other text or explanation."""
            message_list = [("system", system_message)]
            message_list.append(("human", user_message))

            hallucination_judge_prompt = ChatPromptTemplate.from_messages(message_list)
            # 로직 선택용 ChatOllama 객체를 생성합니다. format="json" 인자를 적용하여 출력 양식을 json으로 강제합니다.
            # 같은 질문에 항상 같은 대답을 유도하기 위해 temperature를 0으로 설정합니다.
            router_chain = (
                hallucination_judge_prompt | self.route_llm | JsonOutputParser()
            )

            result = router_chain.invoke({"generation": generation, "docs": docs})
            print(result)
            if str(result["answer"]).lower() in ["yes", "true"]:
                return False
            return True
        elif base64_image:
            query = "Verify the AI's answer is based on the given image. If the AI's answer is true, select 'yes'; otherwise, select 'no'."
            query += "Please provide your answer in JSON format with the 'answer' key only, and do not generate any other text or explanation."
            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": query},
                ],
            )

            router_chain = self.route_llm | JsonOutputParser()

            result = router_chain.invoke([message])
            print(result)
            if str(result["answer"]).lower() in ["yes", "true"]:
                return False
            return True
