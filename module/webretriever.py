# 챗봇의 각 기능을 모듈로 나누어 구현합니다.
import os

import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from module import BaseRAGModule, State


class WebRetrieverRAGModule(BaseRAGModule):
    node_id = "web_retriever"

    def __init__(self):
        if "TAVILY_API_KEY" not in os.environ:
            os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

    def run(self, state: State) -> State:
        print("---Web Search---")
        tavily_search_tool = TavilySearchResults(max_results=1)
        results = tavily_search_tool.invoke({"query": state["question"]})
        contents_list = [r["content"] for r in results]
        state["data"] = contents_list
        return state
