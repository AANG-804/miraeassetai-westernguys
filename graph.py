"""
LangGraph를 사용한 도구 기반 챗봇 구현
"""

import json
import os
from typing import Annotated, List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from db_configuration import create_vector_db_tool

# 환경 변수 로드
load_dotenv()


class State(TypedDict):
    """그래프 상태를 정의하는 타입"""
    messages: Annotated[List[BaseMessage], add_messages]


# Tool 생성
## 위키피디아 도구
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=2)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

## 타빌리 검색 도구
tavily_tool = TavilySearch(max_results=2)

## MCP 도구
from langchain_mcp_adapters.client import MultiServerMCPClient
import json
import asyncio

async def get_mcp_tools():
    """MCP 도구들을 비동기적으로 가져오기"""
    try:
        ### config 파일 읽어들이기
        with open("mcp-server-config.json", "r") as f:
            mcp_config = json.load(f)

        ### print
        print(mcp_config)

        mcp_client = MultiServerMCPClient(
            mcp_config["mcpServers"]
        )

        mcp_tools = await mcp_client.get_tools()
        return mcp_tools
    except Exception as e:
        print(f"MCP 도구 로딩 중 오류 발생: {e}")
        return []



# 기본 설정
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can use tools to answer questions."

# 기본 도구들 (MCP 도구는 지연 로딩)
BASE_TOOLS = [wiki_tool, tavily_tool]

# MCP 도구 캐시
_mcp_tools_cache = None

def get_available_tools():
    """사용 가능한 도구들을 동적으로 가져오기"""
    global _mcp_tools_cache
    tools = BASE_TOOLS.copy()
    
    # MCP 도구 로딩 시도 (캐시된 결과 사용)
    if _mcp_tools_cache is None:
        try:
            _mcp_tools_cache = asyncio.run(get_mcp_tools())
        except Exception as e:
            print(f"MCP 도구 로딩 실패: {e}")
            _mcp_tools_cache = []
    
    if _mcp_tools_cache:
        tools.extend(_mcp_tools_cache)
    
    # 벡터 DB 도구 생성 시도
    # vector_db_tool = create_vector_db_tool()
    # if vector_db_tool is not None:
    #     tools.append(vector_db_tool)
    
    return tools


def build_graph(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: List = None
):
    """
    LangGraph 애플리케이션 빌더
    
    Args:
        model: 사용할 모델명
        temperature: 모델의 temperature 설정
        system_prompt: 시스템 프롬프트
        tools: 사용할 도구 리스트
    
    Returns:
        컴파일된 그래프
    """
    if tools is None:
        tools = get_available_tools()
    
    # None 값 필터링
    tools = [tool for tool in tools if tool is not None]
    
    # LLM 생성
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    ).bind_tools(tools)
    
    # 챗봇 노드 정의
    def chatbot_node(state: State) -> State:
        response = llm.invoke(
            [SystemMessage(content=system_prompt)] +
            state["messages"]
        )
        return {"messages": [response]}
    
    # 라우터 정의
    def router(state: State) -> str:
        last_message = state["messages"][-1]
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return END
        else:
            return "tools"
    
    # 그래프 빌더 생성
    builder = StateGraph(State)
    
    # 노드 추가
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", ToolNode(tools))
    
    # 엣지 추가
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", router, ["tools", END])
    builder.add_edge("tools", "chatbot")
    
    return builder.compile()


# 기본 그래프 생성 (지연 실행)
graph = build_graph()




