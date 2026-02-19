"""
agent_core/tools/search.py  —  Search simulation tool
=======================================================
Simulates DuckDuckGo-style search.
Replace with a real search API (e.g. duckduckgo-search) in production.
"""

import logging

from agent_core.observability.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer("tools.search")

_SEARCH_DB = {
    "날씨": [
        {
            "title": "오늘의 날씨 예보 - 기상청",
            "url": "https://example.com/weather",
            "snippet": "전국 주요 도시의 현재 날씨와 주간 예보를 확인하세요.",
        },
        {
            "title": "서울 날씨 - 맑음, 최저 -2°C 최고 7°C",
            "url": "https://example.com/seoul-weather",
            "snippet": "서울 오늘 날씨: 맑음. 아침 최저 -2도, 낮 최고 7도 예상.",
        },
    ],
    "파이썬": [
        {
            "title": "Python 공식 문서",
            "url": "https://docs.python.org/ko/3/",
            "snippet": "Python 3 공식 문서입니다. 튜토리얼, 라이브러리 레퍼런스, API 문서를 제공합니다.",
        },
        {
            "title": "파이썬 튜토리얼 - 점프 투 파이썬",
            "url": "https://example.com/python-tutorial",
            "snippet": "파이썬을 처음 배우는 분들을 위한 쉬운 튜토리얼입니다.",
        },
    ],
    "LLM": [
        {
            "title": "대형 언어 모델(LLM)이란? - 개요",
            "url": "https://example.com/llm-overview",
            "snippet": "LLM은 대규모 텍스트 데이터로 훈련된 딥러닝 모델로, GPT, Claude, Gemini 등이 있습니다.",
        },
        {
            "title": "Ollama - 로컬 LLM 실행 가이드",
            "url": "https://ollama.com",
            "snippet": "Ollama를 사용하면 llama3, mistral 등 오픈소스 LLM을 로컬에서 실행할 수 있습니다.",
        },
    ],
    "OpenTelemetry": [
        {
            "title": "OpenTelemetry 공식 사이트",
            "url": "https://opentelemetry.io",
            "snippet": "OpenTelemetry는 분산 시스템의 관측 가능성(트레이싱, 메트릭, 로그)을 위한 오픈 표준입니다.",
        },
    ],
}


def search(query: str, max_results: int = 3) -> list[dict]:
    """
    Return search results for a query.

    Parameters
    ----------
    query : str
        Search query
    max_results : int
        Maximum number of results to return (default: 3)

    Returns
    -------
    list[dict]
        Each item has title, url, snippet keys
    """
    with tracer.start_as_current_span("tool-call") as span:
        span.set_attribute("tool.name", "search")
        span.set_attribute("tool.input.query", query)
        span.set_attribute("tool.input.max_results", max_results)

        logger.info(f"[SearchTool] query: '{query}'")

        results = []
        query_lower = query.lower()
        for keyword, items in _SEARCH_DB.items():
            if keyword.lower() in query_lower or query_lower in keyword.lower():
                results.extend(items)

        if not results:
            results = [
                {
                    "title": f"'{query}' 검색 결과",
                    "url": "https://example.com/no-results",
                    "snippet": f"'{query}'에 대한 시뮬레이션 검색 결과입니다. 실제 정보는 검색 엔진에서 확인하세요.",
                }
            ]

        results = results[:max_results]
        span.set_attribute("tool.output.result_count", len(results))
        span.set_attribute("tool.success", True)

        logger.info(f"[SearchTool] {len(results)} results returned")
        return results
