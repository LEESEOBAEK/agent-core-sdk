"""
agent_core/tools/weather.py  —  Weather lookup tool
=====================================================
Returns dummy weather data for a city name.
Replace the dummy database with a real API (e.g. OpenWeatherMap) in production.
"""

import logging
from agent_core.observability.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer("tools.weather")

_WEATHER_DB = {
    "서울": {
        "city": "서울",
        "condition": "맑음",
        "temperature_c": 3,
        "feels_like_c": -1,
        "humidity_pct": 45,
        "wind_kmh": 12,
        "description": "맑고 건조한 겨울 날씨입니다. 체감온도가 낮으니 외출 시 방한에 유의하세요.",
    },
    "부산": {
        "city": "부산",
        "condition": "흐림",
        "temperature_c": 7,
        "feels_like_c": 4,
        "humidity_pct": 65,
        "wind_kmh": 18,
        "description": "해안가 특성상 바람이 강하고 습도가 높습니다.",
    },
    "도쿄": {
        "city": "도쿄",
        "condition": "비",
        "temperature_c": 10,
        "feels_like_c": 7,
        "humidity_pct": 80,
        "wind_kmh": 8,
        "description": "가벼운 비가 내리고 있습니다. 우산을 챙기세요.",
    },
    "뉴욕": {
        "city": "뉴욕",
        "condition": "눈",
        "temperature_c": -2,
        "feels_like_c": -8,
        "humidity_pct": 70,
        "wind_kmh": 20,
        "description": "눈이 내리고 있습니다. 도로가 미끄러울 수 있습니다.",
    },
}

_DEFAULT_WEATHER = {
    "condition": "맑음",
    "temperature_c": 15,
    "feels_like_c": 13,
    "humidity_pct": 55,
    "wind_kmh": 10,
    "description": "전형적인 날씨입니다.",
}


def get_weather(city: str) -> dict:
    """
    Return current weather information for a city.

    Parameters
    ----------
    city : str
        City name (e.g. "서울", "부산")

    Returns
    -------
    dict
        Weather data dictionary
    """
    with tracer.start_as_current_span("tool-call") as span:
        span.set_attribute("tool.name", "get_weather")
        span.set_attribute("tool.input.city", city)

        logger.info(f"[WeatherTool] lookup: {city}")

        data = _WEATHER_DB.get(city)
        if data is None:
            data = {**_DEFAULT_WEATHER, "city": city}
            logger.info(f"[WeatherTool] '{city}' not in DB → using defaults")

        result = {**data}
        span.set_attribute("tool.output.condition", result["condition"])
        span.set_attribute("tool.output.temperature_c", result["temperature_c"])
        span.set_attribute("tool.success", True)

        logger.info(
            f"[WeatherTool] result: {city} / {result['condition']} / {result['temperature_c']}°C"
        )
        return result
