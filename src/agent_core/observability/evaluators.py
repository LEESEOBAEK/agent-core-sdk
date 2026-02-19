"""
agent_core/observability/evaluators.py  —  LLM-as-Judge evaluation module
==========================================================================
Automatically evaluates RAG pipeline output quality on three dimensions:
  - Relevance    : Are retrieved docs relevant to the question?
  - Faithfulness : Is the answer grounded only in retrieved docs?
  - Correctness  : Does the answer match the ground truth?

Each evaluation is recorded as a separate 'eval.*' span in Phoenix.
Scores can also be posted to the Phoenix /v1/span_annotations REST API.

Usage:
    from agent_core.llm.client import LLMClient
    from agent_core.observability.evaluators import (
        evaluate_relevance, evaluate_faithfulness, evaluate_correctness,
        post_span_annotation,
    )

    judge = LLMClient(provider="gemini", model="gemini-2.0-flash")
    result = evaluate_relevance(question, context, judge)
    print(result.score, result.label, result.reasoning)
"""

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """
    Result of a single evaluation item.

    Attributes
    ----------
    score     : 0.0–1.0 (1.0 = best)
    label     : "pass" (score >= 0.5), "fail", or "error"
    reasoning : Explanation from the Judge LLM
    """
    score: float
    label: str
    reasoning: str = field(default="")

    @classmethod
    def error(cls, message: str) -> "EvalResult":
        return cls(score=0.0, label="error", reasoning=message)


def _parse_judge_response(response_text: str) -> tuple[float, str]:
    """
    Extract score and reasoning from a Judge LLM response.

    1st attempt : JSON parse  {"score": 0.8, "reasoning": "..."}
    2nd attempt : regex extract a decimal number (fallback)
    On failure  : return (0.0, original text)
    """
    json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reasoning = str(data.get("reasoning", ""))
            return score, reasoning
        except (json.JSONDecodeError, ValueError):
            pass

    numbers = re.findall(r'\b(0\.\d+|1\.0|[01])\b', response_text)
    if numbers:
        try:
            score = float(numbers[0])
            score = max(0.0, min(1.0, score))
            return score, response_text.strip()
        except ValueError:
            pass

    logger.warning(f"[Evaluator] Failed to parse judge response: {response_text[:200]}")
    return 0.0, response_text.strip()


def _score_to_label(score: float) -> str:
    return "pass" if score >= 0.5 else "fail"


def post_span_annotation(
    phoenix_endpoint: str,
    span_id: str,
    name: str,
    score: float,
    explanation: str = "",
    annotator_kind: str = "LLM",
) -> None:
    """
    Send an evaluation score to Phoenix /v1/span_annotations REST API.
    Scores appear in the Phoenix UI Annotations tab.

    Parameters
    ----------
    phoenix_endpoint : str
        Phoenix server address (e.g. "http://localhost:6006")
    span_id : str
        Span ID to annotate (16-char hex string)
    name : str
        Evaluation metric name (e.g. "relevance", "faithfulness")
    score : float
        0.0–1.0
    explanation : str
        Reasoning text (optional)
    annotator_kind : str
        "LLM" | "HUMAN" | "CODE" (default: "LLM")
    """
    try:
        import requests
    except ImportError:
        logger.debug("[Evaluator] requests not installed; skipping Phoenix Annotations.")
        return

    payload = {
        "data": [
            {
                "span_id": span_id,
                "name": name,
                "annotator_kind": annotator_kind,
                "result": {
                    "score": score,
                    "explanation": explanation[:1000] if explanation else "",
                },
            }
        ]
    }

    try:
        resp = requests.post(
            f"{phoenix_endpoint}/v1/span_annotations",
            json=payload,
            timeout=5,
        )
        if resp.status_code not in (200, 201, 204):
            logger.debug(
                f"[Evaluator] Phoenix Annotation response: {resp.status_code} - {resp.text[:200]}"
            )
    except Exception as e:
        logger.debug(f"[Evaluator] Phoenix Annotations send failed (ignored): {e}")


def evaluate_relevance(
    question: str,
    context: str,
    judge_llm,
) -> EvalResult:
    """
    Evaluate whether retrieved documents (context) are relevant to the question.

    Parameters
    ----------
    question  : User question
    context   : RAG retrieval result (output of retriever.build_context())
    judge_llm : LLMClient instance to use as judge

    Returns
    -------
    EvalResult
    """
    from agent_core.observability.tracing import get_tracer
    from opentelemetry.trace import StatusCode

    tracer = get_tracer("evaluators.relevance")

    with tracer.start_as_current_span("eval.relevance") as span:
        span.set_attribute("eval.metric", "relevance")
        span.set_attribute("eval.question", question)
        span.set_attribute("eval.context_preview", context[:500])

        judge_prompt = (
            "당신은 RAG 시스템의 검색 품질을 평가하는 전문가입니다.\n\n"
            f"[질문]\n{question}\n\n"
            f"[검색된 문서]\n{context}\n\n"
            "위 검색 결과가 질문에 답변하는 데 얼마나 관련이 있는지 평가하세요.\n\n"
            "평가 기준:\n"
            "  1.0 = 검색 문서가 질문에 직접적으로 관련된 정보를 포함\n"
            "  0.5 = 부분적으로 관련 있음\n"
            "  0.0 = 검색 문서가 질문과 전혀 관련 없음\n\n"
            "반드시 다음 JSON 형식으로만 답변하세요 (다른 텍스트 없이):\n"
            '{"score": 0.0에서 1.0 사이의 숫자, "reasoning": "판단 근거를 한 문장으로"}'
        )

        try:
            response = judge_llm.chat(prompt=judge_prompt)
            score, reasoning = _parse_judge_response(response)
            label = _score_to_label(score)

            span.set_attribute("eval.score", score)
            span.set_attribute("eval.label", label)
            span.set_attribute("eval.reasoning", reasoning[:500])

            logger.info(f"[Relevance] score={score:.2f}, label={label}")
            return EvalResult(score=score, label=label, reasoning=reasoning)

        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"[Relevance] Evaluation failed: {e}")
            return EvalResult.error(str(e))


def evaluate_faithfulness(
    context: str,
    answer: str,
    judge_llm,
) -> EvalResult:
    """
    Evaluate whether the LLM answer is grounded only in the retrieved context.

    Parameters
    ----------
    context   : RAG retrieval result
    answer    : Final answer generated by the RAG pipeline
    judge_llm : LLMClient instance to use as judge

    Returns
    -------
    EvalResult
    """
    from agent_core.observability.tracing import get_tracer
    from opentelemetry.trace import StatusCode

    tracer = get_tracer("evaluators.faithfulness")

    with tracer.start_as_current_span("eval.faithfulness") as span:
        span.set_attribute("eval.metric", "faithfulness")
        span.set_attribute("eval.context_preview", context[:500])
        span.set_attribute("eval.answer_preview", answer[:500])

        judge_prompt = (
            "당신은 RAG 시스템의 답변 충실성을 평가하는 전문가입니다.\n\n"
            f"[참고 문서]\n{context}\n\n"
            f"[AI 답변]\n{answer}\n\n"
            "위 AI 답변이 참고 문서의 내용에만 근거하고 있는지 평가하세요.\n"
            "참고 문서에 없는 내용을 답변에 포함하면 할루시네이션(환각)입니다.\n\n"
            "평가 기준:\n"
            "  1.0 = 답변의 모든 내용이 참고 문서에 근거함\n"
            "  0.5 = 일부 내용은 문서 근거, 일부는 문서 외 정보\n"
            "  0.0 = 답변 대부분이 참고 문서와 무관한 정보 (심각한 할루시네이션)\n\n"
            "반드시 다음 JSON 형식으로만 답변하세요 (다른 텍스트 없이):\n"
            '{"score": 0.0에서 1.0 사이의 숫자, "reasoning": "판단 근거를 한 문장으로"}'
        )

        try:
            response = judge_llm.chat(prompt=judge_prompt)
            score, reasoning = _parse_judge_response(response)
            label = _score_to_label(score)

            span.set_attribute("eval.score", score)
            span.set_attribute("eval.label", label)
            span.set_attribute("eval.reasoning", reasoning[:500])

            logger.info(f"[Faithfulness] score={score:.2f}, label={label}")
            return EvalResult(score=score, label=label, reasoning=reasoning)

        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"[Faithfulness] Evaluation failed: {e}")
            return EvalResult.error(str(e))


def evaluate_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    judge_llm,
) -> EvalResult:
    """
    Evaluate whether the LLM answer matches the ground truth.

    Parameters
    ----------
    question     : User question
    answer       : Final answer generated by the RAG pipeline
    ground_truth : Expected correct answer
    judge_llm    : LLMClient instance to use as judge

    Returns
    -------
    EvalResult
    """
    from agent_core.observability.tracing import get_tracer
    from opentelemetry.trace import StatusCode

    tracer = get_tracer("evaluators.correctness")

    with tracer.start_as_current_span("eval.correctness") as span:
        span.set_attribute("eval.metric", "correctness")
        span.set_attribute("eval.question", question)
        span.set_attribute("eval.answer_preview", answer[:500])
        span.set_attribute("eval.ground_truth_preview", ground_truth[:500])

        judge_prompt = (
            "당신은 RAG 시스템의 답변 정확성을 평가하는 전문가입니다.\n\n"
            f"[질문]\n{question}\n\n"
            f"[AI 답변]\n{answer}\n\n"
            f"[정답(Ground Truth)]\n{ground_truth}\n\n"
            "AI 답변이 정답의 핵심 내용을 얼마나 정확하게 포함하는지 평가하세요.\n"
            "표현이 달라도 의미가 같으면 정확한 것으로 봅니다.\n\n"
            "평가 기준:\n"
            "  1.0 = AI 답변이 정답의 핵심 내용을 모두 포함하고 정확함\n"
            "  0.5 = 핵심 내용의 일부만 포함하거나 부분적으로 정확함\n"
            "  0.0 = AI 답변이 정답과 다르거나 핵심 내용을 포함하지 않음\n\n"
            "반드시 다음 JSON 형식으로만 답변하세요 (다른 텍스트 없이):\n"
            '{"score": 0.0에서 1.0 사이의 숫자, "reasoning": "판단 근거를 한 문장으로"}'
        )

        try:
            response = judge_llm.chat(prompt=judge_prompt)
            score, reasoning = _parse_judge_response(response)
            label = _score_to_label(score)

            span.set_attribute("eval.score", score)
            span.set_attribute("eval.label", label)
            span.set_attribute("eval.reasoning", reasoning[:500])

            logger.info(f"[Correctness] score={score:.2f}, label={label}")
            return EvalResult(score=score, label=label, reasoning=reasoning)

        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"[Correctness] Evaluation failed: {e}")
            return EvalResult.error(str(e))
