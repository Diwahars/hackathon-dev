
# agent.py

import json

from bifrost_client import chat, MODEL
from tavily_client import search_web
from freshness import detect_stale_answer
from observability import Trace


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search current web information using Tavily. "
                "Use for latest releases, changing facts, "
                "current prices, and recent events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]


TOOL_MAP = {
    "search_web": search_web,
}


def run_agent(user_query: str):

    trace = Trace(user_query)

    root = trace.start_span(
        name="agent.run",
        kind="agent",
        attributes={
            "agent.name": "research-agent",
            "agent.max_iterations": 6,
        },
    )

    messages = [
        {
            "role": "system",
            "content": """
You are an AI research agent.

Use search_web for current or changing information.
Use tools when useful.
Never invent tool results.
Give a concise, evidence-based answer.
Do not expose private chain-of-thought.
""",
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    try:

        for step in range(6):

            # --------------------------------------
            # Bifrost LLM span
            # --------------------------------------

            with trace.span(
                name="bifrost.chat_completion",
                kind="llm",
                parent_span=root,
                attributes={
                    "llm.provider": "bifrost",
                    "llm.model": MODEL,
                    "agent.step": step,
                },
            ) as llm_span:

                response = chat(
                    messages=messages,
                    tools=TOOLS,
                )

                message = response.choices[0].message

                usage = response.usage

                llm_span.set_attributes(
                    **{
                        "llm.response_model": response.model,
                        "llm.prompt_tokens": (
                            usage.prompt_tokens
                            if usage else 0
                        ),
                        "llm.completion_tokens": (
                            usage.completion_tokens
                            if usage else 0
                        ),
                        "llm.finish_reason": (
                            response.choices[0].finish_reason
                        ),
                    }
                )

            # --------------------------------------
            # Agent decision span
            # --------------------------------------

            with trace.span(
                name="agent.tool_selection",
                kind="agent",
                parent_span=root,
                attributes={
                    "agent.step": step,
                },
            ) as decision_span:

                if message.tool_calls:

                    selected_tools = [
                        call.function.name
                        for call in message.tool_calls
                    ]

                    decision_span.set_attributes(
                        **{
                            "decision.action": "execute_tool",
                            "decision.selected_tools": (
                                selected_tools
                            ),
                            "decision.reason": (
                                "The model requested "
                                "one or more tools."
                            ),
                        }
                    )

                else:

                    decision_span.set_attributes(
                        **{
                            "decision.action": "final_answer",
                            "decision.reason": (
                                "The model returned a "
                                "final response without "
                                "requesting tools."
                            ),
                        }
                    )

            # --------------------------------------
            # Final answer and freshness detection
            # --------------------------------------

            if not message.tool_calls:

                answer = message.content or ""

                with trace.span(
                    name="agent.final_answer",
                    kind="agent",
                    parent_span=root,
                    attributes={
                        "answer.length": len(answer),
                    },
                ) as answer_span:

                    answer_span.event(
                        "answer_generated",
                    )

                with trace.span(
                    name="freshness.evaluate",
                    kind="freshness",
                    parent_span=root,
                    attributes={
                        "freshness.query": user_query,
                    },
                ) as freshness_span:

                    freshness = detect_stale_answer(
                        query=user_query,
                        answer=answer,
                        trace=trace,
                        parent_span=freshness_span,
                    )

                    freshness_span.set_attributes(
                        **{
                            "freshness.stale": (
                                freshness.get("stale", False)
                            ),
                            "freshness.score": (
                                freshness.get("score", 0)
                            ),
                            "freshness.time_sensitive": (
                                freshness.get(
                                    "time_sensitive",
                                    False,
                                )
                            ),
                        }
                    )

                root.end(
                    status="success",
                    **{
                        "agent.status": "completed",
                    },
                )

                trace.finish(answer)

                return {
                    "trace_id": trace.trace_id,
                    "answer": answer,
                    "freshness": freshness,
                    "trace": trace.to_dict(),
                }

            # --------------------------------------
            # Tool execution
            # --------------------------------------

            messages.append(
                message.model_dump(
                    exclude_none=True,
                )
            )

            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name

                arguments = json.loads(
                    tool_call.function.arguments
                )

                tool_span = trace.start_span(
                    name=f"tool.{tool_name}",
                    kind="tool",
                    parent_span=root,
                    attributes={
                        "tool.name": tool_name,
                        "tool.call_id": tool_call.id,
                        "tool.arguments": arguments,
                    },
                )

                try:

                    tool_function = TOOL_MAP.get(
                        tool_name
                    )

                    if tool_function is None:
                        raise ValueError(
                            f"Unknown tool: {tool_name}"
                        )

                    result = tool_function(
                        **arguments
                    )

                    tool_span.end(
                        status="success",
                        **{
                            "tool.result_count": (
                                len(result.get("results", []))
                                if isinstance(result, dict)
                                else 0
                            ),
                        },
                    )

                except Exception as exc:

                    result = {
                        "error": str(exc),
                    }

                    tool_span.end(
                        status="error",
                        **{
                            "error.message": str(exc),
                        },
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

        raise RuntimeError(
            "Agent reached maximum iterations"
        )

    except Exception as exc:

        root.end(
            status="error",
            **{
                "error.message": str(exc),
            },
        )

        trace.finish(
            answer="",
            status="failed",
        )

        raise
