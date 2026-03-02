"""Build and execute the OpenAI function-calling analyst agent."""

from __future__ import annotations

import json
from typing import Any

from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from autonomous_analyst.agent.prompts import SYSTEM_PROMPT
from autonomous_analyst.agent.tools import AnalysisContext, build_tools
from autonomous_analyst.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE


def build_agent_executor(context: AnalysisContext) -> AgentExecutor:
    """Build a function-calling agent executor with project tools."""
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY or None,
    )
    tools = build_tools(context)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


def run_agent_workflow(context: AnalysisContext) -> dict[str, Any]:
    """Execute agent workflow and return structured final output."""
    executor = build_agent_executor(context)
    instruction = (
        f"Analyze the uploaded dataset using target column '{context.target_column}'. "
        "Run all tools in required order and return JSON only."
    )

    result = executor.invoke({"input": instruction})
    _ = result.get("output", "")

    if context.training_result is None:
        raise RuntimeError("Agent did not complete model training step.")
    if not context.explanation:
        context.explanation = (
            f"Model {context.training_result.model_name} completed with metrics "
            f"{context.training_result.metrics}."
        )

    # Build response from verified tool outputs for deterministic API behavior.
    compact_eda = {
        "shape": context.eda_summary.get("shape", {}),
        "numeric_columns": context.eda_summary.get("numeric_columns", []),
        "categorical_columns": context.eda_summary.get("categorical_columns", []),
        "missing_values": context.eda_summary.get("null_count", {}),
    }

    return {
        "eda_summary": json.dumps(compact_eda, ensure_ascii=True),
        "plots": context.plot_paths,
        "problem_type": context.training_result.problem_type,
        "model_used": context.training_result.model_name,
        "metrics": context.training_result.metrics,
        "preprocessing_summary": context.training_result.preprocessing_summary,
        "explanation": context.explanation,
    }
