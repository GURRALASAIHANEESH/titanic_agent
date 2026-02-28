# ==============================================================
# FILE: titanic_agent/backend/agent.py
# PURPOSE: LangChain ReAct agent — compatible with langchain 0.3.x + Groq
# ==============================================================

import logging
from functools import lru_cache
from typing import Optional

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq

from config import settings
from tools import ALL_TOOLS, get_last_chart_b64, reset_chart_buffer

logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# System prompt
# --------------------------------------------------------------

SYSTEM_PROMPT = """You are TitanicBot, a specialized data analyst for the Titanic passenger dataset.

SCOPE:
- Answer ONLY questions related to the Titanic dataset and its passengers.
- Politely decline any unrelated questions.

BEHAVIOR RULES:
1. NEVER fabricate statistics. Always use a tool to compute answers dynamically.
2. When a question asks for a chart, histogram, plot, graph or visualization, call the appropriate plotting tool.
3. After calling a plotting tool, provide a concise text summary of the key insight.
4. Compute all percentages, averages and counts using tools — never guess.
5. If a query is ambiguous, infer the most reasonable Titanic-related interpretation.
6. If a query cannot be answered from the dataset, say so clearly.
7. Keep responses factual, concise and well-structured.

TOOL SELECTION GUIDE:
- "histogram of [column]"        -> plot_histogram
- "bar chart of [column]"        -> plot_bar_chart
- "survival by [column]"         -> plot_survival_by_category
- "correlation" or "heatmap"     -> plot_correlation_heatmap
- "age by class"                 -> plot_age_by_class
- Gender statistics              -> get_gender_stats
- Age statistics                 -> get_age_stats
- Fare statistics                -> get_fare_stats
- Embarkation / port statistics  -> get_embarkation_stats
- Survival statistics            -> get_survival_stats
- Family / alone statistics      -> get_family_stats
- Dataset overview               -> get_dataset_summary
- Custom or complex computation  -> run_dataframe_query"""


# --------------------------------------------------------------
# ReAct prompt — pulled from LangChain hub and system prompt injected
# --------------------------------------------------------------

def _build_react_prompt():
    from langchain_core.prompts import PromptTemplate

    template = """You are TitanicBot, a specialized data analyst for the Titanic passenger dataset.

SCOPE:
- Answer questions related to the Titanic dataset, its passengers, and general
  Titanic historical facts (ship dimensions, the disaster, voyage details).
- For historical facts not in the dataset, answer from general knowledge.
- Decline questions unrelated to Titanic entirely (movies, fiction, etc.).

RULES:
1. NEVER fabricate statistics. Always use a tool to compute answers dynamically.
2. When a question asks for a chart, histogram, plot, graph or visualization, call the appropriate plotting tool.
3. After calling a plotting tool, provide a concise text summary of the key insight.
4. Compute all percentages, averages and counts using tools.
5. Keep responses factual and concise.

TOOL SELECTION GUIDE:
- "histogram of [column]"        -> plot_histogram(column)
- "bar chart of [column]"        -> plot_bar_chart(column)
- "survival by [column]"         -> plot_survival_by_category(column)
- "correlation" or "heatmap"     -> plot_correlation_heatmap
- "age by class"                 -> plot_age_by_class
- Gender stats                   -> get_gender_stats
- Age stats                      -> get_age_stats
- Fare stats                     -> get_fare_stats
- Embarkation stats              -> get_embarkation_stats
- Survival stats                 -> get_survival_stats
- Family stats                   -> get_family_stats
- Dataset overview               -> get_dataset_summary
- Custom computation             -> run_dataframe_query(expression)

You have access to the following tools:

{tools}

Use the following format STRICTLY:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    return PromptTemplate.from_template(template)


# --------------------------------------------------------------
# Agent construction — cached after first build
# --------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_executor() -> AgentExecutor:
    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        temperature=settings.TEMPERATURE,
        
        api_key=settings.GROQ_API_KEY,
    )

    prompt = _build_react_prompt()
    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=settings.AGENT_VERBOSE,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


# --------------------------------------------------------------
# Public agent interface
# --------------------------------------------------------------

class TitanicAgent:
    def __init__(self) -> None:
        self._executor = _build_executor()

    def run(self, question: str) -> tuple[str, Optional[str]]:
        """
        Process a user question through the ReAct agent.
        Returns (answer, chart_base64_or_None).
        """
        reset_chart_buffer()

        try:
            result = self._executor.invoke({"input": question})
            answer: str = result.get("output", "No answer was returned by the agent.")
        except Exception as exc:
            logger.exception("Agent execution failed: %s", exc)
            answer = (
                "An error occurred while processing your question. "
                "Please rephrase it or try a simpler query."
            )

        chart_b64 = get_last_chart_b64()
        logger.info(
            "Agent completed. has_chart=%s, answer_length=%d",
            chart_b64 is not None,
            len(answer),
        )
        return answer, chart_b64
