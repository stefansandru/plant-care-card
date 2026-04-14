#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph RAG pipeline for generating PlantCareCards using Mistral AI API.

Architecture: Research → Generate → Validate → (loop or end)
"""

import os
from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

from fastapi.logger import logger

from plant_care_card import PlantCareCard


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESEARCH_PROMPT = """You are a botanical researcher gathering plant care information.

Generate 3 targeted search queries to find:
1. Basic botanical info (scientific name, family, native habitat)
2. Core care requirements (light, water, soil, temperature, humidity)
3. Common problems (pests, diseases, toxicity)
"""

CARD_GENERATION_PROMPT = """You are a world-class botanist creating a structured plant care reference card.

Based on the research provided, generate a complete PlantCareCard.

Ignore **Validation Feedback:** section when it contains no content.

Be conservative and precise. If you're unsure about any aspect or if the report lacks necessary information, say "I don't have enough information to confidently assess this."
"""

VALIDATION_PROMPT = """You are a horticultural expert reviewing a plant care card for accuracy and completeness.

Check for:
1. **Accuracy**: Botanical name correct? Care requirements safe and appropriate?
2. **Completeness**: Any critical care info missing? (light/water/soil/temp/toxicity)
3. **Specificity**: Are instructions actionable? (avoid vague terms like "moderate")
4. **Consistency**: Do recommendations align? (e.g., "full sun" + "indoors" = possible issue)

Provide:
- List of specific errors to fix
- Missing critical information to add
- Recommendations for improvement

If the card is accurate and complete, respond with: "APPROVED - No changes needed"
"""


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    plant: str
    plant_care_card: PlantCareCard
    validation_feedback: Optional[str]
    content: List[str]
    revision_number: int
    max_revisions: int


# ---------------------------------------------------------------------------
# Helper: Queries schema for structured output
# ---------------------------------------------------------------------------

class Queries(BaseModel):
    queries: List[str]


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def _get_llm():
    """Create a Mistral LLM instance."""
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "mistral-small-latest")
    return ChatMistralAI(
        model=model,
        temperature=0.3,
        api_key=api_key,
    )


def _get_tavily():
    """Create a Tavily search client."""
    return TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))


def research_node(state: AgentState):
    llm = _get_llm()
    tavily = _get_tavily()

    logger.info("RESEARCHER: Planning search queries...")
    messages = [
        SystemMessage(content=RESEARCH_PROMPT),
        HumanMessage(
            content=f"Generate search queries for gathering information about the plant: {state['plant']}"
        ),
    ]
    queries = llm.with_structured_output(Queries).invoke(messages)

    logger.info(f"RESEARCHER: Search queries planned: {queries.queries}. Searching...")
    content = state.get("content", [])

    for i, q in enumerate(queries.queries, 1):
        logger.info(f"  Query {i}/{len(queries.queries)}: {q}")
        response = tavily.search(query=q, max_results=2)

        for r in response["results"]:
            source_url = r.get("url", "N/A")
            content.append(f"Source: {source_url}\n{r['content']}")

    logger.info(f"RESEARCHER: Search completed. {len(content)} results collected.")
    return {"content": content}


def generate_card_node(state: AgentState):
    llm = _get_llm()

    logger.info("GENERATOR: Generating plant care card...")
    content = "\n\n".join(state.get("content", []))

    validation_feedback = state.get("validation_feedback", "")
    feedback_section = ""
    if validation_feedback:
        feedback_section = f"\n\n**CRITICAL - Address these issues:**\n{validation_feedback}"

    messages = [
        SystemMessage(content=CARD_GENERATION_PROMPT),
        HumanMessage(
            content=f"""Create a PlantCareCard for: **{state['plant']}**

            **Research Content:**
            {content}

            **Validation Feedback:**
            {feedback_section}
            """
        ),
    ]
    plant_care_card = llm.with_structured_output(PlantCareCard).invoke(messages)
    logger.info(f"CARD GENERATOR: Card created for {plant_care_card.common_name}")
    return {
        "plant_care_card": plant_care_card,
        "revision_number": state.get("revision_number", 0) + 1,
    }


def validate_node(state: AgentState):
    llm = _get_llm()

    logger.info("VALIDATOR: Reviewing PlantCareCard...")
    card = state["plant_care_card"]

    messages = [
        SystemMessage(content=VALIDATION_PROMPT),
        HumanMessage(
            content=f"""Review this PlantCareCard for **{state['plant']}**:

{card.model_dump_json(indent=2)}
"""
        ),
    ]

    response = llm.invoke(messages)

    if "APPROVED" in response.content.upper():
        logger.info("VALIDATOR: PlantCareCard approved ✓")
        return {"validation_feedback": None}

    logger.info("VALIDATOR: Issues found, requesting revision")
    return {"validation_feedback": response.content}


def should_continue(state: AgentState) -> str:
    """Determines next step: re-research, revise, or end."""
    if state.get("validation_feedback") is None:
        logger.info("ROUTER: PlantCareCard approved. Ending.")
        return END

    if state["revision_number"] >= state["max_revisions"]:
        logger.info(
            f"ROUTER: Max revisions ({state['max_revisions']}) reached. Ending."
        )
        return END

    feedback = state.get("validation_feedback", "").lower()
    missing_keywords = ["missing", "lack", "insufficient", "no information", "not found"]

    if any(keyword in feedback for keyword in missing_keywords):
        logger.info("ROUTER: Missing info detected. Re-researching...")
        return "research"

    logger.info(
        f"ROUTER: Revising card (revision {state['revision_number']}/{state['max_revisions']})..."
    )
    return "revise"


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def _build_graph():
    """Build and compile the LangGraph state machine."""
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("researcher", research_node)
    graph_builder.add_node("generator", generate_card_node)
    graph_builder.add_node("validator", validate_node)

    graph_builder.set_entry_point("researcher")

    graph_builder.add_edge("researcher", "generator")
    graph_builder.add_edge("generator", "validator")

    graph_builder.add_conditional_edges(
        "validator",
        should_continue,
        {
            "revise": "generator",
            "research": "researcher",
            END: END,
        },
    )

    return graph_builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_plant_care_card(
    plant_name: str, max_revisions: int = 2
) -> PlantCareCard:
    """
    Run the full RAG pipeline: research → generate → validate → (loop or end).

    Args:
        plant_name: Common name of the plant (e.g., "coconut", "tomato").
        max_revisions: Maximum number of revision cycles before stopping.

    Returns:
        A validated PlantCareCard instance.
    """
    graph = _build_graph()

    initial_state = {
        "plant": plant_name,
        "content": [],
        "revision_number": 0,
        "max_revisions": max_revisions,
    }

    logger.info(f"Starting RAG pipeline for plant: {plant_name}")
    result = graph.invoke(initial_state)
    logger.info(f"RAG pipeline completed for plant: {plant_name}")

    return result["plant_care_card"]
