"""
IBM Granite Integration – Blueprint Generator
Orchestrates RAG retrieval + Granite LLM generation.
"""

import logging
import re
from typing import Dict, Any, List

from granite.granite_client import GraniteClient
from granite.prompt_builder import build_blueprint_prompt

logger = logging.getLogger(__name__)


def _parse_section(text: str, section_number: int, section_name: str) -> str:
    """Extract a specific numbered section from the generated blueprint text."""
    # Match patterns like "### 7. MARKET RESEARCH" or "## 7. MARKET RESEARCH"
    pattern = rf"###?\s*{section_number}\.\s*{re.escape(section_name)}(.*?)(?=###?\s*\d+\.|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_readiness_score(text: str) -> Dict[str, Any]:
    """Extract readiness scores from section 18 of the blueprint."""
    scores: Dict[str, Any] = {}

    dimensions = [
        ("idea_clarity", r"Idea Clarity.*?(\d+)\s*/\s*20"),
        ("market_opportunity", r"Market Opportunity.*?(\d+)\s*/\s*20"),
        ("technical_feasibility", r"Technical Feasibility.*?(\d+)\s*/\s*20"),
        ("business_model_strength", r"Business Model Strength.*?(\d+)\s*/\s*20"),
        ("team_resource_readiness", r"Team.*?Resource Readiness.*?(\d+)\s*/\s*10"),
        ("regulatory_readiness", r"Regulatory Readiness.*?(\d+)\s*/\s*10"),
    ]

    for key, pattern in dimensions:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                scores[key] = int(match.group(1))
            except ValueError:
                scores[key] = 0

    # Extract overall score
    overall_match = re.search(r"OVERALL SCORE.*?(\d+)\s*/\s*100", text, re.IGNORECASE | re.DOTALL)
    scores["overall"] = int(overall_match.group(1)) if overall_match else sum(scores.values())

    return scores


class BlueprintGenerator:
    """
    Orchestrates RAG + IBM Granite to produce a startup blueprint.
    """

    def __init__(self, granite_client: GraniteClient):
        self.granite_client = granite_client

    def generate(
        self,
        startup_input: Dict[str, Any],
        rag_retriever,
    ) -> Dict[str, Any]:
        """
        Generate a complete startup blueprint.

        Args:
            startup_input: User-provided startup details.
            rag_retriever: Initialised RAGRetriever instance.

        Returns:
            Dict containing:
                - 'blueprint_raw': Full raw text from Granite.
                - 'sections': Parsed individual sections.
                - 'retrieved_context': List of RAG-retrieved docs.
                - 'readiness_score': Parsed score breakdown.
                - 'label_counts': Count of [RETRIEVED FACT], [ESTIMATE], etc.
        """
        # --- Step 1: RAG Retrieval ---
        logger.info("Retrieving context from knowledge base...")
        retrieved_docs = rag_retriever.retrieve_for_startup(startup_input)
        rag_context = rag_retriever.format_context(retrieved_docs)
        logger.info("Retrieved %d context chunks.", len(retrieved_docs))

        # --- Step 2: Build Prompt ---
        prompt = build_blueprint_prompt(startup_input, rag_context)

        # --- Step 3: Call IBM Granite ---
        logger.info("Calling IBM Granite for blueprint generation...")
        blueprint_raw = self.granite_client.generate(prompt)
        logger.info("Blueprint generated (%d characters).", len(blueprint_raw))

        # --- Step 4: Parse label counts ---
        label_counts = {
            "retrieved_facts": blueprint_raw.count("[RETRIEVED FACT]"),
            "estimates": blueprint_raw.count("[ESTIMATE]"),
            "assumptions": blueprint_raw.count("[ASSUMPTION]"),
            "recommendations": blueprint_raw.count("[RECOMMENDATION]"),
        }

        # --- Step 5: Parse readiness score ---
        readiness_section = _parse_section(blueprint_raw, 18, "STARTUP READINESS SCORE")
        readiness_score = parse_readiness_score(readiness_section or blueprint_raw)

        # --- Step 6: Build sections dict ---
        section_definitions = [
            (1, "STARTUP IDEA ANALYSIS"),
            (2, "PROBLEM STATEMENT"),
            (3, "PROPOSED SOLUTION"),
            (4, "TARGET CUSTOMERS"),
            (5, "UNIQUE VALUE PROPOSITION"),
            (6, "BUSINESS MODEL CANVAS"),
            (7, "MARKET RESEARCH"),
            (8, "COMPETITOR ANALYSIS"),
            (9, "REVENUE MODEL"),
            (10, "ESTIMATED STARTUP BUDGET"),
            (11, "GO-TO-MARKET STRATEGY"),
            (12, "FUNDING OPPORTUNITIES"),
            (13, "GOVERNMENT SCHEMES"),
            (14, "LEGAL AND COMPLIANCE CHECKLIST"),
            (15, "INCUBATORS AND ACCELERATORS"),
            (16, "POTENTIAL INVESTOR CATEGORIES"),
            (17, "30/60/90 DAY ACTION PLAN"),
            (18, "STARTUP READINESS SCORE"),
        ]

        sections = {}
        for num, name in section_definitions:
            parsed = _parse_section(blueprint_raw, num, name)
            key = name.lower().replace(" ", "_").replace("/", "_")
            sections[key] = parsed if parsed else f"(Section not parsed – see full blueprint)"

        return {
            "blueprint_raw": blueprint_raw,
            "sections": sections,
            "retrieved_context": retrieved_docs,
            "readiness_score": readiness_score,
            "label_counts": label_counts,
        }
