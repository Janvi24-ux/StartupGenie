"""
IBM Granite Integration – Prompt Builder
Constructs structured prompts for the IBM Granite LLM.
"""

from typing import Dict, Any


# Labelling and formatting rules injected into the user turn of the prompt.
# The system role persona is set separately by GraniteClient.wrap_prompt().
LABEL_RULES = """LABELLING RULES – apply to every sentence in your response:
- [RETRIEVED FACT] → Information sourced directly from the knowledge base context below.
- [ESTIMATE] → Any financial figure, market size, or quantitative projection.
- [ASSUMPTION] → Logical inference made in the absence of explicit data.
- [RECOMMENDATION] → Strategic advice or actionable suggestion.

FORMATTING RULES:
- Use ### headings for each numbered section.
- Use bullet points and numbered lists where appropriate.
- Every financial number MUST carry an [ESTIMATE] label.
- Be specific and actionable. Do not use vague filler sentences.
- Label all government scheme details as [RETRIEVED FACT].
"""


def build_blueprint_prompt(
    startup_input: Dict[str, Any],
    rag_context: str,
) -> str:
    """
    Build the full prompt to send to IBM Granite for startup blueprint generation.

    Args:
        startup_input: Dict with user-provided fields.
        rag_context: Formatted context retrieved from the RAG knowledge base.

    Returns:
        Complete prompt string.
    """
    idea = startup_input.get("idea", "")
    industry = startup_input.get("industry", "")
    target_customer = startup_input.get("target_customer", "")
    location = startup_input.get("location", "India")
    budget = startup_input.get("budget", "")
    stage = startup_input.get("stage", "Idea Stage")

    prompt = f"""{LABEL_RULES}

## KNOWLEDGE BASE CONTEXT (Retrieved via RAG)
The following information has been retrieved from authoritative startup knowledge documents.
Use [RETRIEVED FACT] when referencing this information.

{rag_context}

---

## USER'S STARTUP INPUT
- **Startup Idea:** {idea}
- **Industry:** {industry}
- **Target Customer:** {target_customer}
- **Location/Country:** {location}
- **Available Budget:** {budget}
- **Startup Stage:** {stage}

---

## YOUR TASK
Generate a comprehensive startup blueprint with ALL of the following sections.
Label every claim appropriately as [RETRIEVED FACT], [ESTIMATE], [ASSUMPTION], or [RECOMMENDATION].

### 1. STARTUP IDEA ANALYSIS
Analyse the core startup idea. Identify the core problem it solves, the innovation, and the feasibility.

### 2. PROBLEM STATEMENT
Define the specific problem being solved. Include the scale of the problem and who is affected.

### 3. PROPOSED SOLUTION
Describe the product/service in detail. Explain how it solves the problem uniquely.

### 4. TARGET CUSTOMERS
Define primary and secondary customer segments. Include demographics, psychographics, and behavior patterns.

### 5. UNIQUE VALUE PROPOSITION
Craft a crisp UVP statement. Explain what makes this startup different from existing alternatives.

### 6. BUSINESS MODEL CANVAS
Provide all 9 building blocks:
- Customer Segments, Value Propositions, Channels, Customer Relationships, Revenue Streams
- Key Resources, Key Activities, Key Partnerships, Cost Structure

### 7. MARKET RESEARCH
Provide TAM, SAM, SOM analysis with [ESTIMATE] figures. Include market trends and growth rates for {industry} in {location}.

### 8. COMPETITOR ANALYSIS
List 3–5 key competitors. For each: product/service, strengths, weaknesses, market position.
Identify the competitive gap this startup can exploit.

### 9. REVENUE MODEL
Detail the primary and secondary revenue streams. Include pricing strategy, [ESTIMATE] unit economics (CAC, LTV, payback period).

### 10. ESTIMATED STARTUP BUDGET
Provide a detailed budget breakdown across categories (Technology, Team, Marketing, Legal, Operations).
ALL figures must be labelled [ESTIMATE]. Cover initial 6–12 months.

### 11. GO-TO-MARKET STRATEGY
Define Phase 1 (0–3 months), Phase 2 (3–6 months), Phase 3 (6–12 months) GTM plan.
Include specific channels, partnerships, and launch tactics.

### 12. FUNDING OPPORTUNITIES
List relevant funding sources for this startup (grants, angels, VCs, government funds).
Reference specific schemes from the knowledge base as [RETRIEVED FACT].

### 13. GOVERNMENT SCHEMES
List applicable government schemes in {location} with eligibility criteria and benefits.
Label all scheme details as [RETRIEVED FACT].

### 14. LEGAL AND COMPLIANCE CHECKLIST
Provide a step-by-step compliance checklist specific to {industry} in {location}.
Include registrations, licences, IP protection, and data privacy requirements.
Label regulatory requirements as [RETRIEVED FACT].

### 15. INCUBATORS AND ACCELERATORS
Recommend 5 incubators/accelerators best suited to this startup. For each: name, focus, benefits, and how to apply.
Label known program details as [RETRIEVED FACT].

### 16. POTENTIAL INVESTOR CATEGORIES
List investor types and specific funds/angels that invest in {industry} at the {stage} stage.
Include typical ticket sizes [ESTIMATE] and what they look for.

### 17. 30/60/90 DAY ACTION PLAN
Provide a concrete action plan:
- **30 Days:** Immediate priorities (validation, team, legal setup).
- **60 Days:** Product/service development milestones.
- **90 Days:** Launch preparation and first customer acquisition.

### 18. STARTUP READINESS SCORE
Score the startup out of 100 across 6 dimensions:
1. Idea Clarity (0–20)
2. Market Opportunity (0–20)
3. Technical Feasibility (0–20)
4. Business Model Strength (0–20)
5. Team/Resource Readiness (0–10)
6. Regulatory Readiness (0–10)

Provide the score and a brief justification for each dimension.
End with an OVERALL SCORE and a one-paragraph assessment.

---
Generate the complete blueprint now. Be specific, actionable, and realistic.
"""
    return prompt
