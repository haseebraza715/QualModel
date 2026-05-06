"""Prompt templates for the extraction pipeline.

Historical note: these constants are now thin wrappers around the versioned
prompt registry in `llm_survey/prompts/registry/`. The registry is the single
source of truth — editing the .md files there will update these constants on
next import. The fallback string literals below are kept so the package still
works if the registry is missing or fails to load (e.g. partial install).
"""
from typing import Any, Dict

from llm_survey.utils.prompt_safety import build_structured_extraction_user_message


def _load_or_default(name: str, default: str) -> str:
    """Load a prompt from the registry; fall back to the bundled literal."""
    try:
        from llm_survey.prompts.registry import default_registry
        return default_registry().text(name)
    except Exception:
        return default


# Base prompt for extracting variables and relationships
_BASE_EXTRACTION_FALLBACK = """You are an AI scientific assistant specialized in extracting variables and relationships from qualitative survey data.

Given the following qualitative survey excerpt, extract:
1. Key variables (with clear, concise descriptions)
2. Causal or conditional relationships between variables
3. Hypotheses (if evident in the text)
4. Any moderating or mediating factors mentioned

Format your response as YAML with the following structure (do not use markdown code blocks or backticks):

Variables:
  - VariableName: Brief description of what this variable represents
  - AnotherVariable: Description...

Relationships:
  - If [condition], then [outcome]
  - [Variable1] affects [Variable2] through [mechanism]
  - When [context], [relationship] occurs

Hypotheses:
  - Hypothesis1: Description of the proposed relationship
  - Hypothesis2: Another proposed relationship

Moderators:
  - ModeratorVariable: How it affects the relationship

Context:
{context}

Excerpt:
{input_text}

Please be precise and only extract what is explicitly mentioned or strongly implied in the text.

IMPORTANT QUALITY REQUIREMENTS:
1. Variables should be specific and measurable concepts
2. Relationships must specify direction (positive/negative) and mechanism
3. Hypotheses should be testable and specific
4. Avoid generic or overly broad statements
5. Each relationship should clearly state how one variable affects another

Return only valid YAML without any markdown formatting."""

# Enhanced prompt with RAG context
_RAG_ENHANCED_FALLBACK = """You are an AI scientific assistant specialized in extracting variables and relationships from qualitative survey data.

You have access to relevant context from similar survey responses and research documents. Use this context to enhance your understanding and provide more comprehensive model specifications.

Given the following qualitative survey excerpt and relevant context, extract:
1. Key variables (with clear, concise descriptions)
2. Causal or conditional relationships between variables
3. Hypotheses (if evident in the text)
4. Any moderating or mediating factors mentioned
5. Connections to broader themes or patterns from the context

Format your response as YAML with the following structure (do not use markdown code blocks or backticks):

Variables:
  - VariableName: Brief description of what this variable represents
  - AnotherVariable: Description...

Relationships:
  - If [condition], then [outcome]
  - [Variable1] affects [Variable2] through [mechanism]
  - When [context], [relationship] occurs

Hypotheses:
  - Hypothesis1: Description of the proposed relationship
  - Hypothesis2: Another proposed relationship

Moderators:
  - ModeratorVariable: How it affects the relationship

Themes:
  - Theme1: Connection to broader patterns
  - Theme2: Another thematic connection

Relevant Context:
{context}

Current Excerpt:
{input_text}

Please be precise and only extract what is explicitly mentioned or strongly implied in the text, but use the context to identify broader patterns and connections.

IMPORTANT QUALITY REQUIREMENTS:
1. Variables should be specific and measurable concepts
2. Relationships must specify direction (positive/negative) and mechanism
3. Hypotheses should be testable and specific
4. Avoid generic or overly broad statements
5. Each relationship should clearly state how one variable affects another
6. Use context to identify deeper patterns but stay grounded in the text

Return only valid YAML without any markdown formatting."""

# Prompt for thematic analysis
_THEMATIC_ANALYSIS_FALLBACK = """You are an AI assistant specialized in thematic analysis of qualitative data.

Analyze the following text excerpts and identify:
1. Recurring themes or patterns
2. Key concepts that appear across multiple responses
3. Potential research questions or hypotheses
4. Variables that could be operationalized for quantitative research

Format your response as YAML:

Themes:
  - ThemeName: Description of the theme and its significance
  - AnotherTheme: Description...

KeyConcepts:
  - Concept1: Definition and examples
  - Concept2: Definition and examples

ResearchQuestions:
  - Question1: Specific research question that could be investigated
  - Question2: Another research question

OperationalizableVariables:
  - Variable1: How this could be measured quantitatively
  - Variable2: How this could be measured quantitatively

Text Excerpts:
{text_excerpts}

Please provide a comprehensive analysis that could guide further research."""

# Prompt for model refinement
_MODEL_REFINEMENT_FALLBACK = """You are an AI assistant helping to refine and validate scientific model specifications.

Review the following model specification and suggest improvements:

1. Check for logical consistency
2. Identify missing variables or relationships
3. Suggest clearer variable definitions
4. Identify potential confounding factors
5. Recommend additional hypotheses to test

Original Model:
{original_model}

Context from which it was derived:
{context}

Please provide:
1. A refined version of the model
2. A list of suggested improvements
3. Additional variables or relationships to consider
4. Potential research questions to investigate

Format your response as YAML:

RefinedModel:
  Variables:
    - VariableName: Improved description
  Relationships:
    - Refined relationship statement
  Hypotheses:
    - Refined hypothesis

Suggestions:
  - Suggestion1: Description of improvement
  - Suggestion2: Another improvement

AdditionalConsiderations:
  - Consideration1: Additional factor to consider
  - Consideration2: Another consideration"""


_EXTRACTION_SYSTEM_FALLBACK = """You are a senior qualitative research extraction assistant.

Return only schema-valid JSON for the requested response model.
Keep outputs grounded in the provided chunk text.
Use survey and literature context only to improve precision, never to invent unsupported claims.
Always populate the gaps field with concrete missing-information items when appropriate.

For every variable, relationship, hypothesis, and moderator:
- source_chunk_ids: list the chunk id you are extracting from (provided in user message metadata if present).
- evidence_strength: "direct" if the participant literally stated the construct; "inferred" if it is a reasonable
  reading but not explicit; "weak" if evidence is a single ambiguous phrase or very thin.
"""


# --------------------------------------------------------------------------
# Public constants — resolved from the prompt registry at import time so the
# .md files under `prompts/registry/v1.0/` are the single source of truth.
# The `_FALLBACK` literals above are only used when the registry is missing
# (partial install, broken file, etc.).
# --------------------------------------------------------------------------
BASE_EXTRACTION_PROMPT = _load_or_default("base_extraction", _BASE_EXTRACTION_FALLBACK)
RAG_ENHANCED_PROMPT = _load_or_default("rag_enhanced", _RAG_ENHANCED_FALLBACK)
THEMATIC_ANALYSIS_PROMPT = _load_or_default("thematic_analysis", _THEMATIC_ANALYSIS_FALLBACK)
MODEL_REFINEMENT_PROMPT = _load_or_default("model_refinement", _MODEL_REFINEMENT_FALLBACK)
EXTRACTION_SYSTEM_PROMPT = _load_or_default("extraction_system", _EXTRACTION_SYSTEM_FALLBACK)


def get_prompt_template(prompt_type: str = "base") -> str:
    """Get the appropriate prompt template based on type."""
    templates = {
        "base": BASE_EXTRACTION_PROMPT,
        "rag": RAG_ENHANCED_PROMPT,
        "thematic": THEMATIC_ANALYSIS_PROMPT,
        "refinement": MODEL_REFINEMENT_PROMPT,
    }
    return templates.get(prompt_type, BASE_EXTRACTION_PROMPT)


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with the given parameters."""
    return template.format(**kwargs)


def format_structured_extraction_prompt(
    chunk_text: str,
    survey_context: str,
    literature_context: str,
) -> str:
    """Build user prompt for typed extraction (user text is never passed through str.format)."""
    return build_structured_extraction_user_message(
        chunk_text or "",
        survey_context or "",
        literature_context or "",
    )
