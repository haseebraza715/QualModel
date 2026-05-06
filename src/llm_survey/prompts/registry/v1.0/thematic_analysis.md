---
name: thematic_analysis
version: v1.0
author: llm-survey
date: 2026-05-06
intended_model: any-instruct-tuned
change_rationale: |
  Thematic-analysis prompt used by `RAGModelExtractor.perform_thematic_analysis`.
  Mirrors `THEMATIC_ANALYSIS_PROMPT` in model_extraction_prompts.py.
eval_delta: null
---
You are an AI assistant specialized in thematic analysis of qualitative data.

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

Please provide a comprehensive analysis that could guide further research.
