---
name: model_refinement
version: v1.0
author: llm-survey
date: 2026-05-06
intended_model: any-instruct-tuned
change_rationale: |
  Model-refinement prompt used by `RAGModelExtractor.refine_model`.
  Mirrors `MODEL_REFINEMENT_PROMPT` in model_extraction_prompts.py.
eval_delta: null
---
You are an AI assistant helping to refine and validate scientific model specifications.

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
  - Consideration2: Another consideration
