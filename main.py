#!/usr/bin/env python3
"""
LLM Model Specification Generator
Main entry point for the pipeline
"""
from __future__ import annotations

import os
import argparse
import json
import sys
from typing import Any, Dict
from dotenv import load_dotenv

# When running from a source checkout without `pip install -e .`, ensure the
# in-tree `src/` is importable. Once the package is installed, this is a no-op.
try:  # pragma: no cover - trivial import guard
    import llm_survey  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables
load_dotenv()

from llm_survey.rag_pipeline import RAGModelExtractor

# Import TopicAnalyzer only when needed
def get_topic_analyzer():
    """Import TopicAnalyzer only when needed to avoid dependency issues."""
    try:
        from llm_survey.topic_analysis import TopicAnalyzer
        return TopicAnalyzer
    except ImportError as e:
        print(f"Warning: Topic analysis not available due to dependency issue: {e}")
        return None

def run_complete_pipeline(
    input_file: str,
    openrouter_api_key: str = "",
    use_rag: bool = True,
    perform_topic_analysis: bool = True,
    output_dir: str = "outputs",
    llm_model: str = "google/gemma-4-31b-it",
    base_url: str = "https://openrouter.ai/api/v1",
    extra_headers: Dict[str, str] | None = None,
    enable_literature_retrieval: bool = True,
    enable_refinement_loop: bool = True,
    max_refinement_iterations: int = 2,
    completeness_threshold: float = 0.75,
) -> Dict[str, Any]:
    """
    Run the complete pipeline from data processing to model extraction.
    
    Args:
        input_file: Path to input data file
        openrouter_api_key: OpenRouter API key
        use_rag: Whether to use RAG enhancement
        perform_topic_analysis: Whether to perform topic analysis
        output_dir: Output directory for results
    
    Returns:
        Dictionary containing all results
    """
    
    print("Starting LLM Model Specification Pipeline")
    print("=" * 50)
    
    # Initialize extractor. Settings provides typed defaults for temperature
    # and headers; explicit args still win.
    from llm_survey.config import get_settings
    cfg = get_settings()
    extractor = RAGModelExtractor(
        openai_api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY", ""),
        llm_model=llm_model,
        base_url=base_url,
        extra_headers=extra_headers or cfg.http_extra_headers(),
        temperature=cfg.llm_temperature,
        enable_literature_retrieval=enable_literature_retrieval,
    )
    
    # Step 1: Process and store data
    print("\nStep 1: Processing and storing data...")
    processed_chunks = extractor.process_and_store_data(
        input_file,
        max_tokens=500,
        save_processed=True
    )
    print(f"Processed {len(processed_chunks)} text chunks")
    
    # Step 2: Extract models
    print("\nStep 2: Extracting models...")
    extraction_results = extractor.extract_models_from_all_chunks(
        use_rag=use_rag,
        save_results=True
    )
    
    successful_extractions = [r for r in extraction_results if r['success']]
    print(f"Extracted models from {len(successful_extractions)}/{len(extraction_results)} chunks")

    # Step 3: Cross-chunk gap detection
    print("\nStep 3: Detecting cross-chunk gaps...")
    gap_report = extractor.detect_cross_chunk_gaps(
        extraction_results=extraction_results,
        save_results=True,
    )
    print(
        "Gap detection complete: "
        f"{len(gap_report.get('gaps', []))} gaps, "
        f"coverage={gap_report.get('structural_coverage_score', gap_report.get('overall_model_completeness', 0)):.2f}, "
        f"testability={gap_report.get('model_testability_score', 0):.2f}"
    )

    # Step 4: Clarification planning
    print("\nStep 4: Building clarification plan...")
    clarification_plan = extractor.generate_clarification_plan(
        gap_report=gap_report,
        save_results=True,
    )
    print(
        "Clarification planning complete: "
        f"{len(clarification_plan.get('questions', []))} questions, "
        f"auto_answers={len(clarification_plan.get('auto_answers', []))}"
    )

    # Step 5: Refinement loop (optional)
    refinement_loop = None
    if enable_refinement_loop:
        print("\nStep 5: Running refinement loop...")
        refinement_loop = extractor.run_refinement_loop(
            extraction_results=extraction_results,
            gap_report=gap_report,
            clarification_plan=clarification_plan,
            use_rag=use_rag,
            max_iterations=max_refinement_iterations,
            completeness_threshold=completeness_threshold,
            save_results=True,
        )
        extraction_results = refinement_loop["final_extraction_results"]
        gap_report = refinement_loop["final_gap_report"]
        clarification_plan = refinement_loop["final_clarification_plan"]
        successful_extractions = [r for r in extraction_results if r['success']]
        loop_report = refinement_loop["report"]
        print(
            "Refinement loop complete: "
            f"iterations={loop_report.get('iterations_completed', 0)}, "
            f"stop_reason={loop_report.get('stop_reason', 'unknown')}, "
            f"final_completeness={loop_report.get('final_completeness', 0):.2f}"
        )
    else:
        print("\nStep 5: Refinement loop skipped.")

    # Step 6: Topic analysis (optional)
    topic_results = None
    if perform_topic_analysis:
        print("\nStep 6: Performing topic analysis...")
        TopicAnalyzer = get_topic_analyzer()
        if TopicAnalyzer:
            topic_analyzer = TopicAnalyzer(
                embedding_model=extractor.embedding_model_name,
                nr_topics=10,
                min_topic_size=5
            )
            
            texts = [chunk['text'] for chunk in processed_chunks]
            topic_results = topic_analyzer.analyze_topics(texts, save_results=True)
            
            # Generate visualizations
            plot_paths = topic_analyzer.create_topic_visualizations(topic_results)
            print(f"Created {len(plot_paths)} visualizations")
            
            # Generate summary
            summary = topic_analyzer.generate_topic_summary(topic_results)
            print("Generated topic summary")
        else:
            print("Topic analysis skipped due to dependency issues")

    # Step 7: Consolidation, conflicts, validation, exports
    print("\nStep 7: Consolidating model and building final outputs...")
    finalization = extractor.finalize_model_outputs(
        extraction_results=extraction_results,
        gap_report=gap_report,
        clarification_plan=clarification_plan,
        refinement_report=refinement_loop["report"] if refinement_loop else None,
        output_dir=output_dir,
        save_results=True,
    )
    consolidated_model = finalization["consolidated_model"]
    conflict_report = finalization["conflict_report"]
    literature_validation = finalization["literature_validation"]
    final_exports = finalization["final_exports"]
    print(
        "Finalization complete: "
        f"{len(consolidated_model.get('variables', []))} variables, "
        f"{len(consolidated_model.get('relationships', []))} relationships, "
        f"{conflict_report.get('unresolved_count', 0)} unresolved contradictions"
    )

    # Step 8: Generate comprehensive report
    print("\nStep 8: Generating comprehensive report...")
    
    report = {
        'pipeline_info': {
            'input_file': input_file,
            'use_rag': use_rag,
            'perform_topic_analysis': perform_topic_analysis,
            'enable_literature_retrieval': enable_literature_retrieval,
            'enable_refinement_loop': enable_refinement_loop,
            'max_refinement_iterations': max_refinement_iterations,
            'completeness_threshold': completeness_threshold,
            'total_chunks': len(processed_chunks)
        },
        'extraction_results': {
            'total_extractions': len(extraction_results),
            'successful_extractions': len(successful_extractions),
            'success_rate': len(successful_extractions) / len(extraction_results) if extraction_results else 0,
            'models': successful_extractions
        },
        'gap_detection': gap_report,
        'clarification_plan': clarification_plan,
        'refinement_loop': refinement_loop["report"] if refinement_loop else None,
        'topic_analysis': topic_results if topic_results else None,
        'consolidated_model': consolidated_model,
        'conflict_report': conflict_report,
        'literature_validation': literature_validation,
        'final_exports': final_exports.get("paths", {}),
    }
    
    # Save comprehensive report
    report_path = f"{output_dir}/comprehensive_report.json"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Comprehensive report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Total chunks processed: {len(processed_chunks)}")
    print(f"Models extracted: {len(successful_extractions)}/{len(extraction_results)}")
    print(f"Success rate: {report['extraction_results']['success_rate']*100:.1f}%")
    cov = gap_report.get("structural_coverage_score", gap_report.get("overall_model_completeness", 0))
    print(f"Structural coverage (heuristic): {float(cov)*100:.1f}%")
    print(f"Model testability: {gap_report.get('model_testability_score', 0)*100:.1f}%")
    print(f"Clarification questions: {len(clarification_plan.get('questions', []))}")
    print(f"Consolidated variables: {len(consolidated_model.get('variables', []))}")
    print(f"Consolidated relationships: {len(consolidated_model.get('relationships', []))}")
    print(f"Unresolved contradictions: {conflict_report.get('unresolved_count', 0)}")
    print(f"Novel hypotheses: {literature_validation.get('novelty_count', 0)}")
    if refinement_loop:
        print(
            f"Refinement iterations: {refinement_loop['report'].get('iterations_completed', 0)} "
            f"(stop: {refinement_loop['report'].get('stop_reason', 'unknown')})"
        )
    
    if topic_results:
        topic_info = topic_results['model_info']
        print(f"Topics identified: {len([t for t in topic_results['topic_info'] if t['Topic'] != -1])}")
    
    print(f"Results saved to: {output_dir}")
    print("=" * 50)
    
    return report

def run_interactive_mode():
    """Run the pipeline in interactive mode."""
    print("Interactive Mode")
    print("=" * 30)
    
    # Get input file
    input_file = input("Enter path to input file (CSV/TXT/PDF/DOCX): ").strip()
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    # Get API key
    api_key = input("Enter OpenRouter API key (or press Enter to use environment variable): ").strip()
    if not api_key:
        from llm_survey.config import get_settings
        api_key = get_settings().openrouter_api_key
        if not api_key:
            print("No API key provided")
            return
    
    # Get options
    use_rag = input("Use RAG enhancement? (y/n, default: y): ").strip().lower() != 'n'
    use_literature = input("Enable literature retrieval? (y/n, default: y): ").strip().lower() != 'n'
    use_refinement = input("Enable refinement loop? (y/n, default: y): ").strip().lower() != 'n'
    max_iterations_raw = input("Max refinement iterations (default: 2): ").strip()
    threshold_raw = input("Completeness threshold 0-1 (default: 0.75): ").strip()
    perform_topic_analysis = input("Perform topic analysis? (y/n, default: y): ").strip().lower() != 'n'
    try:
        max_iterations = int(max_iterations_raw) if max_iterations_raw else 2
    except ValueError:
        max_iterations = 2
    try:
        threshold = float(threshold_raw) if threshold_raw else 0.75
    except ValueError:
        threshold = 0.75
    
    # Run pipeline
    try:
        results = run_complete_pipeline(
            input_file=input_file,
            openrouter_api_key=api_key,
            use_rag=use_rag,
            perform_topic_analysis=perform_topic_analysis,
            enable_literature_retrieval=use_literature,
            enable_refinement_loop=use_refinement,
            max_refinement_iterations=max_iterations,
            completeness_threshold=threshold,
        )
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")

def create_sample_data() -> None:
    """Print the path to the bundled synthetic survey (preferred sample for demos)."""
    from llm_survey.utils.preprocess import create_sample_data as bundled_sample_path

    path = bundled_sample_path()
    print(f"Bundled sample survey: {path}")

def main():
    parser = argparse.ArgumentParser(description="LLM Model Specification Generator")
    parser.add_argument("--input", "-i", help="Input file path")
    parser.add_argument("--api-key", "-k", help="OpenRouter API key")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG enhancement")
    parser.add_argument("--no-topic-analysis", action="store_true", help="Disable topic analysis")
    parser.add_argument("--interactive", "-I", action="store_true", help="Run in interactive mode")
    parser.add_argument("--create-sample", action="store_true", help="Create sample data")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    parser.add_argument("--llm-model", default="google/gemma-4-31b-it", help="OpenRouter model name")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="OpenRouter-compatible base URL")
    parser.add_argument("--no-literature", action="store_true", help="Disable literature retrieval/enrichment")
    parser.add_argument("--no-refinement", action="store_true", help="Disable iterative refinement loop")
    parser.add_argument("--max-refinement-iterations", type=int, default=2, help="Maximum refinement iterations")
    parser.add_argument(
        "--completeness-threshold",
        type=float,
        default=0.75,
        help="Stop refinement when structural coverage heuristic reaches this threshold",
    )
    parser.add_argument("--http-referer", default="", help="Optional HTTP Referer header for OpenRouter")
    parser.add_argument("--x-title", default="", help="Optional X-Title header for OpenRouter")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
        return
    
    if args.interactive:
        run_interactive_mode()
        return
    
    if not args.input:
        print("Please provide an input file or use --interactive mode")
        parser.print_help()
        return
    
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    try:
        extra_headers: Dict[str, str] = {}
        if args.http_referer:
            extra_headers["HTTP-Referer"] = args.http_referer
        if args.x_title:
            extra_headers["X-Title"] = args.x_title

        results = run_complete_pipeline(
            input_file=args.input,
            openrouter_api_key=args.api_key,
            use_rag=not args.no_rag,
            perform_topic_analysis=not args.no_topic_analysis,
            output_dir=args.output_dir,
            llm_model=args.llm_model,
            base_url=args.base_url,
            extra_headers=extra_headers,
            enable_literature_retrieval=not args.no_literature,
            enable_refinement_loop=not args.no_refinement,
            max_refinement_iterations=args.max_refinement_iterations,
            completeness_threshold=args.completeness_threshold,
        )
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")

if __name__ == "__main__":
    main() 
