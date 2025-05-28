"""
CLI interface for the WeClone Evaluation Framework
"""

import asyncio
import click
from pathlib import Path

from weclone.utils.log import logger
from weclone.eval.framework import run_evaluation_from_config


@click.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to evaluation configuration file (YAML or JSON)'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='eval_runs',
    help='Output directory for evaluation results'
)
def run_eval(config: Path, output_dir: Path):
    """Run comprehensive model evaluation using configuration file"""
    logger.info(f"Starting evaluation with config: {config}")
    
    try:
        # Run the evaluation
        results = asyncio.run(run_evaluation_from_config(str(config)))
        
        logger.info(f"Evaluation completed successfully")
        logger.info(f"Processed {len(results)} test cases")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        models = set(r["model"] for r in results)
        prompts = set(r["prompt"] for r in results)
        
        print(f"Models tested: {', '.join(models)}")
        print(f"Prompts tested: {', '.join(prompts)}")
        print(f"Total test cases: {len(results)}")
        
        # Calculate average metrics
        if results:
            avg_metrics = {}
            for result in results:
                for metric_name, scores in result["metrics"].items():
                    if metric_name not in avg_metrics:
                        avg_metrics[metric_name] = {}
                    for score_name, score_value in scores.items():
                        if isinstance(score_value, (int, float)):
                            if score_name not in avg_metrics[metric_name]:
                                avg_metrics[metric_name][score_name] = []
                            avg_metrics[metric_name][score_name].append(score_value)
            
            print("\nAverage Metrics:")
            for metric_name, scores in avg_metrics.items():
                print(f"  {metric_name}:")
                for score_name, values in scores.items():
                    avg_value = sum(values) / len(values) if values else 0
                    print(f"    {score_name}: {avg_value:.3f}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise click.ClickException(f"Evaluation failed: {e}")


def main():
    """Main entry point for CLI"""
    run_eval()


if __name__ == '__main__':
    main() 