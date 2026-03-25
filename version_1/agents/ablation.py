"""
Ablation Agent — Runs controlled experiments to test hypotheses.

For each hypothesis, systematically tests parameter changes and measures
their effect on parity loss.
"""

from typing import Dict, List, Any

from ..trace.schema import PipelineTrace
from ..alignment.experiment_runner import ExperimentRunner
from ..diff.parity_loss import ParityLoss


class AblationAgent:
    """
    Automated ablation testing: change one factor → measure parity loss.
    
    This is the automated version of what was done manually in Stage 3.
    """

    def __init__(self, experiment_runner: ExperimentRunner):
        self.runner = experiment_runner

    def test_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        current_config: Dict[str, Any],
        image_paths: List[str],
        online_traces: List[PipelineTrace],
    ) -> Dict[str, Any]:
        """
        Test a list of hypotheses via controlled ablation.

        For each hypothesis, ablates the suggested parameters and
        records which values reduce parity loss.

        Args:
            hypotheses: Output from HypothesisAgent.generate()
            current_config: Current offline pipeline config.
            image_paths: Test image paths.
            online_traces: Fixed online reference traces.

        Returns:
            Dict with ablation results per hypothesis, ranked by improvement.
        """
        results = []

        for hypothesis in hypotheses:
            h_name = hypothesis["hypothesis"]
            params = hypothesis["params_to_test"]

            print(f"\n[AblationAgent] Testing hypothesis: {h_name}")
            print(f"  Parameters: {params}")

            param_results = []
            for param in params:
                ablation = self.runner.run_ablation(
                    param_name=param,
                    base_config=current_config,
                    image_paths=image_paths,
                    online_traces=online_traces,
                )
                param_results.append(ablation)

            # Find the best result across all params for this hypothesis
            best_improvement = 0.0
            best_param = None
            best_value = None
            best_loss = float("inf")

            for pr in param_results:
                if pr["improvement"] > best_improvement:
                    best_improvement = pr["improvement"]
                    best_param = pr["parameter"]
                    best_value = pr["best_config"]
                    best_loss = pr["best_loss"]

            results.append({
                "hypothesis": h_name,
                "description": hypothesis["description"],
                "priority": hypothesis["priority"],
                "param_results": param_results,
                "best_param": best_param,
                "best_value": best_value,
                "best_loss": best_loss,
                "improvement": best_improvement,
            })

        # Sort by improvement (largest first)
        results.sort(key=lambda r: r["improvement"], reverse=True)

        return {
            "ablation_results": results,
            "best_overall": results[0] if results else None,
        }
