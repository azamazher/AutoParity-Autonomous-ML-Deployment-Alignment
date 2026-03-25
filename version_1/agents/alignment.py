"""
Alignment Agent — Keeps the best configuration found by ablation.

Applies the best parameter changes from experiments and maintains
the optimal configuration state.
"""

import yaml
import copy
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class AlignmentAgent:
    """
    Maintains and updates the optimal offline pipeline configuration.
    
    Applies improvements found by the Ablation Agent and saves
    the configuration history for reproducibility.
    """

    def __init__(self, config_path: str, results_dir: str = "results"):
        self.config_path = Path(config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.history: list = []

    def apply_best(
        self,
        current_config: Dict[str, Any],
        ablation_results: Dict[str, Any],
        min_improvement: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Apply the best parameter change from ablation results.

        Only applies if improvement exceeds min_improvement to avoid
        noise-driven changes.

        Args:
            current_config: Current offline config.
            ablation_results: Output from AblationAgent.test_hypotheses()
            min_improvement: Minimum parity loss improvement to accept.

        Returns:
            Updated config dict (or unchanged if no improvement).
        """
        best = ablation_results.get("best_overall")

        if not best:
            print("[AlignmentAgent] No ablation results to apply.")
            return current_config

        if best["improvement"] < min_improvement:
            print(f"[AlignmentAgent] Best improvement ({best['improvement']:.6f}) "
                  f"below threshold ({min_improvement}). Keeping current config.")
            return current_config

        # Apply the best change
        new_config = copy.deepcopy(current_config)
        param = best["best_param"]
        value = best["best_value"]

        if param and value is not None:
            old_value = current_config.get(param)
            new_config[param] = value

            change = {
                "timestamp": datetime.now().isoformat(),
                "parameter": param,
                "old_value": old_value,
                "new_value": value,
                "improvement": best["improvement"],
                "new_loss": best["best_loss"],
                "hypothesis": best["hypothesis"],
            }
            self.history.append(change)

            print(f"[AlignmentAgent] Applied: {param} = {old_value} → {value}")
            print(f"  Improvement: {best['improvement']:.6f}")
            print(f"  New loss: {best['best_loss']:.6f}")
        else:
            print("[AlignmentAgent] No actionable parameter change found.")

        return new_config

    def save_config(self, config: Dict[str, Any]) -> str:
        """Save the current best config to YAML."""
        # Load full config
        full_config = {}
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                full_config = yaml.safe_load(f) or {}

        # Update the offline section
        full_config.setdefault("offline", {}).update(config)

        with open(self.config_path, "w") as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

        return str(self.config_path)

    def save_history(self) -> str:
        """Save the change history for reproducibility."""
        path = self.results_dir / "alignment_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        return str(path)

    def get_summary(self) -> str:
        """Get a human-readable summary of all changes made."""
        if not self.history:
            return "No changes applied yet."

        lines = ["Alignment History:"]
        total_improvement = 0.0
        for i, change in enumerate(self.history, 1):
            lines.append(
                f"  {i}. {change['parameter']}: {change['old_value']} → {change['new_value']} "
                f"(Δ={change['improvement']:.6f})"
            )
            total_improvement += change["improvement"]

        lines.append(f"\nTotal improvement: {total_improvement:.6f}")
        return "\n".join(lines)
