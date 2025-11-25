"""
Hyperparameter Tuning Script with Neptune.ai Integration
==========================================================
This script extends the base hyperparameter tuner with Neptune.ai logging
for real-time tracking and visualization of tuning experiments.

Author: Generated for 01_RAG project
Date: November 25, 2025
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hyperparameter_tuner import HyperparameterTuner, ConfigModifier

try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    print("⚠️  Warning: neptune package not installed.")
    print("   Running without Neptune integration.")


class NeptuneHyperparameterTuner(HyperparameterTuner):
    """Hyperparameter tuner with Neptune.ai integration for experiment tracking."""
    
    def __init__(
        self,
        results_dir: Optional[str] = None,
        neptune_project: Optional[str] = None,
        neptune_api_token: Optional[str] = None,
        enable_neptune: bool = True
    ):
        """
        Initialize Neptune-integrated hyperparameter tuner.
        
        Args:
            results_dir: Directory for saving results (defaults to tests/logs/hyperparameter_tuning)
            neptune_project: Neptune.ai project name (or use NEPTUNE_PROJECT env var)
            neptune_api_token: Neptune.ai API token (or use NEPTUNE_API_TOKEN env var)
            enable_neptune: Whether to enable Neptune logging
        """
        # Import here to get the default RESULTS_DIR
        from hyperparameter_tuner import RESULTS_DIR as DEFAULT_RESULTS_DIR
        
        # Use provided results_dir or fall back to default
        if results_dir is None:
            results_dir = DEFAULT_RESULTS_DIR
        
        super().__init__(results_dir)
        
        self.enable_neptune = enable_neptune and NEPTUNE_AVAILABLE
        self.neptune_run = None
        
        if self.enable_neptune:
            # Get Neptune credentials
            self.neptune_project = neptune_project or os.environ.get('NEPTUNE_PROJECT')
            self.neptune_api_token = neptune_api_token or os.environ.get('NEPTUNE_API_TOKEN')
            
            if not self.neptune_project or not self.neptune_api_token:
                print("⚠️  Neptune credentials not found. Running without Neptune integration.")
                self.enable_neptune = False
        
        if not self.enable_neptune:
            print("ℹ️  Neptune integration disabled. Results will be saved locally only.")
    
    def start_neptune_run(self, tags: Optional[List[str]] = None, description: Optional[str] = None):
        """
        Start a Neptune run for tracking the tuning session.
        
        Args:
            tags: Optional list of tags for the run
            description: Optional description for the run
        """
        if not self.enable_neptune:
            return
        
        try:
            print("\n🚀 Starting Neptune run...")
            
            custom_run_id = f"hyperparam_tuning_{self.session_timestamp}"
            
            self.neptune_run = neptune.init_run(
                project=self.neptune_project,
                api_token=self.neptune_api_token,
                custom_run_id=custom_run_id,
                tags=tags or ["hyperparameter_tuning", "chunking"],
                name=f"Hyperparameter Tuning {self.session_timestamp}",
                description=description or "RAG hyperparameter tuning session"
            )
            
            print(f"✅ Neptune run created: {self.neptune_run['sys/id'].fetch()}")
            print(f"🔗 View run: {self.neptune_run.get_url()}")
            
            # Log session metadata
            self.neptune_run["session/timestamp"] = self.session_timestamp
            self.neptune_run["session/log_file"] = str(self.session_log)
            
        except Exception as e:
            print(f"⚠️  Failed to start Neptune run: {e}")
            print("   Continuing without Neptune integration.")
            self.enable_neptune = False
    
    def run_experiment(self, params: Dict[str, Any], experiment_name: str = None) -> Dict[str, Any]:
        """
        Run a single experiment with Neptune logging.
        
        Args:
            params: Dictionary of parameters to test
            experiment_name: Optional name for this experiment
            
        Returns:
            Dictionary containing experiment results
        """
        # Run the base experiment
        experiment = super().run_experiment(params, experiment_name)
        
        # Log to Neptune if enabled
        if self.enable_neptune and self.neptune_run:
            self._log_experiment_to_neptune(experiment)
        
        return experiment
    
    def _log_experiment_to_neptune(self, experiment: Dict[str, Any]):
        """Log a single experiment to Neptune."""
        try:
            exp_name = experiment["name"]
            
            # Log parameters
            for key, value in experiment["parameters"].items():
                self.neptune_run[f"experiments/{exp_name}/parameters/{key}"] = value
            
            # Log results
            self.neptune_run[f"experiments/{exp_name}/status"] = experiment["status"]
            self.neptune_run[f"experiments/{exp_name}/duration_seconds"] = experiment.get("duration_seconds", 0)
            self.neptune_run[f"experiments/{exp_name}/exit_code"] = experiment.get("exit_code", -1)
            
            # Log metrics
            metrics = experiment.get("metrics", {})
            for key, value in metrics.items():
                self.neptune_run[f"experiments/{exp_name}/metrics/{key}"] = value
            
            # Log time series for comparative charts
            exp_idx = len(self.results)
            
            # Also log experiment name in a separate namespace for reference
            self.neptune_run[f"experiment_names/{exp_idx}"] = exp_name
            
            # Log parameters as time series for tracking across experiments
            params = experiment["parameters"]
            if "CHUNK_STRATEGY" in params:
                # Convert strategy to numeric for charting (fixed_size=1, by_sentence=2)
                strategy_val = 1 if params["CHUNK_STRATEGY"] == "fixed_size" else 2
                self.neptune_run["charts/chunk_strategy"].append(strategy_val, step=exp_idx)
            
            if "FIXED_SIZE_CHUNK_SIZE" in params:
                self.neptune_run["charts/chunk_size"].append(
                    params["FIXED_SIZE_CHUNK_SIZE"],
                    step=exp_idx
                )
            
            if "FIXED_SIZE_OVERLAP" in params:
                self.neptune_run["charts/overlap"].append(
                    params["FIXED_SIZE_OVERLAP"],
                    step=exp_idx
                )
            
            if "BATCH_SIZE" in params:
                self.neptune_run["charts/batch_size"].append(
                    params["BATCH_SIZE"],
                    step=exp_idx
                )
            
            # Log metrics
            if metrics:
                if "total_time_seconds" in metrics:
                    self.neptune_run["charts/execution_time"].append(
                        metrics["total_time_seconds"],
                        step=exp_idx
                    )
                if "total_chunks" in metrics:
                    self.neptune_run["charts/total_chunks"].append(
                        metrics["total_chunks"],
                        step=exp_idx
                    )
                if "chunk_size_avg" in metrics:
                    self.neptune_run["charts/avg_chunk_size"].append(
                        metrics["chunk_size_avg"],
                        step=exp_idx
                    )
            
            # Log success/failure
            success_val = 1 if experiment["status"] == "success" else 0
            self.neptune_run["charts/success_rate"].append(
                success_val,
                step=exp_idx
            )
            
        except Exception as e:
            print(f"⚠️  Failed to log experiment to Neptune: {e}")
    
    def _print_summary(self):
        """Print summary and log final stats to Neptune."""
        super()._print_summary()
        
        if self.enable_neptune and self.neptune_run:
            try:
                # Log overall statistics
                successful = [r for r in self.results if r["status"] == "success"]
                failed = [r for r in self.results if r["status"] != "success"]
                
                self.neptune_run["summary/total_experiments"] = len(self.results)
                self.neptune_run["summary/successful"] = len(successful)
                self.neptune_run["summary/failed"] = len(failed)
                self.neptune_run["summary/success_rate"] = len(successful) / len(self.results) * 100 if self.results else 0
                
                # Find best experiment (fastest successful one)
                if successful:
                    best = min(successful, key=lambda x: x.get("duration_seconds", float('inf')))
                    self.neptune_run["summary/best_experiment"] = best["name"]
                    self.neptune_run["summary/best_duration"] = best.get("duration_seconds", 0)
                    
                    # Log best parameters
                    for key, value in best["parameters"].items():
                        self.neptune_run[f"summary/best_parameters/{key}"] = value
                
            except Exception as e:
                print(f"⚠️  Failed to log summary to Neptune: {e}")
    
    def cleanup(self):
        """Restore original config and stop Neptune run."""
        super().cleanup()
        
        if self.enable_neptune and self.neptune_run:
            try:
                print("\n📊 Stopping Neptune run...")
                self.neptune_run.stop()
                print("✅ Neptune run stopped")
            except Exception as e:
                print(f"⚠️  Failed to stop Neptune run: {e}")


# =============================================================================
# PREDEFINED EXPERIMENT CONFIGURATIONS WITH NEPTUNE
# =============================================================================

def quick_test_with_neptune(
    neptune_project: Optional[str] = None,
    neptune_api_token: Optional[str] = None
):
    """Quick test with Neptune tracking."""
    tuner = NeptuneHyperparameterTuner(
        neptune_project=neptune_project,
        neptune_api_token=neptune_api_token
    )
    tuner.config_modifier.backup_config()
    tuner.start_neptune_run(tags=["quick_test"])
    
    try:
        # Test 1: Baseline
        tuner.run_experiment({
            'CHUNK_STRATEGY': 'fixed_size',
            'FIXED_SIZE_CHUNK_SIZE': 1000,
            'FIXED_SIZE_OVERLAP': 150,
            'BATCH_SIZE': 50
        }, "baseline")
        
        # Test 2: Smaller chunks
        tuner.run_experiment({
            'CHUNK_STRATEGY': 'fixed_size',
            'FIXED_SIZE_CHUNK_SIZE': 500,
            'FIXED_SIZE_OVERLAP': 100,
            'BATCH_SIZE': 50
        }, "smaller_chunks")
        
        # Test 3: Larger chunks
        tuner.run_experiment({
            'CHUNK_STRATEGY': 'fixed_size',
            'FIXED_SIZE_CHUNK_SIZE': 1500,
            'FIXED_SIZE_OVERLAP': 200,
            'BATCH_SIZE': 50
        }, "larger_chunks")
        
    finally:
        tuner.cleanup()


def chunk_size_optimization_with_neptune(
    neptune_project: Optional[str] = None,
    neptune_api_token: Optional[str] = None
):
    """Test different chunk sizes and overlaps with Neptune tracking."""
    tuner = NeptuneHyperparameterTuner(
        neptune_project=neptune_project,
        neptune_api_token=neptune_api_token
    )
    tuner.config_modifier.backup_config()
    tuner.start_neptune_run(
        tags=["chunk_size_optimization", "grid_search"],
        description="Grid search over chunk sizes and overlaps"
    )
    
    try:
        param_grid = {
            'CHUNK_STRATEGY': ['fixed_size'],
            'FIXED_SIZE_CHUNK_SIZE': [500, 1000],
            'FIXED_SIZE_OVERLAP': [200, 250],
            'BATCH_SIZE': [50]
        }
        
        tuner.run_grid_search(param_grid)
        
    finally:
        tuner.cleanup()


def batch_size_optimization_with_neptune(
    neptune_project: Optional[str] = None,
    neptune_api_token: Optional[str] = None
):
    """Test different batch sizes with Neptune tracking."""
    tuner = NeptuneHyperparameterTuner(
        neptune_project=neptune_project,
        neptune_api_token=neptune_api_token
    )
    tuner.config_modifier.backup_config()
    tuner.start_neptune_run(
        tags=["batch_size_optimization", "grid_search"],
        description="Grid search over batch sizes for embedding generation"
    )
    
    try:
        param_grid = {
            'CHUNK_STRATEGY': ['fixed_size'],
            'FIXED_SIZE_CHUNK_SIZE': [1000],
            'FIXED_SIZE_OVERLAP': [250],
            'BATCH_SIZE': [10, 25, 50, 75, 100]
        }
        
        tuner.run_grid_search(param_grid)
        
    finally:
        tuner.cleanup()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with Neptune.ai integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with Neptune
  python hyperparameter_tuner_neptune.py --mode quick
  
  # Chunk size optimization with Neptune
  python hyperparameter_tuner_neptune.py --mode chunk_size
  
  # Batch size optimization with Neptune
  python hyperparameter_tuner_neptune.py --mode batch_size
  
  # Without Neptune integration
  python hyperparameter_tuner_neptune.py --mode quick --no-neptune

Environment Variables:
  NEPTUNE_API_TOKEN: Your Neptune.ai API token
  NEPTUNE_PROJECT: Your Neptune.ai project (e.g., username/project-name)
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick', 'chunk_size', 'batch_size'],
        default='quick',
        help='Tuning mode to run'
    )
    
    parser.add_argument(
        '--neptune-project',
        help='Neptune.ai project name (or use NEPTUNE_PROJECT env var)'
    )
    
    parser.add_argument(
        '--neptune-api-token',
        help='Neptune.ai API token (or use NEPTUNE_API_TOKEN env var)'
    )
    
    parser.add_argument(
        '--no-neptune',
        action='store_true',
        help='Disable Neptune integration'
    )
    
    args = parser.parse_args()
    
    # Check if Neptune should be disabled
    if args.no_neptune:
        os.environ.pop('NEPTUNE_API_TOKEN', None)
        os.environ.pop('NEPTUNE_PROJECT', None)
    
    print("="*70)
    print("RAG HYPERPARAMETER TUNING WITH NEPTUNE.AI")
    print("="*70)
    print(f"\nMode: {args.mode}")
    print(f"Neptune integration: {'Disabled' if args.no_neptune else 'Enabled'}")
    print()
    
    # Run selected mode
    if args.mode == 'quick':
        quick_test_with_neptune(args.neptune_project, args.neptune_api_token)
    elif args.mode == 'chunk_size':
        chunk_size_optimization_with_neptune(args.neptune_project, args.neptune_api_token)
    elif args.mode == 'batch_size':
        batch_size_optimization_with_neptune(args.neptune_project, args.neptune_api_token)
