"""
Hyperparameter Tuning Script for RAG System
============================================
This script automatically tests different hyperparameter combinations by:
1. Modifying rag_config.py with test parameters
2. Running chunk_qwen3_0_6B.py to create embeddings database
3. Logging results for comparison

Author: Generated for 01_RAG project
Date: November 25, 2025
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import re

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_FILE = os.path.join(SCRIPT_DIR, "rag_config.py")
CHUNKING_SCRIPT = os.path.join(SCRIPT_DIR, "chunk_qwen3_0_6B.py")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "tests", "logs", "hyperparameter_tuning")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

class ConfigModifier:
    """Utility to modify configuration parameters in rag_config.py"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.original_content = None
        
    def backup_config(self):
        """Backup original config file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.original_content = f.read()
    
    def restore_config(self):
        """Restore original config file"""
        if self.original_content:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(self.original_content)
    
    def modify_parameter(self, param_name: str, new_value: Any):
        """
        Modify a single parameter in the config file.
        
        Args:
            param_name: Name of the parameter (e.g., 'FIXED_SIZE_CHUNK_SIZE')
            new_value: New value for the parameter
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Format value based on type
        if isinstance(new_value, str):
            value_str = f'"{new_value}"'
        elif isinstance(new_value, bool):
            value_str = str(new_value)
        else:
            value_str = str(new_value)
        
        # Replace the parameter value
        # Match patterns like: PARAM_NAME = value
        pattern = rf'^{param_name}\s*=\s*.*?$'
        replacement = f'{param_name} = {value_str}'
        
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def modify_parameters(self, params: Dict[str, Any]):
        """
        Modify multiple parameters at once.
        
        Args:
            params: Dictionary mapping parameter names to their new values
        """
        for param_name, param_value in params.items():
            self.modify_parameter(param_name, param_value)


class HyperparameterTuner:
    """Main class for hyperparameter tuning experiments"""
    
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir
        self.config_modifier = ConfigModifier(CONFIG_FILE)
        self.results = []
        
        # Create timestamp for this tuning session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = os.path.join(
            results_dir, 
            f"tuning_session_{self.session_timestamp}.json"
        )
        
    def run_experiment(self, params: Dict[str, Any], experiment_name: str = None) -> Dict[str, Any]:
        """
        Run a single experiment with given parameters.
        
        Args:
            params: Dictionary of parameters to test
            experiment_name: Optional name for this experiment
            
        Returns:
            Dictionary containing experiment results
        """
        if experiment_name is None:
            experiment_name = f"exp_{len(self.results) + 1}"
        
        print("\n" + "="*70)
        print(f"Running Experiment: {experiment_name}")
        print("="*70)
        print("Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print()
        
        # Record experiment details
        experiment = {
            "name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": params.copy(),
            "status": "running"
        }
        
        try:
            # Modify config
            self.config_modifier.modify_parameters(params)
            
            # Run chunking script
            start_time = time.time()
            
            # Set UTF-8 encoding for Windows to handle Unicode characters
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(
                [sys.executable, CHUNKING_SCRIPT],
                capture_output=True,
                text=True,
                encoding='utf-8',
                env=env,
                errors='replace'  # Replace encoding errors instead of crashing
            )
            
            duration = time.time() - start_time
            
            # Record results
            experiment["duration_seconds"] = duration
            experiment["exit_code"] = result.returncode
            experiment["stdout"] = result.stdout
            experiment["stderr"] = result.stderr
            experiment["status"] = "success" if result.returncode == 0 else "failed"
            
            # Extract metrics from output
            experiment["metrics"] = self._extract_metrics(result.stdout)
            
            print(f"\n{'✓' if result.returncode == 0 else '✗'} Experiment completed in {duration:.2f}s")
            if experiment["metrics"]:
                print("Metrics:")
                for key, value in experiment["metrics"].items():
                    print(f"  {key}: {value}")
            
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                
        except Exception as e:
            experiment["status"] = "error"
            experiment["error"] = str(e)
            print(f"✗ Experiment failed with error: {e}")
        
        self.results.append(experiment)
        self._save_session()
        
        return experiment
    
    def _extract_metrics(self, output: str) -> Dict[str, Any]:
        """Extract metrics from script output"""
        metrics = {}
        
        # Extract total time
        time_match = re.search(r'Total time taken:\s*(\d+\.?\d*)s', output)
        if time_match:
            metrics["total_time_seconds"] = float(time_match.group(1))
        
        # Extract chunk count
        chunk_match = re.search(r'Created\s+(\d+)\s+chunks from\s+(\d+)\s+documents', output)
        if chunk_match:
            metrics["total_chunks"] = int(chunk_match.group(1))
            metrics["total_documents"] = int(chunk_match.group(2))
        
        # Extract chunk size statistics
        stats_match = re.search(r'Chunk size statistics:\s*min=(\d+),\s*max=(\d+),\s*avg=([\d.]+)', output)
        if stats_match:
            metrics["chunk_size_min"] = int(stats_match.group(1))
            metrics["chunk_size_max"] = int(stats_match.group(2))
            metrics["chunk_size_avg"] = float(stats_match.group(3))
        
        # Extract collection count
        collection_match = re.search(r'Collection ready with\s+(\d+)\s+chunks', output)
        if collection_match:
            metrics["collection_chunk_count"] = int(collection_match.group(1))
        
        return metrics
    
    def _save_session(self):
        """Save current session results to JSON file"""
        session_data = {
            "session_timestamp": self.session_timestamp,
            "total_experiments": len(self.results),
            "results": self.results
        }
        
        with open(self.session_log, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"\n📊 Session results saved to: {self.session_log}")
    
    def run_grid_search(self, param_grid: Dict[str, List[Any]]):
        """
        Run grid search over parameter combinations.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values to test
            
        Example:
            param_grid = {
                'FIXED_SIZE_CHUNK_SIZE': [500, 1000, 1500],
                'FIXED_SIZE_OVERLAP': [100, 150, 200],
                'BATCH_SIZE': [25, 50]
            }
        """
        from itertools import product
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        print(f"\n🔬 Starting grid search with {len(combinations)} combinations")
        print("="*70)
        print(f"\nParameter grid:")
        for key, values in param_grid.items():
            print(f"  {key}: {values}")
        print(f"\nTotal combinations: {len(combinations)}")
        print("="*70)
        
        for i, combination in enumerate(combinations, 1):
            params = dict(zip(keys, combination))
            
            print(f"\n[Progress: {i}/{len(combinations)}]")
            
            # Create descriptive name from parameter values
            param_str = "_".join([f"{k.replace('FIXED_SIZE_', '').replace('CHUNK_SIZE_MAX_BY_', '').replace('CHUNK_', '').lower()}{v}" 
                                  for k, v in params.items()])
            experiment_name = f"grid_{i:03d}_{param_str}"
            
            self.run_experiment(params, experiment_name)
            
            # Add small delay between experiments
            time.sleep(1)
        
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of all experiments"""
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*70)
        
        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] != "success"]
        
        print(f"\nTotal experiments: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print("\n--- Top 5 Results (by total time) ---")
            sorted_results = sorted(
                successful, 
                key=lambda x: x.get("duration_seconds", float('inf'))
            )[:5]
            
            for i, result in enumerate(sorted_results, 1):
                print(f"\n{i}. {result['name']}")
                print(f"   Duration: {result.get('duration_seconds', 'N/A'):.2f}s")
                print("   Parameters:")
                for key, value in result['parameters'].items():
                    print(f"     {key}: {value}")
                if result.get('metrics'):
                    print("   Metrics:")
                    for key, value in result['metrics'].items():
                        print(f"     {key}: {value}")
        
        print("\n" + "="*70)
    
    def cleanup(self):
        """Restore original config file"""
        print("\n🔄 Restoring original configuration...")
        self.config_modifier.restore_config()
        print("✓ Original configuration restored")


# =============================================================================
# PREDEFINED EXPERIMENT CONFIGURATIONS
# =============================================================================

def quick_test():
    """Quick test with 2-3 parameter combinations"""
    tuner = HyperparameterTuner()
    tuner.config_modifier.backup_config()
    
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


def chunk_size_optimization():
    """Test different chunk sizes and overlaps"""
    tuner = HyperparameterTuner()
    tuner.config_modifier.backup_config()
    
    try:
        param_grid = {
            'CHUNK_STRATEGY': ['fixed_size'],
            'FIXED_SIZE_CHUNK_SIZE': [500, 750, 1000, 1250, 1500, 2000, 2500],
            'FIXED_SIZE_OVERLAP': [100, 250, 400],
            'BATCH_SIZE': [50, 150]
        }
        
        tuner.run_grid_search(param_grid)
        
    finally:
        tuner.cleanup()


def batch_size_optimization():
    """Test different batch sizes for embedding generation"""
    tuner = HyperparameterTuner()
    tuner.config_modifier.backup_config()
    
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


def strategy_comparison():
    """Compare fixed_size vs by_sentence chunking strategies"""
    tuner = HyperparameterTuner()
    tuner.config_modifier.backup_config()
    
    try:
        # Fixed size tests
        tuner.run_experiment({
            'CHUNK_STRATEGY': 'fixed_size',
            'FIXED_SIZE_CHUNK_SIZE': 1000,
            'FIXED_SIZE_OVERLAP': 150,
            'BATCH_SIZE': 50
        }, "fixed_size_1000_150")
        
        tuner.run_experiment({
            'CHUNK_STRATEGY': 'fixed_size',
            'FIXED_SIZE_CHUNK_SIZE': 1500,
            'FIXED_SIZE_OVERLAP': 200,
            'BATCH_SIZE': 50
        }, "fixed_size_1500_200")
        
        # Sentence-based tests
        tuner.run_experiment({
            'CHUNK_STRATEGY': 'by_sentence',
            'CHUNK_SIZE_MAX_BY_SENTENCE': 1000,
            'BATCH_SIZE': 50
        }, "by_sentence_1000")
        
        tuner.run_experiment({
            'CHUNK_STRATEGY': 'by_sentence',
            'CHUNK_SIZE_MAX_BY_SENTENCE': 1500,
            'BATCH_SIZE': 50
        }, "by_sentence_1500")
        
    finally:
        tuner.cleanup()


def custom_experiment(param_sets: List[Dict[str, Any]]):
    """
    Run custom experiments with user-defined parameters.
    
    Args:
        param_sets: List of parameter dictionaries to test
        
    Example:
        custom_experiment([
            {'FIXED_SIZE_CHUNK_SIZE': 800, 'FIXED_SIZE_OVERLAP': 120},
            {'FIXED_SIZE_CHUNK_SIZE': 1200, 'FIXED_SIZE_OVERLAP': 180},
        ])
    """
    tuner = HyperparameterTuner()
    tuner.config_modifier.backup_config()
    
    try:
        for i, params in enumerate(param_sets, 1):
            tuner.run_experiment(params, f"custom_{i:03d}")
    finally:
        tuner.cleanup()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RAG HYPERPARAMETER TUNING SCRIPT")
    print("="*70)
    print("\nAvailable test configurations:")
    print("  1. quick_test() - Test 3 basic configurations")
    print("  2. chunk_size_optimization() - Grid search over chunk sizes and overlaps")
    print("  3. batch_size_optimization() - Test different batch sizes")
    print("  4. strategy_comparison() - Compare chunking strategies")
    print("  5. custom_experiment() - Run your own parameter sets")
    print("\nExample usage:")
    print("  python hyperparameter_tuner.py")
    print("\nOr import and use in your own script:")
    print("  from hyperparameter_tuner import HyperparameterTuner")
    print("  tuner = HyperparameterTuner()")
    print("  tuner.config_modifier.backup_config()")
    print("  tuner.run_experiment({'FIXED_SIZE_CHUNK_SIZE': 800, ...})")
    print("  tuner.cleanup()")
    print("="*70)
    
    # Default: Run quick test
    print("\n🚀 Running quick test by default...")
    print("💡 Edit this file to run different tests or import as a module\n")
    
    chunk_size_optimization()
    #batch_size_optimization()
