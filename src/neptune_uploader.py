"""
Neptune.ai Log Uploader
========================
Upload Jenkins test session logs and inference results to Neptune.ai for
data visualization, tracking, and evaluation.

This script can be run inside Jenkins containers to automatically upload
test results to Neptune.ai cloud platform.

Usage:
    python neptune_uploader.py --api-token YOUR_TOKEN --project YOUR_PROJECT
    python neptune_uploader.py --api-token YOUR_TOKEN --project YOUR_PROJECT --session-log path/to/session.log
    python neptune_uploader.py --upload-latest
    python neptune_uploader.py --upload-all

Environment Variables:
    NEPTUNE_API_TOKEN: Your Neptune.ai API token (can be used instead of --api-token)
    NEPTUNE_PROJECT: Your Neptune.ai project name (can be used instead of --project)

Author: Generated for 01_RAG project
Date: November 23, 2025
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    print("⚠️  Warning: neptune package not installed.")
    print("   Install with: pip install neptune")

from src.inference_logger import InferenceLogger


class NeptuneUploader:
    """Handler for uploading test results to Neptune.ai."""
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        project: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize Neptune uploader.
        
        Args:
            api_token: Neptune.ai API token (or use NEPTUNE_API_TOKEN env var)
            project: Neptune.ai project name (or use NEPTUNE_PROJECT env var)
            log_dir: Directory containing log files (defaults to tests/logs)
        """
        if not NEPTUNE_AVAILABLE:
            raise ImportError("neptune package is not installed. Install with: pip install neptune")
        
        # Get credentials from args or environment
        self.api_token = api_token or os.environ.get('NEPTUNE_API_TOKEN')
        self.project = project or os.environ.get('NEPTUNE_PROJECT')
        
        if not self.api_token:
            raise ValueError(
                "Neptune API token is required. "
                "Provide via --api-token or set NEPTUNE_API_TOKEN environment variable."
            )
        
        if not self.project:
            raise ValueError(
                "Neptune project name is required. "
                "Provide via --project or set NEPTUNE_PROJECT environment variable."
            )
        
        # Set up log directory
        if log_dir is None:
            self.log_dir = project_root / "tests" / "logs"
        else:
            self.log_dir = Path(log_dir)
        
        self.sessions_dir = self.log_dir / "sessions"
        self.inference_logger = InferenceLogger()
    
    def parse_session_log(self, log_file: Path) -> Dict[str, Any]:
        """
        Parse a session log file to extract metadata and results.
        Optimized for fast line-by-line parsing instead of regex on full content.
        
        Args:
            log_file: Path to the session log file
        
        Returns:
            Dictionary containing parsed session data
        """
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        session_data = {
            "session_name": log_file.stem,
            "log_file_path": str(log_file),
            "metadata": {},
            "hardware_info": {},
            "model_config": {},
            "rag_config": {},
            "test_results": [],
            "summary": {}
        }
        
        # Parse line by line for speed
        current_section = None
        current_question = None
        in_ram_section = False
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip()
                
                # Track sections
                if "PROJECT INFORMATION" in line:
                    current_section = "project"
                elif "HARDWARE INFORMATION" in line:
                    current_section = "hardware"
                elif "LLM MODEL CONFIGURATION" in line:
                    current_section = "model"
                elif "RAG SYSTEM CONFIGURATION" in line:
                    current_section = "rag"
                elif "SESSION SUMMARY" in line:
                    current_section = "summary"
                elif line.startswith("QUESTION "):
                    # Extract question ID
                    match = re.match(r"QUESTION (\d+)", line)
                    if match:
                        current_question = {
                            "question_id": int(match.group(1)),
                            "category": "",
                            "question": "",
                            "response_time": 0.0,
                            "chunks_retrieved": 0,
                            "answer_length": 0,
                            "success": False
                        }
                
                # Track RAM subsection
                if current_section == "hardware":
                    if "RAM:" in line:
                        in_ram_section = True
                    elif line.startswith("GPU") or line.startswith("CUDA") or line.startswith("Disk"):
                        in_ram_section = False
                
                # Extract metadata (quick simple checks)
                if "Session Start Time:" in line:
                    session_data["metadata"]["start_time"] = line.split(":", 1)[1].strip()
                elif "Session ID:" in line:
                    session_data["metadata"]["session_id"] = line.split(":", 1)[1].strip()
                elif "Test Mode:" in line:
                    session_data["metadata"]["test_mode"] = line.split(":", 1)[1].strip()
                elif "Selected Model:" in line:
                    session_data["metadata"]["model_name"] = line.split(":", 1)[1].strip()
                
                # Extract hardware info (simplified)
                elif current_section == "hardware":
                    if "System:" in line and "Operating" not in line:
                        session_data["hardware_info"]["os"] = line.split(":", 1)[1].strip()
                    elif "Processor:" in line:
                        session_data["hardware_info"]["cpu"] = line.split(":", 1)[1].strip()
                    elif "Total:" in line and in_ram_section:
                        session_data["hardware_info"]["ram_total"] = line.split(":", 1)[1].strip()
                    elif "Available:" in line and "CUDA" not in line_stripped:
                        session_data["hardware_info"]["cuda_available"] = line.split(":", 1)[1].strip()
                    elif "Version:" in line and "CUDA" in current_section or "cuda" in line.lower():
                        session_data["hardware_info"]["cuda_version"] = line.split(":", 1)[1].strip()
                
                # Extract model config
                elif current_section == "model":
                    if "Context Size (n_ctx):" in line:
                        try:
                            session_data["model_config"]["context_size"] = int(line.split(":", 1)[1].strip())
                        except:
                            pass
                    elif "Temperature:" in line:
                        try:
                            session_data["model_config"]["temperature"] = float(line.split(":", 1)[1].strip())
                        except:
                            pass
                    elif "GPU Layers:" in line:
                        try:
                            session_data["model_config"]["gpu_layers"] = int(line.split(":", 1)[1].strip())
                        except:
                            pass
                
                # Extract RAG config
                elif current_section == "rag":
                    if "Embedding Model:" in line:
                        session_data["rag_config"]["embedding_model"] = line.split(":", 1)[1].strip()
                    elif "Top K Results:" in line:
                        try:
                            session_data["rag_config"]["top_k"] = int(line.split(":", 1)[1].strip())
                        except:
                            pass
                    elif "Semantic Weight:" in line:
                        try:
                            session_data["rag_config"]["semantic_weight"] = float(line.split(":", 1)[1].strip())
                        except:
                            pass
                    elif "Keyword Weight:" in line:
                        try:
                            session_data["rag_config"]["keyword_weight"] = float(line.split(":", 1)[1].strip())
                        except:
                            pass
                
                # Extract summary stats
                elif current_section == "summary":
                    if "Total Tests Run:" in line:
                        try:
                            session_data["summary"]["total_tests"] = int(line.split(":", 1)[1].strip())
                        except:
                            pass
                    elif "Successful:" in line and ":" in line:
                        try:
                            session_data["summary"]["successful"] = int(line.split(":", 1)[1].strip())
                        except:
                            pass
                    elif "Failed:" in line:
                        try:
                            session_data["summary"]["failed"] = int(line.split(":", 1)[1].strip())
                        except:
                            pass
                    elif "Success Rate:" in line:
                        try:
                            session_data["summary"]["success_rate"] = float(line.split(":", 1)[1].replace("%", "").strip())
                        except:
                            pass
                    elif "Average:" in line and "seconds" in line:
                        try:
                            session_data["summary"]["avg_response_time"] = float(line.split(":")[1].split()[0].strip())
                        except:
                            pass
                    elif "Average Chunks Retrieved:" in line:
                        try:
                            session_data["summary"]["avg_chunks"] = float(line.split(":", 1)[1].strip())
                        except:
                            pass
                
                # Extract question details
                if current_question:
                    if "Category:" in line and not current_question["category"]:
                        current_question["category"] = line.split(":", 1)[1].strip()
                    elif "Response Time:" in line:
                        try:
                            current_question["response_time"] = float(line.split(":")[1].split()[0].strip())
                        except:
                            pass
                    elif "Chunks Retrieved:" in line:
                        try:
                            current_question["chunks_retrieved"] = int(line.split(":", 1)[1].strip())
                        except:
                            pass
                    elif "Answer Length:" in line:
                        try:
                            val = line.split(":", 1)[1].strip()
                            current_question["answer_length"] = int(val.split()[0])
                        except:
                            pass
                    elif "Success:" in line:
                        current_question["success"] = "True" in line
                        # Question complete, add to results
                        session_data["test_results"].append(current_question)
                        current_question = None
        
        return session_data
    
    def upload_session(
        self,
        session_log_path: Path,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Upload a single test session to Neptune.ai.
        
        Args:
            session_log_path: Path to the session log file
            tags: Optional list of tags for the run
            description: Optional description for the run
        
        Returns:
            Neptune run ID
        """
        print(f"\n{'='*80}")
        print(f"Uploading session: {session_log_path.name}")
        print(f"{'='*80}\n")
        
        # Parse session log
        print("📖 Parsing session log...")
        session_data = self.parse_session_log(session_log_path)
        
        # Initialize Neptune run
        print("🚀 Initializing Neptune run...")
        run = neptune.init_run(
            project=self.project,
            api_token=self.api_token,
            tags=tags or [],
            name=session_data["session_name"],
            description=description or f"RAG Test Session: {session_data['session_name']}"
        )
        
        print(f"✅ Neptune run created: {run['sys/id'].fetch()}")
        
        try:
            # Upload metadata
            print("📊 Uploading metadata...")
            for key, value in session_data["metadata"].items():
                run[f"metadata/{key}"] = value
            
            # Upload hardware info
            print("🖥️  Uploading hardware information...")
            for key, value in session_data["hardware_info"].items():
                if isinstance(value, list):
                    run[f"hardware/{key}"] = str(value)
                else:
                    run[f"hardware/{key}"] = value
            
            # Upload model configuration
            print("🤖 Uploading model configuration...")
            for key, value in session_data["model_config"].items():
                run[f"model_config/{key}"] = value
            
            # Upload RAG configuration
            print("🔍 Uploading RAG configuration...")
            for key, value in session_data["rag_config"].items():
                run[f"rag_config/{key}"] = value
            
            # Upload summary statistics
            print("📈 Uploading summary statistics...")
            for key, value in session_data["summary"].items():
                run[f"summary/{key}"] = value
            
            # Upload individual test results as metrics
            print("📝 Uploading test results...")
            for result in session_data["test_results"]:
                q_id = result["question_id"]
                run[f"tests/q{q_id}/response_time"].append(result["response_time"])
                run[f"tests/q{q_id}/chunks_retrieved"].append(result["chunks_retrieved"])
                run[f"tests/q{q_id}/answer_length"].append(result["answer_length"])
                run[f"tests/q{q_id}/success"].append(1 if result["success"] else 0)
                run[f"tests/q{q_id}/question"] = result["question"]
                run[f"tests/q{q_id}/category"] = result["category"]
            
            # Upload the actual log file
            print("📄 Uploading log file...")
            run["logs/session_log"].upload(str(session_log_path))
            
            # Check for environment report
            env_report = session_log_path.parent / f"{session_log_path.stem}_environment.txt"
            if env_report.exists():
                print("🌍 Uploading environment report...")
                run["logs/environment_report"].upload(str(env_report))
            
            # Create summary charts (Neptune will auto-generate some visualizations)
            print("📊 Creating summary visualizations...")
            
            # Response time per question
            if session_data["test_results"]:
                avg_times = [r["response_time"] for r in session_data["test_results"]]
                for idx, time in enumerate(avg_times, 1):
                    run["charts/response_times_by_question"].append(time, step=idx)
            
            # Chunks retrieved per question
            if session_data["test_results"]:
                chunks = [r["chunks_retrieved"] for r in session_data["test_results"]]
                for idx, chunk_count in enumerate(chunks, 1):
                    run["charts/chunks_by_question"].append(chunk_count, step=idx)
            
            # Answer length per question
            if session_data["test_results"]:
                answer_lengths = [r["answer_length"] for r in session_data["test_results"]]
                for idx, length in enumerate(answer_lengths, 1):
                    run["charts/answer_length_by_question"].append(length, step=idx)
            
            # Success rate per question
            if session_data["test_results"]:
                success_values = [1 if r["success"] else 0 for r in session_data["test_results"]]
                for idx, success in enumerate(success_values, 1):
                    run["charts/success_by_question"].append(success, step=idx)
            
            print(f"\n✅ Session uploaded successfully!")
            print(f"🔗 View in Neptune: {run.get_url()}")
            
            run_id = run["sys/id"].fetch()
            
        finally:
            # Stop the run
            run.stop()
        
        return run_id
    
    def upload_latest_session(
        self,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Upload the most recent test session log.
        
        Args:
            tags: Optional list of tags for the run
        
        Returns:
            Neptune run ID, or None if no sessions found
        """
        if not self.sessions_dir.exists():
            print(f"❌ Sessions directory not found: {self.sessions_dir}")
            return None
        
        # Find all session log files (excluding environment reports)
        log_files = [
            f for f in self.sessions_dir.glob("test_session_*.log")
            if not f.stem.endswith("_environment")
        ]
        
        if not log_files:
            print(f"❌ No session log files found in: {self.sessions_dir}")
            return None
        
        # Get most recent
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        
        return self.upload_session(latest_log, tags=tags)
    
    def upload_all_sessions(
        self,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Upload all test session logs to Neptune.ai.
        
        Args:
            tags: Optional list of tags for all runs
            limit: Maximum number of sessions to upload (most recent first)
        
        Returns:
            List of Neptune run IDs
        """
        if not self.sessions_dir.exists():
            print(f"❌ Sessions directory not found: {self.sessions_dir}")
            return []
        
        # Find all session log files (excluding environment reports)
        log_files = [
            f for f in self.sessions_dir.glob("test_session_*.log")
            if not f.stem.endswith("_environment")
        ]
        
        if not log_files:
            print(f"❌ No session log files found in: {self.sessions_dir}")
            return []
        
        # Sort by modification time (most recent first)
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        if limit:
            log_files = log_files[:limit]
        
        print(f"\n{'='*80}")
        print(f"Found {len(log_files)} session log(s) to upload")
        print(f"{'='*80}\n")
        
        run_ids = []
        
        for idx, log_file in enumerate(log_files, 1):
            print(f"\n[{idx}/{len(log_files)}] ", end="")
            try:
                run_id = self.upload_session(log_file, tags=tags)
                run_ids.append(run_id)
            except Exception as e:
                print(f"❌ Failed to upload {log_file.name}: {e}")
                continue
        
        print(f"\n{'='*80}")
        print(f"Upload complete: {len(run_ids)}/{len(log_files)} successful")
        print(f"{'='*80}\n")
        
        return run_ids
    
    def upload_inference_logs(
        self,
        tags: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> str:
        """
        Upload inference logs from InferenceLogger to Neptune.ai.
        
        Args:
            tags: Optional list of tags for the run
            model_name: Filter by model name
            limit: Maximum number of logs to upload
        
        Returns:
            Neptune run ID
        """
        print(f"\n{'='*80}")
        print(f"Uploading inference logs")
        print(f"{'='*80}\n")
        
        # Get logs from InferenceLogger
        logs = self.inference_logger.get_logs(
            limit=limit,
            model_name=model_name,
            success_only=False
        )
        
        if not logs:
            print("❌ No inference logs found")
            return None
        
        print(f"📊 Found {len(logs)} inference log(s)")
        
        # Initialize Neptune run
        run_name = f"Inference_Logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if model_name:
            run_name += f"_{model_name}"
        
        print("🚀 Initializing Neptune run...")
        run = neptune.init_run(
            project=self.project,
            api_token=self.api_token,
            tags=tags or ["inference_logs"],
            name=run_name,
            description="Inference logs from InferenceLogger"
        )
        
        print(f"✅ Neptune run created: {run['sys/id'].fetch()}")
        
        try:
            # Upload statistics
            print("📊 Uploading statistics...")
            stats = self.inference_logger.get_statistics(model_name=model_name)
            for key, value in stats.items():
                run[f"statistics/{key}"] = value
            
            # Upload individual logs
            print("📝 Uploading individual logs...")
            for log in logs:
                q_id = log.get("question_id", "unknown")
                timestamp = log.get("timestamp", "")
                
                run[f"logs/q{q_id}/timestamp"].append(timestamp)
                run[f"logs/q{q_id}/response_time"].append(log.get("response_time_seconds", 0))
                run[f"logs/q{q_id}/chunks_retrieved"].append(log.get("num_chunks_retrieved", 0))
                run[f"logs/q{q_id}/answer_length"].append(log.get("answer_length", 0))
                run[f"logs/q{q_id}/success"].append(1 if log.get("success") else 0)
                
                if log.get("question"):
                    run[f"logs/q{q_id}/question"] = log["question"][:200]
            
            # Upload CSV and JSONL files
            print("📄 Uploading log files...")
            if self.inference_logger.csv_log.exists():
                run["files/inference_summary.csv"].upload(str(self.inference_logger.csv_log))
            
            if self.inference_logger.jsonl_log.exists():
                run["files/inference_log.jsonl"].upload(str(self.inference_logger.jsonl_log))
            
            print(f"\n✅ Inference logs uploaded successfully!")
            print(f"🔗 View in Neptune: {run.get_url()}")
            
            run_id = run["sys/id"].fetch()
            
        finally:
            run.stop()
        
        return run_id


def main():
    """Main entry point for Neptune uploader."""
    parser = argparse.ArgumentParser(
        description="Upload Jenkins test logs to Neptune.ai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload the latest session
  python neptune_uploader.py --upload-latest
  
  # Upload a specific session
  python neptune_uploader.py --session-log tests/logs/sessions/test_session_Qwen3-8B_20251123.log
  
  # Upload all sessions (limit to 10 most recent)
  python neptune_uploader.py --upload-all --limit 10
  
  # Upload inference logs
  python neptune_uploader.py --upload-inference-logs --model Qwen3-8B-Q5_K_M
  
  # With custom tags
  python neptune_uploader.py --upload-latest --tags jenkins ci quick-test

Environment Variables:
  NEPTUNE_API_TOKEN: Your Neptune.ai API token
  NEPTUNE_PROJECT: Your Neptune.ai project (e.g., username/project-name)
        """
    )
    
    # Authentication
    parser.add_argument(
        '--api-token',
        help='Neptune.ai API token (or use NEPTUNE_API_TOKEN env var)'
    )
    
    parser.add_argument(
        '--project',
        help='Neptune.ai project name (or use NEPTUNE_PROJECT env var)'
    )
    
    # Upload modes
    parser.add_argument(
        '--session-log',
        type=Path,
        help='Path to specific session log file to upload'
    )
    
    parser.add_argument(
        '--upload-latest',
        action='store_true',
        help='Upload the most recent session log'
    )
    
    parser.add_argument(
        '--upload-all',
        action='store_true',
        help='Upload all session logs'
    )
    
    parser.add_argument(
        '--upload-inference-logs',
        action='store_true',
        help='Upload inference logs from InferenceLogger'
    )
    
    # Filters
    parser.add_argument(
        '--model',
        help='Filter by model name'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of sessions to upload'
    )
    
    parser.add_argument(
        '--tags',
        nargs='+',
        help='Tags to add to Neptune runs'
    )
    
    parser.add_argument(
        '--description',
        help='Description for the Neptune run'
    )
    
    # Directories
    parser.add_argument(
        '--log-dir',
        type=Path,
        help='Directory containing log files (defaults to tests/logs)'
    )
    
    args = parser.parse_args()
    
    # Validate that at least one upload mode is specified
    if not any([args.session_log, args.upload_latest, args.upload_all, args.upload_inference_logs]):
        parser.error(
            "Please specify an upload mode: "
            "--session-log, --upload-latest, --upload-all, or --upload-inference-logs"
        )
    
    if not NEPTUNE_AVAILABLE:
        print("\n❌ Neptune package not installed.")
        print("   Install with: pip install neptune")
        sys.exit(1)
    
    # Initialize uploader
    try:
        uploader = NeptuneUploader(
            api_token=args.api_token,
            project=args.project,
            log_dir=args.log_dir
        )
    except (ValueError, ImportError) as e:
        print(f"\n❌ Initialization failed: {e}")
        sys.exit(1)
    
    # Execute upload
    try:
        if args.session_log:
            uploader.upload_session(
                args.session_log,
                tags=args.tags,
                description=args.description
            )
        
        elif args.upload_latest:
            uploader.upload_latest_session(tags=args.tags)
        
        elif args.upload_all:
            uploader.upload_all_sessions(
                tags=args.tags,
                limit=args.limit
            )
        
        elif args.upload_inference_logs:
            uploader.upload_inference_logs(
                tags=args.tags,
                model_name=args.model,
                limit=args.limit
            )
        
        print("\n✅ All uploads completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
