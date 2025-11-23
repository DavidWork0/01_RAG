"""
Test Neptune.ai Integration
============================
Simple test script to verify Neptune.ai integration is working correctly.

Usage:
    python tests/test_neptune_integration.py
    python tests/test_neptune_integration.py --api-token YOUR_TOKEN --project YOUR_PROJECT

Author: Generated for 01_RAG project
Date: November 23, 2025
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_neptune_import():
    """Test if neptune package is installed."""
    print("="*80)
    print("TEST 1: Neptune Package Import")
    print("="*80)
    
    try:
        import neptune
        print("✅ neptune package imported successfully")
        print(f"   Version: {neptune.__version__}")
        return True
    except ImportError as e:
        print("❌ neptune package not installed")
        print(f"   Error: {e}")
        print("   Install with: pip install neptune")
        return False


def test_credentials(api_token=None, project=None):
    """Test if Neptune credentials are configured."""
    print("\n" + "="*80)
    print("TEST 2: Neptune Credentials")
    print("="*80)
    
    api_token = api_token or os.environ.get('NEPTUNE_API_TOKEN')
    project = project or os.environ.get('NEPTUNE_PROJECT')
    
    if not api_token:
        print("❌ NEPTUNE_API_TOKEN not configured")
        print("   Set via --api-token or NEPTUNE_API_TOKEN environment variable")
        return False, None, None
    
    if not project:
        print("❌ NEPTUNE_PROJECT not configured")
        print("   Set via --project or NEPTUNE_PROJECT environment variable")
        return False, None, None
    
    print("✅ Neptune credentials configured")
    print(f"   Project: {project}")
    print(f"   API Token: {'*' * 20}{api_token[-8:]}")
    
    return True, api_token, project


def test_connection(api_token, project):
    """Test connection to Neptune.ai."""
    print("\n" + "="*80)
    print("TEST 3: Neptune Connection")
    print("="*80)
    
    try:
        import neptune
        
        print("🔌 Testing connection to Neptune.ai...")
        
        # Try to initialize a test run
        run = neptune.init_run(
            project=project,
            api_token=api_token,
            tags=["test", "integration-check"],
            name="Neptune Integration Test",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False
        )
        
        print(f"✅ Successfully connected to Neptune.ai")
        print(f"   Run ID: {run['sys/id'].fetch()}")
        print(f"   URL: {run.get_url()}")
        
        # Log some test data
        run["test/status"] = "success"
        run["test/timestamp"] = str(Path(__file__).stat().st_mtime)
        
        print("✅ Test data logged successfully")
        
        # Stop the run
        run.stop()
        print("✅ Run stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n   Possible issues:")
        print("   - Invalid API token")
        print("   - Invalid project name")
        print("   - No internet connection")
        print("   - Neptune.ai service unavailable")
        return False


def test_uploader():
    """Test if neptune_uploader module can be imported."""
    print("\n" + "="*80)
    print("TEST 4: Neptune Uploader Module")
    print("="*80)
    
    try:
        from src.neptune_uploader import NeptuneUploader
        print("✅ NeptuneUploader imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import NeptuneUploader: {e}")
        return False


def test_log_files():
    """Test if log files exist."""
    print("\n" + "="*80)
    print("TEST 5: Test Log Files")
    print("="*80)
    
    sessions_dir = project_root / "tests" / "logs" / "sessions"
    
    if not sessions_dir.exists():
        print(f"⚠️  Sessions directory not found: {sessions_dir}")
        print("   No test logs to upload")
        return False
    
    log_files = list(sessions_dir.glob("test_session_*.log"))
    log_files = [f for f in log_files if not f.stem.endswith("_environment")]
    
    if not log_files:
        print(f"⚠️  No session log files found in: {sessions_dir}")
        print("   Run some tests first: python tests/test_inference.py --model Qwen3-8B-Q5_K_M --mode quick")
        return False
    
    print(f"✅ Found {len(log_files)} session log file(s)")
    
    # Show most recent 3
    log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    print("\n   Most recent logs:")
    for i, log_file in enumerate(log_files[:3], 1):
        print(f"   {i}. {log_file.name}")
    
    return True


def test_full_upload(api_token, project):
    """Test uploading the latest session to Neptune."""
    print("\n" + "="*80)
    print("TEST 6: Full Upload Test")
    print("="*80)
    
    try:
        from src.neptune_uploader import NeptuneUploader
        
        uploader = NeptuneUploader(
            api_token=api_token,
            project=project
        )
        
        print("🚀 Attempting to upload latest session...")
        run_id = uploader.upload_latest_session(tags=["test", "integration-check"])
        
        if run_id:
            print(f"✅ Upload successful!")
            print(f"   Run ID: {run_id}")
            return True
        else:
            print("⚠️  No sessions found to upload")
            return False
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test Neptune.ai integration")
    parser.add_argument('--api-token', help='Neptune.ai API token')
    parser.add_argument('--project', help='Neptune.ai project name')
    parser.add_argument('--skip-upload', action='store_true', help='Skip actual upload test')
    
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*22 + "Neptune.ai Integration Test" + " "*29 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    results = []
    
    # Test 1: Import
    results.append(("Import", test_neptune_import()))
    
    if not results[-1][1]:
        print("\n❌ Cannot proceed without neptune package")
        print("   Install with: pip install neptune")
        sys.exit(1)
    
    # Test 2: Credentials
    creds_ok, api_token, project = test_credentials(args.api_token, args.project)
    results.append(("Credentials", creds_ok))
    
    if not creds_ok:
        print("\n⚠️  Skipping connection tests (no credentials)")
    else:
        # Test 3: Connection
        results.append(("Connection", test_connection(api_token, project)))
    
    # Test 4: Uploader Module
    results.append(("Uploader Module", test_uploader()))
    
    # Test 5: Log Files
    results.append(("Log Files", test_log_files()))
    
    # Test 6: Full Upload (optional)
    if not args.skip_upload and creds_ok:
        # Check if running in non-interactive environment (like Jenkins)
        import sys
        if not sys.stdin.isatty():
            print("\n⚠️  Non-interactive environment detected (Jenkins/Docker)")
            print("   Skipping upload test - use --skip-upload flag explicitly")
            print("   To test upload in Jenkins, run: python src/neptune_uploader.py --upload-latest")
        else:
            print("\n⚠️  About to upload test data to Neptune.ai")
            print("   This will create a new run in your project.")
            try:
                response = input("   Continue? (y/n): ")
                if response.lower() == 'y':
                    results.append(("Full Upload", test_full_upload(api_token, project)))
                else:
                    print("   Skipped full upload test")
            except EOFError:
                print("\n   Skipped full upload test (non-interactive environment)")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Neptune integration is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
