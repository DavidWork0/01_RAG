"""
Environment Collector Module
=============================
Collects and formats environmental variables and Python environment information
for test sessions and debugging purposes.

Author: Generated for 01_RAG project
Date: November 18, 2025
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import json


def get_environment_variables() -> Dict[str, str]:
    """
    Get all environment variables.
    
    Returns:
        Dictionary of environment variable names and values
    """
    return dict(os.environ)


def get_filtered_environment_variables(sensitive_patterns: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Get environment variables with sensitive information filtered out.
    
    Args:
        sensitive_patterns: List of patterns to filter (case-insensitive)
                          Default: ['password', 'token', 'secret', 'key', 'api']
    
    Returns:
        Dictionary of filtered environment variables
    """
    if sensitive_patterns is None:
        sensitive_patterns = ['password', 'token', 'secret', 'key', 'api', 'credential']
    
    env_vars = get_environment_variables()
    filtered_vars = {}
    
    for key, value in env_vars.items():
        # Check if key contains any sensitive pattern
        is_sensitive = any(pattern.lower() in key.lower() for pattern in sensitive_patterns)
        
        if is_sensitive:
            filtered_vars[key] = "***REDACTED***"
        else:
            filtered_vars[key] = value
    
    return filtered_vars


def get_python_environment() -> Dict[str, any]:
    """
    Get comprehensive Python environment information.
    
    Returns:
        Dictionary with Python environment details
    """
    env_info = {
        'python_version': sys.version,
        'python_version_info': {
            'major': sys.version_info.major,
            'minor': sys.version_info.minor,
            'micro': sys.version_info.micro,
            'releaselevel': sys.version_info.releaselevel,
            'serial': sys.version_info.serial
        },
        'python_implementation': platform.python_implementation(),
        'python_compiler': platform.python_compiler(),
        'python_build': platform.python_build(),
        'executable': sys.executable,
        'prefix': sys.prefix,
        'base_prefix': sys.base_prefix,
        'exec_prefix': sys.exec_prefix,
        'base_exec_prefix': sys.base_exec_prefix,
        'path': sys.path,
        'platform': sys.platform,
        'maxsize': sys.maxsize,
        'is_virtual_env': sys.prefix != sys.base_prefix,
        'default_encoding': sys.getdefaultencoding(),
        'filesystem_encoding': sys.getfilesystemencoding(),
    }
    
    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        env_info['virtual_env_type'] = 'virtualenv' if hasattr(sys, 'real_prefix') else 'venv'
        if 'VIRTUAL_ENV' in os.environ:
            env_info['virtual_env_path'] = os.environ['VIRTUAL_ENV']
    
    return env_info


def get_installed_packages() -> List[Dict[str, str]]:
    """
    Get list of installed Python packages using pip.
    
    Returns:
        List of dictionaries with package name and version
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=json'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            packages = json.loads(result.stdout)
            return packages
        else:
            return [{'name': 'ERROR', 'version': f'pip list failed: {result.stderr}'}]
    
    except subprocess.TimeoutExpired:
        return [{'name': 'ERROR', 'version': 'pip list timed out'}]
    except Exception as e:
        return [{'name': 'ERROR', 'version': f'Failed to get packages: {str(e)}'}]


def get_pip_freeze() -> str:
    """
    Get pip freeze output (requirements format).
    
    Returns:
        String containing pip freeze output
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"ERROR: pip freeze failed\n{result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "ERROR: pip freeze timed out"
    except Exception as e:
        return f"ERROR: Failed to run pip freeze: {str(e)}"


def format_environment_report(
    include_env_vars: bool = True,
    include_python_env: bool = True,
    include_packages: bool = True,
    include_pip_freeze: bool = False,
    filter_sensitive: bool = True
) -> str:
    """
    Generate a formatted text report of the environment.
    
    Args:
        include_env_vars: Include environment variables section
        include_python_env: Include Python environment section
        include_packages: Include installed packages section
        include_pip_freeze: Include pip freeze output
        filter_sensitive: Filter sensitive information from env vars
    
    Returns:
        Formatted text report
    """
    lines = []
    
    lines.append("=" * 100)
    lines.append("ENVIRONMENT INFORMATION")
    lines.append("=" * 100)
    lines.append("")
    
    # Environment Variables
    if include_env_vars:
        lines.append("╔" + "═" * 98 + "╗")
        lines.append("║" + " " * 37 + "ENVIRONMENT VARIABLES" + " " * 40 + "║")
        lines.append("╚" + "═" * 98 + "╝")
        lines.append("")
        
        if filter_sensitive:
            env_vars = get_filtered_environment_variables()
            lines.append("Note: Sensitive variables (containing 'password', 'token', 'secret', 'key', 'api', 'credential')")
            lines.append("      have been redacted for security.")
            lines.append("")
        else:
            env_vars = get_environment_variables()
        
        lines.append("-" * 100)
        for key, value in sorted(env_vars.items()):
            # Truncate very long values
            if len(value) > 200:
                value = value[:197] + "..."
            lines.append(f"  {key:<40} = {value}")
        lines.append("")
    
    # Python Environment
    if include_python_env:
        lines.append("╔" + "═" * 98 + "╗")
        lines.append("║" + " " * 38 + "PYTHON ENVIRONMENT" + " " * 42 + "║")
        lines.append("╚" + "═" * 98 + "╝")
        lines.append("")
        
        py_env = get_python_environment()
        
        lines.append("Python Information:")
        lines.append("-" * 100)
        lines.append(f"  Python Version:         {py_env['python_version'].split()[0]}")
        lines.append(f"  Python Implementation:  {py_env['python_implementation']}")
        lines.append(f"  Python Compiler:        {py_env['python_compiler']}")
        lines.append(f"  Python Build:           {py_env['python_build'][0]} ({py_env['python_build'][1]})")
        lines.append(f"  Platform:               {py_env['platform']}")
        lines.append(f"  Max Size (bits):        {py_env['maxsize'].bit_length()}")
        lines.append("")
        
        lines.append("Executable and Paths:")
        lines.append("-" * 100)
        lines.append(f"  Python Executable:      {py_env['executable']}")
        lines.append(f"  Prefix:                 {py_env['prefix']}")
        lines.append(f"  Base Prefix:            {py_env['base_prefix']}")
        lines.append(f"  Exec Prefix:            {py_env['exec_prefix']}")
        lines.append(f"  Base Exec Prefix:       {py_env['base_exec_prefix']}")
        lines.append("")
        
        lines.append("Virtual Environment:")
        lines.append("-" * 100)
        lines.append(f"  Is Virtual Env:         {py_env['is_virtual_env']}")
        if py_env.get('virtual_env_type'):
            lines.append(f"  Virtual Env Type:       {py_env['virtual_env_type']}")
        if py_env.get('virtual_env_path'):
            lines.append(f"  Virtual Env Path:       {py_env['virtual_env_path']}")
        lines.append("")
        
        lines.append("Encoding:")
        lines.append("-" * 100)
        lines.append(f"  Default Encoding:       {py_env['default_encoding']}")
        lines.append(f"  Filesystem Encoding:    {py_env['filesystem_encoding']}")
        lines.append("")
        
        lines.append("Python Path (sys.path):")
        lines.append("-" * 100)
        for i, path in enumerate(py_env['path'], 1):
            lines.append(f"  {i:2}. {path}")
        lines.append("")
    
    # Installed Packages
    if include_packages:
        lines.append("╔" + "═" * 98 + "╗")
        lines.append("║" + " " * 38 + "INSTALLED PACKAGES" + " " * 42 + "║")
        lines.append("╚" + "═" * 98 + "╝")
        lines.append("")
        
        packages = get_installed_packages()
        
        lines.append(f"Total Packages: {len(packages)}")
        lines.append("-" * 100)
        lines.append(f"{'Package Name':<50} {'Version':<30}")
        lines.append("-" * 100)
        
        for pkg in sorted(packages, key=lambda x: x['name'].lower()):
            lines.append(f"  {pkg['name']:<48} {pkg['version']:<30}")
        lines.append("")
    
    # Pip Freeze
    if include_pip_freeze:
        lines.append("╔" + "═" * 98 + "╗")
        lines.append("║" + " " * 40 + "PIP FREEZE OUTPUT" + " " * 41 + "║")
        lines.append("╚" + "═" * 98 + "╝")
        lines.append("")
        
        freeze_output = get_pip_freeze()
        lines.append("Requirements Format (pip freeze):")
        lines.append("-" * 100)
        lines.extend(freeze_output.split('\n'))
        lines.append("")
    
    lines.append("=" * 100)
    lines.append("END OF ENVIRONMENT INFORMATION")
    lines.append("=" * 100)
    
    return '\n'.join(lines)


def write_environment_report(
    output_path: Path,
    include_env_vars: bool = True,
    include_python_env: bool = True,
    include_packages: bool = True,
    include_pip_freeze: bool = False,
    filter_sensitive: bool = True
) -> None:
    """
    Write environment report to a file.
    
    Args:
        output_path: Path to output file
        include_env_vars: Include environment variables section
        include_python_env: Include Python environment section
        include_packages: Include installed packages section
        include_pip_freeze: Include pip freeze output
        filter_sensitive: Filter sensitive information from env vars
    """
    report = format_environment_report(
        include_env_vars=include_env_vars,
        include_python_env=include_python_env,
        include_packages=include_packages,
        include_pip_freeze=include_pip_freeze,
        filter_sensitive=filter_sensitive
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


def main():
    """
    Main function for standalone usage.
    Generates an environment report and saves it.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate environment information report',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='environment_report.txt',
        help='Output file path (default: environment_report.txt)'
    )
    
    parser.add_argument(
        '--no-env-vars',
        action='store_true',
        help='Exclude environment variables'
    )
    
    parser.add_argument(
        '--no-python-env',
        action='store_true',
        help='Exclude Python environment information'
    )
    
    parser.add_argument(
        '--no-packages',
        action='store_true',
        help='Exclude installed packages list'
    )
    
    parser.add_argument(
        '--include-pip-freeze',
        action='store_true',
        help='Include pip freeze output (requirements format)'
    )
    
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Do not filter sensitive environment variables (use with caution!)'
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print(f"Generating environment report...")
    
    write_environment_report(
        output_path=output_path,
        include_env_vars=not args.no_env_vars,
        include_python_env=not args.no_python_env,
        include_packages=not args.no_packages,
        include_pip_freeze=args.include_pip_freeze,
        filter_sensitive=not args.no_filter
    )
    
    print(f"  [OK] Environment report written to: {output_path}")
    print(f"   File size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
