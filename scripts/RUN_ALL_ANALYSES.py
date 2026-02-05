#!/usr/bin/env python3
"""
=============================================================================
MASTER SCRIPT: RUN ALL ANALYSES
=============================================================================

This script runs all analysis steps in the correct order.

OUTPUT: All terminal output is saved to results/terminal_output.md

=============================================================================
"""

import subprocess
import sys
import os
from datetime import datetime

# Change to scripts directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# =============================================================================
# TERMINAL OUTPUT CAPTURE
# =============================================================================

class TeeWriter:
    """Write to both console and file simultaneously."""
    
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.file = open(file_path, 'w', encoding='utf-8')
        # Write markdown header
        self.file.write(f"# Analysis Run Output\n\n")
        self.file.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.file.write("```\n")
        self.file.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.write("```\n")
        self.file.close()


# Determine output file path
import config
OUTPUT_LOG = os.path.join(config.OUTPUT_DIR, 'terminal_output.md')

# Find python3 executable
PYTHON = sys.executable if sys.executable else '/usr/bin/python3'

def log(msg):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run_script(script_name, description):
    """Run a Python script and check for errors."""
    log(f"Running: {description}")
    print("=" * 60)
    
    # Capture output so it can be written through TeeWriter
    result = subprocess.run(
        [PYTHON, script_name],
        capture_output=True,
        text=True
    )
    
    # Write captured output through TeeWriter (via sys.stdout/stderr)
    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='')
    
    if result.returncode != 0:
        print(f"\n ERROR: {script_name} failed with return code {result.returncode}")
        return False
    
    print(f"✓ {description} completed successfully")
    return True

def main():
    print("=" * 80)
    print("MASTER SCRIPT: RUNNING ALL ANALYSES")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {SCRIPT_DIR}")
    
    # Check that clean database exists
    if not os.path.exists(config.MAIN_DB):
        print(f"\n ERROR: Main database not found: {config.MAIN_DB}")
        print("Please ensure gemini_esg_data.db is in the data/ directory.")
        sys.exit(1)
    
    print(f"\n✓ Using database: {os.path.basename(config.MAIN_DB)}")
    print(f"✓ Expected N: {config.SAMPLE_COUNTS['is_primary_event']} events")
    print(f"✓ Output log: {OUTPUT_LOG}")
    
    # Analysis scripts are in the analysis/ subdirectory
    ANALYSIS_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    
    # Define scripts to run in order (relative to analysis/ folder)
    scripts = [
        # Step 1: Data Preparation
        ("1_data_preparation.py", "Step 1: Data Preparation"),
        
        # Step 2: Hypothesis Tests
        ("2_h1_replication_fidelity.py", "Step 2: H1 - Replication Fidelity"),
        ("3_h2_finetuning_value.py", "Step 3: H2 - Fine-Tuning Value"),
        ("4_h3_structured_residuals.py", "Step 4: H3 - Structured Residuals"),
        
        # Step 3: Economic Value
        ("5_h4_economic_value.py", "Step 5a: H4 - Economic Value (Main)"),
        ("5b_h4_sector_neutral.py", "Step 5b: H4 - Sector Neutral"),
        ("5c_h4_ff_adjusted.py", "Step 5c: H4 - FF Adjusted"),
        ("5d_h4_sector_neutral_tertiles.py", "Step 5d: H4 - Tertiles"),
        ("5e_h4_in_vs_out_robustness.py", "Step 5e: H4 - In-Sample vs Out-of-Sample"),
        
        # Step 4: Additional Analyses
        ("6_enhancements.py", "Step 6: Additional Diagnostic Analyses"),
        
        # Step 5: Robustness Tests
        ("9_robustness_market_cap.py", "Step 7a: Robustness - Market Cap"),
        ("10_robustness_transaction_costs.py", "Step 7b: Robustness - Transaction Costs"),
        ("11_bootstrap_inference.py", "Step 7c: Robustness - Bootstrap Inference"),
        ("12_calendar_time_portfolio.py", "Step 7d: Robustness - Calendar Time"),
        
        # Step 6: Visualizations
        ("8_create_visualizations.py", "Step 8: Create Visualizations"),
    ]
    
    # Run each script
    success_count = 0
    failed_scripts = []
    
    for script_name, description in scripts:
        script_path = os.path.join(ANALYSIS_DIR, script_name)
        if os.path.exists(script_path):
            if run_script(script_path, description):
                success_count += 1
            else:
                failed_scripts.append(script_name)
                # Continue even if one fails
        else:
            print(f"\n WARNING: Script not found: {script_path}")
            failed_scripts.append(script_name)
    
    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scripts run successfully: {success_count}/{len(scripts)}")
    
    if failed_scripts:
        print(f"\n Failed scripts:")
        for s in failed_scripts:
            print(f"   - {s}")
    else:
        print("\n✓ All scripts completed successfully!")
    
    print("\n" + "=" * 80)
    print("OUTPUT LOCATIONS:")
    print("=" * 80)
    print(f"  Tables (CSV): {config.TABLES_DIR}")
    print(f"  Figures (PNG): {config.FIGURES_DIR}")
    print(f"  LaTeX tables: {config.LATEX_DIR}")
    print(f"  Terminal log: {OUTPUT_LOG}")
    
    print("\nNEXT STEPS:")
    print("  1. Review generated tables in results/csv/")
    print("  2. Review terminal_output.md for full run log")
    print("  3. Copy figures to paper directory")

if __name__ == "__main__":
    # Set up output capture
    tee = TeeWriter(OUTPUT_LOG)
    sys.stdout = tee
    sys.stderr = tee
    
    try:
        main()
    finally:
        # Restore stdout and close file
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.close()
        print(f"\n✓ Output saved to: {OUTPUT_LOG}")

