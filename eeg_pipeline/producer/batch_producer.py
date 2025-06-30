#!/usr/bin/env python3
"""
Batch EDF Data Generation Script
This script processes multiple EDF files in a folder to generate datasets
similar to what your producer.py creates.
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse

def find_edf_files(directory):
    """Find all EDF files in the specified directory and subdirectories."""
    edf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.edf'):
                full_path = os.path.join(root, file)
                edf_files.append(full_path)
    
    # Sort by subject and run number for consistent processing order
    def sort_key(filepath):
        filename = os.path.basename(filepath)
        # Extract subject number (S001, S002, etc.)
        try:
            if filename.startswith('S') and 'R' in filename:
                parts = filename.split('R')
                subject_num = int(parts[0][1:])  # Remove 'S' and convert to int
                run_num = int(parts[1].split('.')[0])  # Extract run number
                return (subject_num, run_num)
        except:
            pass
        return (999, 999)  # Fallback for non-standard filenames
    
    return sorted(edf_files, key=sort_key)

def run_producer_for_file(edf_file, output_dir, producer_script_path, 
                         bootstrap_servers='localhost:9092', 
                         emit_bandpower=True,
                         window_size=4.0, 
                         step_size=2.0,
                         batch_size=256):
    """
    Run the producer script for a single EDF file.
    """
    edf_filename = os.path.basename(edf_file)
    rel_path = os.path.relpath(edf_file, os.path.dirname(output_dir))
    
    print(f"\n{'='*60}")
    print(f"Processing: {rel_path}")
    print(f"{'='*60}")
    
    # Ensure output_dir exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Build the command
        cmd = [
            sys.executable, producer_script_path,
            '--edf-file', edf_file,
            '--bootstrap-servers', bootstrap_servers,
            '--batch-size', str(batch_size),
            '--window-size', str(window_size),
            '--step-size', str(step_size)
        ]
        
        if emit_bandpower:
            cmd.append('--emit-bandpower')

        # Run the producer from the correct working directory
        env = os.environ.copy()
        project_root = Path(__file__).resolve().parent.parent.parent
        env['PYTHONPATH'] = str(project_root)

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir, env=env)

        
        if result.returncode == 0:
            print(f"Successfully processed {edf_filename}")
            if "EDF loaded:" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "EDF loaded:" in line or "Total MNE Annotations Found:" in line:
                        print(f"   {line.strip()}")
        else:
            print(f"Error processing {edf_filename}")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            
    except Exception as e:
        print(f"Exception while processing {edf_filename}: {e}")
        print(f"  - Tried to run producer in directory: {output_dir}")
        print(f"  - Current working directory was: {os.getcwd()}")

def main():
    parser = argparse.ArgumentParser(description="Batch process EDF files to generate datasets")
    parser.add_argument('--edf-directory', required=True, 
                       help='Directory containing EDF files')
    parser.add_argument('--output-directory', required=True,
                       help='Directory where generated files will be saved')
    parser.add_argument('--producer-script', required=True,
                       help='Path to your producer.py script')
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                       help='Kafka bootstrap servers (default: localhost:9092)')
    parser.add_argument('--no-bandpower', action='store_true',
                       help='Disable band power computation')
    parser.add_argument('--window-size', type=float, default=4.0,
                       help='Window size in seconds (default: 4.0)')
    parser.add_argument('--step-size', type=float, default=2.0,
                       help='Step size in seconds (default: 2.0)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for EEG data (default: 256)')
    
    args = parser.parse_args()
    
    # Validate directories and files
    edf_dir = Path(args.edf_directory)
    output_dir = Path(args.output_directory)
    producer_script = Path(args.producer_script).resolve()

    
    if not edf_dir.exists():
        print(f"EDF directory does not exist: {edf_dir}")
        return
    
    if not producer_script.exists():
        print(f"Producer script does not exist: {producer_script}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all EDF files
    edf_files = find_edf_files(str(edf_dir))
    
    if not edf_files:
        print(f"No EDF files found in {edf_dir}")
        return
    
    print(f"Found {len(edf_files)} EDF files to process:")
    for edf_file in edf_files:
        # Show relative path from input directory for cleaner display
        rel_path = os.path.relpath(edf_file, str(edf_dir))
        print(f"  - {rel_path}")
    
    print(f"\nExpected to process subjects S001 through S109...")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Producer script: {producer_script.absolute()}")
    
    # Process each EDF file
    emit_bandpower = not args.no_bandpower
    successful = 0
    failed = 0
    
    for edf_file in edf_files:
        try:
            run_producer_for_file(
                edf_file=edf_file,
                output_dir=str(output_dir),
                producer_script_path=str(producer_script),
                bootstrap_servers=args.bootstrap_servers,
                emit_bandpower=emit_bandpower,
                window_size=args.window_size,
                step_size=args.step_size,
                batch_size=args.batch_size,
            )
            successful += 1

            #  DEBUG FILE LISTING FOR CURRENT EDF
            print(f"[DEBUG] Files in output directory ({output_dir}):")
            for f in sorted(os.listdir(output_dir)):
                print(f"  - {f}")

        except Exception as e:
            print(f"Failed to process {os.path.basename(edf_file)}: {e}")
            failed += 1


    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Output directory: {output_dir.absolute()}")
    
    # List generated files
    generated_files = list(output_dir.glob('*'))
    if generated_files:
        print(f"\n Generated files ({len(generated_files)}):")
        for file in sorted(generated_files):
            print(f"  - {file.name}")

if __name__ == "__main__":
    main()