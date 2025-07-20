#!/usr/bin/env python3
"""
JSONL Command Analysis Tool
==========================

Analyze robot command patterns from JSONL files (like asynchronous_deltas.jsonl).

Usage:
    python3 tools/analyze_commands.py --input output/robot_commands.jsonl
    python3 tools/analyze_commands.py --input ursim_test_v1/asynchronous_deltas.jsonl

Features:
    - Command frequency analysis
    - Movement pattern detection
    - Temporal analysis
    - Confidence statistics
    - Export to CSV/plots
"""

import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_jsonl(file_path):
    """Load JSONL file and parse entries."""
    data = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                entry = json.loads(line)
                entry['line_number'] = line_num
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: Invalid JSON - {e}")
                continue
    
    return data

def analyze_commands(data):
    """Analyze command patterns."""
    print("📊 Command Analysis")
    print("=" * 40)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    print(f"📈 Dataset Overview:")
    print(f"   Total entries: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Command frequency analysis
    if 'command' in df.columns:
        print(f"\n🎯 Command Frequency:")
        command_counts = df['command'].value_counts()
        for cmd, count in command_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {cmd:20}: {count:5d} ({percentage:5.1f}%)")
        
        # Plot command distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        command_counts.plot(kind='bar')
        plt.title('Command Frequency Distribution')
        plt.xlabel('Commands')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        command_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Command Distribution')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.savefig('output/command_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Movement analysis
    if all(col in df.columns for col in ['dx', 'dy', 'dz']):
        print(f"\n📐 Movement Analysis:")
        
        total_distance = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2).sum()
        print(f"   Total distance: {total_distance:.3f} meters")
        
        print(f"   Movement per axis:")
        print(f"     X-axis: {df['dx'].sum():8.3f} m (range: {df['dx'].min():.3f} to {df['dx'].max():.3f})")
        print(f"     Y-axis: {df['dy'].sum():8.3f} m (range: {df['dy'].min():.3f} to {df['dy'].max():.3f})")
        print(f"     Z-axis: {df['dz'].sum():8.3f} m (range: {df['dz'].min():.3f} to {df['dz'].max():.3f})")
        
        # Calculate cumulative position
        df['cum_x'] = df['dx'].cumsum()
        df['cum_y'] = df['dy'].cumsum() 
        df['cum_z'] = df['dz'].cumsum()
        
        # Plot movement patterns
        plt.figure(figsize=(15, 10))
        
        # 3D trajectory
        ax1 = plt.subplot(2, 3, 1, projection='3d')
        ax1.plot(df['cum_x'], df['cum_y'], df['cum_z'], 'b-', alpha=0.7)
        ax1.scatter(df['cum_x'].iloc[0], df['cum_y'].iloc[0], df['cum_z'].iloc[0], 
                   color='green', s=100, label='Start')
        ax1.scatter(df['cum_x'].iloc[-1], df['cum_y'].iloc[-1], df['cum_z'].iloc[-1], 
                   color='red', s=100, label='End')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('Z Position (m)')
        ax1.set_title('3D Robot Trajectory')
        ax1.legend()
        
        # X-Y plane
        plt.subplot(2, 3, 2)
        plt.plot(df['cum_x'], df['cum_y'], 'b-', alpha=0.7)
        plt.scatter(df['cum_x'].iloc[0], df['cum_y'].iloc[0], color='green', s=100, label='Start')
        plt.scatter(df['cum_x'].iloc[-1], df['cum_y'].iloc[-1], color='red', s=100, label='End')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Top View (X-Y Plane)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        # Movement deltas over time
        plt.subplot(2, 3, 3)
        plt.plot(df['dx'], 'r-', label='dx', alpha=0.7)
        plt.plot(df['dy'], 'g-', label='dy', alpha=0.7)
        plt.plot(df['dz'], 'b-', label='dz', alpha=0.7)
        plt.xlabel('Command Index')
        plt.ylabel('Movement Delta (m)')
        plt.title('Movement Deltas Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Position over time
        plt.subplot(2, 3, 4)
        plt.plot(df['cum_x'], 'r-', label='X position', alpha=0.7)
        plt.plot(df['cum_y'], 'g-', label='Y position', alpha=0.7)
        plt.plot(df['cum_z'], 'b-', label='Z position', alpha=0.7)
        plt.xlabel('Command Index')
        plt.ylabel('Cumulative Position (m)')
        plt.title('Position Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Movement magnitude histogram
        plt.subplot(2, 3, 5)
        movement_magnitude = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
        plt.hist(movement_magnitude, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Movement Magnitude (m)')
        plt.ylabel('Frequency')
        plt.title('Movement Magnitude Distribution')
        plt.grid(True, alpha=0.3)
        
        # Direction analysis
        plt.subplot(2, 3, 6)
        directions = []
        for _, row in df.iterrows():
            if abs(row['dx']) > abs(row['dy']) and abs(row['dx']) > abs(row['dz']):
                directions.append('X-axis')
            elif abs(row['dy']) > abs(row['dz']):
                directions.append('Y-axis')
            elif abs(row['dz']) > 0:
                directions.append('Z-axis')
            else:
                directions.append('No movement')
        
        direction_counts = pd.Series(directions).value_counts()
        direction_counts.plot(kind='bar')
        plt.title('Primary Movement Direction')
        plt.xlabel('Axis')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/movement_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Confidence analysis (if available)
    if 'confidence' in df.columns:
        print(f"\n🎯 Confidence Analysis:")
        print(f"   Mean confidence: {df['confidence'].mean():.3f}")
        print(f"   Std confidence:  {df['confidence'].std():.3f}")
        print(f"   Min confidence:  {df['confidence'].min():.3f}")
        print(f"   Max confidence:  {df['confidence'].max():.3f}")
        
        # Confidence distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(df['confidence'], 'b-', alpha=0.7)
        plt.xlabel('Command Index')
        plt.ylabel('Confidence Score')
        plt.title('Confidence Over Time')
        plt.grid(True, alpha=0.3)
        
        # Confidence by command
        plt.subplot(1, 3, 3)
        if 'command' in df.columns:
            conf_by_cmd = df.groupby('command')['confidence'].mean().sort_values()
            conf_by_cmd.plot(kind='bar')
            plt.title('Average Confidence by Command')
            plt.xlabel('Command')
            plt.ylabel('Average Confidence')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Temporal analysis (if timestamps available)
    if 'timestamp' in df.columns:
        print(f"\n⏰ Temporal Analysis:")
        
        # Convert timestamps to datetime
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            duration = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds()
            print(f"   Session duration: {duration:.1f} seconds")
            print(f"   Average command rate: {len(df)/duration:.2f} commands/second")
            
            # Time between commands
            time_diffs = df['datetime'].diff().dt.total_seconds().dropna()
            print(f"   Average time between commands: {time_diffs.mean():.3f} seconds")
            print(f"   Command rate std: {time_diffs.std():.3f} seconds")
            
        except Exception as e:
            print(f"   Could not parse timestamps: {e}")
    
    return df

def export_analysis(df, output_file="output/command_analysis.csv"):
    """Export analysis results to CSV."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n💾 Analysis exported to: {output_path}")

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze robot command JSONL files')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    parser.add_argument('--output', default='output/', help='Output directory')
    parser.add_argument('--export', action='store_true', help='Export to CSV')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return
    
    print(f"📂 Loading JSONL file: {input_path}")
    
    # Load and analyze data
    data = load_jsonl(input_path)
    
    if not data:
        print("❌ No valid data found in file")
        return
    
    df = analyze_commands(data)
    
    # Export if requested
    if args.export:
        export_analysis(df, f"{args.output}/command_analysis.csv")
    
    print(f"\n✅ Analysis completed!")
    print(f"📁 Plots saved to: output/")

if __name__ == "__main__":
    main()
