#!/usr/bin/env python3
"""
Data Structure Inspector for Network Intrusion Detection System
This script helps you understand your data directory structure and what files are available.
"""

import os
import pandas as pd
from pathlib import Path
import json

def inspect_data_structure(data_dir="data_split"):
    """Inspect and report on the data directory structure."""
    
    print("🔍 Network IDS Data Structure Inspector")
    print("=" * 50)
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        print(f"Please ensure your data is in the '{data_dir}' directory")
        return False
    
    print(f"📁 Inspecting directory: {data_path.absolute()}")
    print()
    
    # Find all CSV files
    csv_files = list(data_path.rglob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files:")
    print("-" * 30)
    
    file_info = {}
    total_rows = 0
    
    for csv_file in sorted(csv_files):
        rel_path = csv_file.relative_to(data_path)
        
        try:
            # Get basic file info
            file_size = csv_file.stat().st_size / (1024 * 1024)  # MB
            
            # Try to read the CSV to get row count and columns
            df = pd.read_csv(csv_file, nrows=5)  # Just read first 5 rows for inspection
            full_df = pd.read_csv(csv_file)
            row_count = len(full_df)
            col_count = len(df.columns)
            
            total_rows += row_count
            
            # Categorize the file
            filename_lower = csv_file.stem.lower()
            if 'benign' in filename_lower:
                category = "🟢 Benign Traffic"
            elif any(attack in filename_lower for attack in ['icmp', 'udp', 'syn', 'http', 'scan', 'dos', 'flood']):
                category = "🔴 Attack Traffic"
            else:
                category = "❓ Unknown Type"
            
            file_info[str(rel_path)] = {
                'category': category,
                'size_mb': file_size,
                'rows': row_count,
                'columns': col_count,
                'sample_columns': list(df.columns[:10])  # First 10 columns
            }
            
            print(f"  {category}")
            print(f"    📄 {rel_path}")
            print(f"    📊 {row_count:,} rows, {col_count} columns, {file_size:.1f} MB")
            print(f"    🏷️  Sample columns: {', '.join(df.columns[:5])}...")
            print()
            
        except Exception as e:
            print(f"  ❌ Error reading {rel_path}: {e}")
            print()
    
    # Summary
    print("📈 Summary:")
    print("-" * 20)
    print(f"Total CSV files: {len(csv_files)}")
    print(f"Total data rows: {total_rows:,}")
    
    # Count by category
    categories = {}
    for info in file_info.values():
        cat = info['category']
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    for category, count in categories.items():
        print(f"{category}: {count} files")
    
    print()
    
    # Check for expected structure based on split_summary.json
    print("🎯 Expected Attack Types (based on 5G-NIDD dataset):")
    print("-" * 45)
    expected_attacks = [
        'Benign',
        'SYNScan', 
        'TCPConnectScan',
        'UDPScan',
        'ICMPFlood',
        'UDPFlood',
        'SYNFlood',
        'HTTPFlood',
        'SlowrateDoS'
    ]
    
    found_attacks = set()
    for file_path, info in file_info.items():
        filename = Path(file_path).stem.lower()
        for attack in expected_attacks:
            attack_lower = attack.lower()
            if attack_lower in filename or any(part in filename for part in attack_lower.split()):
                found_attacks.add(attack)
    
    for attack in expected_attacks:
        status = "✅" if attack in found_attacks else "❌"
        print(f"  {status} {attack}")
    
    print()
    
    # Directory structure
    print("📂 Directory Structure:")
    print("-" * 25)
    
    def print_tree(directory, prefix=""):
        """Print directory tree structure."""
        items = sorted(directory.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and not item.name.startswith('.'):
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension)
    
    print_tree(data_path)
    
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    print("-" * 20)
    
    if len(csv_files) == 0:
        print("  ❌ No CSV files found!")
        print("  📋 Please ensure your data is in CSV format")
    elif len(csv_files) < 5:
        print("  ⚠️  Only found a few CSV files")
        print("  📋 Expected at least 8-9 files (1 benign + 8 attack types)")
    else:
        print("  ✅ Good number of CSV files found")
    
    if 'Benign' not in found_attacks:
        print("  ⚠️  No benign traffic file detected")
        print("  📋 Look for a file with 'benign' in the name")
    
    missing_attacks = set(expected_attacks) - found_attacks
    if missing_attacks and 'Benign' in missing_attacks:
        missing_attacks.remove('Benign')
    
    if missing_attacks:
        print(f"  ⚠️  Missing attack types: {', '.join(missing_attacks)}")
        print("  📋 Check if files have different naming patterns")
    
    print()
    print("🚀 Next Steps:")
    print("-" * 15)
    print("  1. Run the fixed complete_faiss_indexer.py")
    print("  2. If successful, run the demo with: python demo_web_interface.py")
    print("  3. Or use the setup script: python setup_and_run.py --setup")
    
    return True

def main():
    """Main function."""
    inspect_data_structure()

if __name__ == "__main__":
    main()