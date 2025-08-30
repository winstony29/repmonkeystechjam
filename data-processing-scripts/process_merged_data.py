#!/usr/bin/env python3
"""
Merged Data Processor

This script combines three operations to process merged-{state}.json files:
1. Convert JSONL format to CSV
2. Clean the CSV data (remove blank reviews, rename columns)
3. Output clean CSV files as csv_{state}.csv

FOLDER SETUP:
Create a folder called "data_csv", and copy and update your base_path on line 40.
Place your merged-{state}.json files in the base directory.
Files should be in JSONL format (one JSON object per line).
You can modify the BASE_PATH variable at the top of the script to change the directory.

USAGE:
1. Process all merged files: python3 process_merged_data.py
2. Process specific state: python3 process_merged_data.py --state alaska
3. Include nested data: python3 process_merged_data.py --include-nested
4. Custom fields only: python3 process_merged_data.py --fields user_id,rating,text,business_name

OUTPUT:
- Creates csv_{state}.csv files with clean, processed data
- Creates backup files before cleaning
- Removes rows with blank review text
- Renames columns: reviewer_name -> author_name, review_text -> text
"""

import json
import csv
import pandas as pd
import argparse
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configuration
BASE_PATH = "/Users/gsharsh/Downloads/data"


def convert_timestamp(timestamp: Optional[int]) -> str:
    """Convert Unix timestamp to readable date."""
    if timestamp is None:
        return ""
    try:
        return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return str(timestamp)


def flatten_list(lst: Optional[List]) -> str:
    """Convert list to pipe-separated string."""
    if not lst:
        return ""
    return " | ".join(str(item) for item in lst)


def flatten_dict(d: Optional[Dict]) -> str:
    """Convert dictionary to JSON string."""
    if not d:
        return ""
    try:
        return json.dumps(d, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(d)


def extract_hours(hours_list: Optional[List]) -> str:
    """Extract and format business hours."""
    if not hours_list:
        return ""
    try:
        hours_dict = {day: time for day, time in hours_list}
        return json.dumps(hours_dict, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return flatten_list(hours_list)


def extract_pics_count(pics: Optional[List]) -> int:
    """Count number of pictures."""
    if not pics:
        return 0
    return len(pics)


def extract_pics_urls(pics: Optional[List]) -> str:
    """Extract picture URLs."""
    if not pics:
        return ""
    urls = []
    try:
        for pic in pics:
            if isinstance(pic, dict) and 'url' in pic:
                if isinstance(pic['url'], list):
                    urls.extend(pic['url'])
                else:
                    urls.append(str(pic['url']))
    except (TypeError, KeyError):
        pass
    return " | ".join(urls)


def process_record(record: Dict, include_nested: bool = False) -> Dict[str, Any]:
    """Process a single JSON record into CSV-friendly format."""
    processed = {}
    
    # Basic fields
    processed['user_id'] = record.get('user_id', '')
    processed['author_name'] = record.get('name_x', '')  # Direct rename to author_name
    processed['business_name'] = record.get('name_y', '')
    processed['rating'] = record.get('rating', '')
    processed['text'] = record.get('text', '')  # Direct rename to text
    processed['review_time'] = convert_timestamp(record.get('time'))
    processed['review_timestamp'] = record.get('time', '')
    
    # Business information
    processed['gmap_id'] = record.get('gmap_id', '')
    processed['address'] = record.get('address', '')
    processed['description'] = record.get('description', '')
    processed['latitude'] = record.get('latitude', '')
    processed['longitude'] = record.get('longitude', '')
    processed['avg_rating'] = record.get('avg_rating', '')
    processed['num_of_reviews'] = record.get('num_of_reviews', '')
    processed['price'] = record.get('price', '')
    processed['state'] = record.get('state', '')
    processed['url'] = record.get('url', '')
    
    # Categories
    processed['categories'] = flatten_list(record.get('category'))
    
    # Pictures
    processed['pics_count'] = extract_pics_count(record.get('pics'))
    if include_nested:
        processed['pics_urls'] = extract_pics_urls(record.get('pics'))
    
    # Business hours
    if include_nested:
        processed['hours'] = extract_hours(record.get('hours'))
    
    # Response from business
    resp = record.get('resp')
    if resp:
        processed['response_time'] = convert_timestamp(resp.get('time'))
        processed['response_text'] = resp.get('text', '')
    else:
        processed['response_time'] = ''
        processed['response_text'] = ''
    
    # MISC data (amenities, services, etc.)
    if include_nested:
        processed['misc_data'] = flatten_dict(record.get('MISC'))
        processed['relative_results'] = flatten_list(record.get('relative_results'))
    
    return processed


def get_csv_headers(include_nested: bool = False) -> List[str]:
    """Get CSV column headers with renamed columns."""
    headers = [
        'user_id',
        'author_name',  # Already renamed from reviewer_name
        'business_name',
        'rating',
        'text',  # Already renamed from review_text
        'review_time',
        'review_timestamp',
        'gmap_id',
        'address',
        'description',
        'latitude',
        'longitude',
        'avg_rating',
        'num_of_reviews',
        'price',
        'state',
        'url',
        'categories',
        'pics_count',
        'response_time',
        'response_text'
    ]
    
    if include_nested:
        headers.extend([
            'pics_urls',
            'hours',
            'misc_data',
            'relative_results'
        ])
    
    return headers


def process_merged_file(input_file: str, output_file: str, include_nested: bool = False, 
                       custom_fields: Optional[List[str]] = None) -> tuple:
    """
    Process a single merged-{state}.json file to csv_{state}.csv with cleaning.
    
    Returns:
        tuple: (success, total_records, clean_records, error_count)
    """
    
    print(f"Processing {input_file} -> {output_file}")
    print(f"Include nested data: {include_nested}")
    
    # Determine headers
    if custom_fields:
        headers = custom_fields
        print(f"Using custom fields: {', '.join(headers)}")
    else:
        headers = get_csv_headers(include_nested)
        print(f"Using {len(headers)} standard fields")
    
    total_records = 0
    clean_records = 0
    error_count = 0
    temp_records = []
    
    try:
        # Step 1: Convert JSONL to structured data
        print("Step 1: Converting JSONL to structured data...")
        
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    
                    if custom_fields:
                        # Extract only requested fields
                        processed = {field: record.get(field, '') for field in custom_fields}
                    else:
                        # Use standard processing
                        processed = process_record(record, include_nested)
                    
                    temp_records.append(processed)
                    total_records += 1
                    
                    # Progress indicator
                    if total_records % 1000 == 0:
                        print(f"  Processed {total_records} records...")
                
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    error_count += 1
        
        # Step 2: Clean the data using pandas
        print("Step 2: Cleaning data...")
        
        # Convert to DataFrame
        df = pd.DataFrame(temp_records)
        original_rows = len(df)
        
        # Clean the data (remove blank text rows)
        if 'text' in df.columns:
            df_cleaned = df.dropna(subset=['text'])  # Remove NaN values
            df_cleaned = df_cleaned[df_cleaned['text'].astype(str).str.strip() != '']  # Remove blank strings
            df_cleaned = df_cleaned[df_cleaned['text'].astype(str).str.strip() != 'nan']  # Remove 'nan' strings
        else:
            df_cleaned = df  # If no text column, keep all data
        
        clean_records = len(df_cleaned)
        rows_removed = original_rows - clean_records
        
        print(f"  Original records: {original_rows}")
        print(f"  Clean records: {clean_records}")
        print(f"  Records removed: {rows_removed}")
        
        # Step 3: Save to CSV
        print("Step 3: Saving to CSV...")
        
        # Create backup if file exists
        if Path(output_file).exists():
            backup_path = output_file.replace('.csv', '_backup.csv')
            if not Path(backup_path).exists():
                Path(output_file).rename(backup_path)
                print(f"  Backup created: {backup_path}")
        
        # Save cleaned data
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"✅ Successfully processed {input_file}")
        print(f"  - Total records processed: {total_records}")
        print(f"  - Clean records saved: {clean_records}")
        print(f"  - Errors: {error_count}")
        print(f"  - Output file size: {Path(output_file).stat().st_size:,} bytes")
        
        return True, total_records, clean_records, error_count
        
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        return False, total_records, clean_records, error_count


def find_merged_files(state: Optional[str] = None) -> List[str]:
    """Find merged-{state}.json files to process."""
    if state:
        pattern = f"{BASE_PATH}/merged-{state.title()}.json"
        files = glob.glob(pattern)
        if not files:
            # Try lowercase
            pattern = f"{BASE_PATH}/merged-{state.lower()}.json"
            files = glob.glob(pattern)
    else:
        files = glob.glob(f"{BASE_PATH}/merged-*.json")
    
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Process merged-{state}.json files to clean CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--state', type=str, help='Process specific state only (e.g., alaska)')
    parser.add_argument('--include-nested', action='store_true',
                       help='Include nested data (pics URLs, hours, misc data) as JSON strings')
    parser.add_argument('--fields', type=str,
                       help='Comma-separated list of specific fields to extract')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what files would be processed without actually processing them')
    
    args = parser.parse_args()
    
    # Find files to process
    files_to_process = find_merged_files(args.state)
    
    if not files_to_process:
        if args.state:
            print(f"❌ No merged files found for state: {args.state}")
            print(f"Looking for: {BASE_PATH}/merged-{args.state.title()}.json or {BASE_PATH}/merged-{args.state.lower()}.json")
        else:
            print(f"❌ No merged-*.json files found in directory: {BASE_PATH}")
        sys.exit(1)
    
    print(f"Found {len(files_to_process)} file(s) to process:")
    for file in files_to_process:
        state_name = Path(file).stem.replace('merged-', '')
        output_file = f"csv_{state_name}.csv"
        print(f"  {file} -> {output_file}")
    print()
    
    if args.dry_run:
        print("Dry run complete. Use without --dry-run to actually process files.")
        sys.exit(0)
    
    # Parse custom fields
    custom_fields = None
    if args.fields:
        custom_fields = [field.strip() for field in args.fields.split(',')]
    
    # Process each file
    total_files = len(files_to_process)
    successful_files = 0
    total_records_processed = 0
    total_clean_records = 0
    total_errors = 0
    
    for input_file in files_to_process:
        # Generate output filename
        state_name = Path(input_file).stem.replace('merged-', '')
        output_file = f"csv_{state_name}.csv"
        
        success, total_records, clean_records, errors = process_merged_file(
            input_file, output_file, args.include_nested, custom_fields
        )
        
        if success:
            successful_files += 1
            total_records_processed += total_records
            total_clean_records += clean_records
            total_errors += errors
        
        print()  # Add spacing between files
    
    # Summary
    print("="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Files found: {total_files}")
    print(f"Files successfully processed: {successful_files}")
    print(f"Files failed: {total_files - successful_files}")
    print(f"Total records processed: {total_records_processed:,}")
    print(f"Total clean records saved: {total_clean_records:,}")
    print(f"Total records removed: {total_records_processed - total_clean_records:,}")
    print(f"Total processing errors: {total_errors}")
    
    if total_records_processed > 0:
        percentage_removed = ((total_records_processed - total_clean_records) / total_records_processed) * 100
        print(f"Percentage of records removed: {percentage_removed:.2f}%")
    
    print("\nColumn transformations applied:")
    print("  name_x -> author_name")
    print("  text -> text (review text)")
    print("  Removed rows with blank/empty review text")
    
    if successful_files > 0:
        print(f"\n✅ Processing complete! Generated {successful_files} CSV file(s)")
    else:
        print(f"\n❌ No files were successfully processed")


if __name__ == '__main__':
    main()
