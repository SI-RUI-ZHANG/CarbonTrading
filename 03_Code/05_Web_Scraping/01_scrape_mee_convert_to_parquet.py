"""
Convert JSON documents to Parquet format
Run: python convert_to_parquet.py
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

def convert_json_to_parquet(input_dir, output_file=None):
    """Convert JSON files to single parquet file"""
    
    print(f"Converting JSON files from: {input_dir}")
    
    documents = []
    json_files = list(Path(input_dir).rglob('*.json'))
    
    # Skip progress.json
    json_files = [f for f in json_files if f.name != 'progress.json']
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                
                # Parse datetime strings back to datetime objects if needed
                if 'publish_date' in doc and isinstance(doc['publish_date'], str):
                    try:
                        doc['publish_date'] = pd.to_datetime(doc['publish_date'])
                    except:
                        pass
                
                if 'scraped_at' in doc and isinstance(doc['scraped_at'], str):
                    try:
                        doc['scraped_at'] = pd.to_datetime(doc['scraped_at'])
                    except:
                        pass
                
                documents.append(doc)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    if documents:
        # Create DataFrame
        df = pd.DataFrame(documents)
        
        # Sort by publish date if available
        if 'publish_date' in df.columns:
            df = df.sort_values('publish_date', ascending=False)
        
        # Generate output filename if not provided
        if not output_file:
            output_file = f"{input_dir}/mee_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        
        # Save to parquet
        df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        
        print(f"\nConversion completed!")
        print(f"Total documents: {len(documents)}")
        print(f"Output file: {output_file}")
        
        # Show summary
        print("\nDocument Summary:")
        print(f"Sections: {df['section'].value_counts().to_dict() if 'section' in df.columns else 'N/A'}")
        
        if 'publish_date' in df.columns:
            print(f"Date range: {df['publish_date'].min()} to {df['publish_date'].max()}")
        
        return df
    else:
        print("No documents found to convert")
        return None

def convert_jsonl_to_parquet(jsonl_file, output_file=None):
    """Convert JSONL file to parquet"""
    
    print(f"Converting JSONL file: {jsonl_file}")
    
    documents = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except Exception as e:
                    print(f"Error parsing line: {e}")
    
    if documents:
        df = pd.DataFrame(documents)
        
        # Parse datetime columns
        for col in ['publish_date', 'scraped_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Generate output filename if not provided
        if not output_file:
            base_name = Path(jsonl_file).stem
            output_file = str(Path(jsonl_file).parent / f"{base_name}.parquet")
        
        df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        
        print(f"Converted {len(documents)} documents to {output_file}")
        return df
    else:
        print("No documents found in JSONL file")
        return None

def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JSON documents to Parquet')
    parser.add_argument('--input', default='../../01_Data_Raw/03_Policy_Documents/MEE',
                       help='Input directory containing JSON files')
    parser.add_argument('--output', help='Output parquet file path')
    parser.add_argument('--jsonl', help='Convert specific JSONL file instead')
    
    args = parser.parse_args()
    
    if args.jsonl:
        # Convert JSONL file
        convert_jsonl_to_parquet(args.jsonl, args.output)
    else:
        # Convert directory of JSON files
        convert_json_to_parquet(args.input, args.output)

if __name__ == "__main__":
    main()