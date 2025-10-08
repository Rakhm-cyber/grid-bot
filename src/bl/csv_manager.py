"""
Efficient CSV Manager for handling cryptocurrency rate data.
Optimized for performance with large datasets and frequent operations.
"""
from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

from src.clients.binance_client import RatePoint


@dataclass(frozen=True)
class CsvMetadata:
    """Metadata about CSV file structure and content."""
    filepath: str
    row_count: int
    columns: List[str]
    file_size: int
    last_modified: datetime
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None


class CsvManager:
    """
    High-performance CSV manager optimized for financial time series data.
    
    Features:
    - Memory-efficient streaming operations
    - Atomic file operations to prevent corruption
    - Fast duplicate detection and merging
    - Optimized for time-series data with timestamp indexing
    - Batch operations for better performance
    """
    
    def __init__(self, buffer_size: int = 8192):
        """
        Initialize CSV manager.
        
        Args:
            buffer_size: Buffer size for file operations (default: 8KB)
        """
        self.buffer_size = buffer_size
        self._default_fieldnames = ['ts_utc', 'close']
    
    def save_rates(self, rates: List[RatePoint], filepath: str, mode: str = 'w') -> None:
        """
        Save rate points to CSV file with atomic operation.
        
        Args:
            rates: List of RatePoint objects to save
            filepath: Target file path
            mode: File mode ('w' for overwrite, 'a' for append)
        """
        if not rates:
            return
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Use atomic write to prevent file corruption
        with self._atomic_write(filepath, mode) as f:
            writer = csv.DictWriter(f, fieldnames=self._default_fieldnames)
            
            # Write header only for new files or overwrite mode
            if mode == 'w' or (mode == 'a' and not os.path.exists(filepath)):
                writer.writeheader()
            
            # Write data in batches for better performance
            batch_size = 1000
            for i in range(0, len(rates), batch_size):
                batch = rates[i:i + batch_size]
                rows = [
                    {
                        'ts_utc': rate.ts.isoformat(),
                        'close': f"{rate.close:.8f}"
                    }
                    for rate in batch
                ]
                writer.writerows(rows)
    
    def load_rates(self, filepath: str, limit: Optional[int] = None) -> List[RatePoint]:
        """
        Load rate points from CSV file.
        
        Args:
            filepath: Source file path
            limit: Maximum number of records to load (None for all)
            
        Returns:
            List of RatePoint objects
        """
        if not os.path.exists(filepath):
            return []
        
        rates = []
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                
                try:
                    ts = datetime.fromisoformat(row['ts_utc'].replace('Z', '+00:00'))
                    close = float(row['close'])
                    rates.append(RatePoint(ts=ts, close=close))
                except (ValueError, KeyError) as e:
                    # Skip malformed rows but log the issue
                    print(f"Warning: Skipping malformed row {i+1} in {filepath}: {e}")
                    continue
        
        return rates
    
    def stream_rates(self, filepath: str) -> Iterator[RatePoint]:
        """
        Stream rate points from CSV file without loading all into memory.
        
        Args:
            filepath: Source file path
            
        Yields:
            RatePoint objects one by one
        """
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                try:
                    ts = datetime.fromisoformat(row['ts_utc'].replace('Z', '+00:00'))
                    close = float(row['close'])
                    yield RatePoint(ts=ts, close=close)
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping malformed row {i+1} in {filepath}: {e}")
                    continue
    
    def append_rates(self, rates: List[RatePoint], filepath: str, 
                    deduplicate: bool = True) -> int:
        """
        Append new rates to existing CSV file with optional deduplication.
        
        Args:
            rates: List of RatePoint objects to append
            filepath: Target file path
            deduplicate: Whether to remove duplicates based on timestamp
            
        Returns:
            Number of records actually added
        """
        if not rates:
            return 0
        
        existing_timestamps = set()
        
        # If deduplication is enabled and file exists, load existing timestamps
        if deduplicate and os.path.exists(filepath):
            existing_timestamps = self._get_existing_timestamps(filepath)
        
        # Filter out duplicates
        new_rates = []
        if deduplicate:
            for rate in rates:
                if rate.ts not in existing_timestamps:
                    new_rates.append(rate)
        else:
            new_rates = rates
        
        if new_rates:
            self.save_rates(new_rates, filepath, mode='a')
        
        return len(new_rates)
    
    def merge_csv_files(self, source_files: List[str], target_file: str, 
                       sort_by_timestamp: bool = True, deduplicate: bool = True) -> int:
        """
        Merge multiple CSV files into one with optional sorting and deduplication.
        
        Args:
            source_files: List of source CSV file paths
            target_file: Target merged file path
            sort_by_timestamp: Whether to sort by timestamp
            deduplicate: Whether to remove duplicate timestamps
            
        Returns:
            Total number of records in merged file
        """
        all_rates = []
        
        # Load all rates from source files
        for source_file in source_files:
            if os.path.exists(source_file):
                rates = self.load_rates(source_file)
                all_rates.extend(rates)
        
        if not all_rates:
            return 0
        
        # Sort by timestamp if requested
        if sort_by_timestamp:
            all_rates.sort(key=lambda r: r.ts)
        
        # Remove duplicates if requested
        if deduplicate:
            seen_timestamps = set()
            unique_rates = []
            for rate in all_rates:
                if rate.ts not in seen_timestamps:
                    unique_rates.append(rate)
                    seen_timestamps.add(rate.ts)
            all_rates = unique_rates
        
        # Save merged data
        self.save_rates(all_rates, target_file)
        return len(all_rates)
    
    def get_file_metadata(self, filepath: str) -> Optional[CsvMetadata]:
        """
        Get metadata about CSV file without loading all data.
        
        Args:
            filepath: CSV file path
            
        Returns:
            CsvMetadata object or None if file doesn't exist
        """
        if not os.path.exists(filepath):
            return None
        
        stat = os.stat(filepath)
        file_size = stat.st_size
        last_modified = datetime.fromtimestamp(stat.st_mtime)
        
        # Quick scan to get row count and timestamp range
        row_count = 0
        columns = []
        first_timestamp = None
        last_timestamp = None
        
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Get column names from header
            try:
                columns = next(reader)
            except StopIteration:
                columns = []
            
            # Count rows and get timestamp range
            for row in reader:
                row_count += 1
                if row and len(row) > 0:  # Ensure row has data
                    try:
                        ts = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                        if first_timestamp is None:
                            first_timestamp = ts
                        last_timestamp = ts
                    except (ValueError, IndexError):
                        continue
        
        return CsvMetadata(
            filepath=filepath,
            row_count=row_count,
            columns=columns,
            file_size=file_size,
            last_modified=last_modified,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp
        )
    
    def filter_by_date_range(self, filepath: str, start_date: datetime, 
                           end_date: datetime, output_file: str) -> int:
        """
        Filter CSV file by date range and save to new file.
        
        Args:
            filepath: Source CSV file path
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            output_file: Output file path
            
        Returns:
            Number of records in filtered file
        """
        filtered_rates = []
        
        for rate in self.stream_rates(filepath):
            if start_date <= rate.ts <= end_date:
                filtered_rates.append(rate)
        
        self.save_rates(filtered_rates, output_file)
        return len(filtered_rates)
    
    def get_latest_timestamp(self, filepath: str) -> Optional[datetime]:
        """
        Get the latest timestamp from CSV file efficiently.
        
        Args:
            filepath: CSV file path
            
        Returns:
            Latest timestamp or None if file is empty/doesn't exist
        """
        if not os.path.exists(filepath):
            return None
        
        # Read file from the end to find last valid timestamp
        try:
            with open(filepath, 'rb') as f:
                # Go to end of file
                f.seek(0, 2)
                file_size = f.tell()
                
                if file_size == 0:
                    return None
                
                # Read last chunk of file
                chunk_size = min(1024, file_size)
                f.seek(max(0, file_size - chunk_size))
                last_chunk = f.read().decode('utf-8')
                
                # Find last complete line with timestamp
                lines = last_chunk.strip().split('\n')
                for line in reversed(lines):
                    if line and ',' in line:
                        try:
                            timestamp_str = line.split(',')[0]
                            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        except (ValueError, IndexError):
                            continue
                            
        except Exception:
            pass
        
        return None
    
    def validate_csv_structure(self, filepath: str) -> Dict[str, Any]:
        """
        Validate CSV file structure and data integrity.
        
        Args:
            filepath: CSV file path
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'row_count': 0,
            'valid_rows': 0
        }
        
        if not os.path.exists(filepath):
            result['valid'] = False
            result['errors'].append('File does not exist')
            return result
        
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Check required columns
                if not reader.fieldnames:
                    result['valid'] = False
                    result['errors'].append('No header found')
                    return result
                
                required_cols = set(self._default_fieldnames)
                actual_cols = set(reader.fieldnames)
                missing_cols = required_cols - actual_cols
                
                if missing_cols:
                    result['valid'] = False
                    result['errors'].append(f'Missing required columns: {missing_cols}')
                
                # Validate data rows
                prev_timestamp = None
                for i, row in enumerate(reader):
                    result['row_count'] += 1
                    
                    try:
                        # Validate timestamp
                        ts_str = row.get('ts_utc', '')
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        
                        # Check chronological order
                        if prev_timestamp and ts < prev_timestamp:
                            result['warnings'].append(f'Row {i+1}: Timestamp out of order')
                        prev_timestamp = ts
                        
                        # Validate close price
                        close_str = row.get('close', '')
                        close = float(close_str)
                        
                        if close <= 0:
                            result['warnings'].append(f'Row {i+1}: Invalid close price: {close}')
                        
                        result['valid_rows'] += 1
                        
                    except (ValueError, TypeError) as e:
                        result['warnings'].append(f'Row {i+1}: Invalid data - {e}')
                        
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Error reading file: {e}')
        
        return result
    
    @contextmanager
    def _atomic_write(self, filepath: str, mode: str = 'w'):
        """
        Context manager for atomic file writing to prevent corruption.
        
        Args:
            filepath: Target file path
            mode: File mode
        """
        filepath = Path(filepath)
        temp_file = None
        
        try:
            # Create temporary file in same directory
            temp_file = tempfile.NamedTemporaryFile(
                mode=mode,
                dir=filepath.parent,
                prefix=f'.{filepath.name}.',
                suffix='.tmp',
                delete=False,
                newline='',
                encoding='utf-8'
            )
            
            yield temp_file
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            
            # Atomic move
            if mode == 'a' and filepath.exists():
                # For append mode, we need to merge with existing file
                with open(filepath, 'r', newline='', encoding='utf-8') as existing:
                    existing_content = existing.read()
                
                with open(temp_file.name, 'r', newline='', encoding='utf-8') as new:
                    new_content = new.read()
                
                with open(temp_file.name, 'w', newline='', encoding='utf-8') as merged:
                    merged.write(existing_content)
                    # Skip header in new content if it exists
                    lines = new_content.strip().split('\n')
                    if lines and lines[0].startswith('ts_utc'):
                        lines = lines[1:]
                    if lines:
                        merged.write('\n' + '\n'.join(lines))
            
            # Atomic replace
            os.replace(temp_file.name, filepath)
            temp_file = None
            
        except Exception:
            if temp_file:
                temp_file.close()
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass
            raise
    
    def _get_existing_timestamps(self, filepath: str) -> set:
        """
        Get set of existing timestamps from CSV file efficiently.
        
        Args:
            filepath: CSV file path
            
        Returns:
            Set of existing timestamps
        """
        timestamps = set()
        
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.fromisoformat(row['ts_utc'].replace('Z', '+00:00'))
                        timestamps.add(ts)
                    except (ValueError, KeyError):
                        continue
        except Exception:
            pass
        
        return timestamps
