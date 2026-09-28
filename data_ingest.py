"""
Treasury Issuance Analysis - Data Ingestion Module
Handles fetching, caching, and initial processing of Treasury API data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import warnings
import time

from config import CONFIG

warnings.filterwarnings('ignore')


class DataFetcher:
    """Handles fetching data from Treasury APIs with caching and retry logic"""
    
    def __init__(self, use_cache: bool = True, cache_ttl_hours: int = 24):
        self.use_cache = use_cache
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.session = requests.Session()
        
    def _get_cache_path(self, cache_key: str) -> Path:
        """Generate cache file path from key"""
        hash_key = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return CONFIG.cache_dir / f"cache_{hash_key}.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is not expired"""
        if not cache_path.exists():
            return False
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - modified_time < self.cache_ttl
    
    def _fetch_with_retry(
        self, 
        url: str, 
        params: Dict[str, Any], 
        max_retries: int = 3
    ) -> Optional[Dict]:
        """Fetch URL with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=CONFIG.request_timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt
                print(f"    Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    print(f"    All {max_retries} attempts failed")
                    raise
        return None
    
    def fetch_mspd_range(
        self, 
        start_date: str, 
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Fetch MSPD data for a specific date range"""
        print(f"  Fetching MSPD range: {start_date} -> {end_date or 'latest'}")
        
        # Build filter
        filter_parts = [f"record_date:gte:{start_date}"]
        if end_date:
            filter_parts.append(f"record_date:lt:{end_date}")
        
        params = {
            'fields': (
                'record_date,security_type_desc,security_class1_desc,'
                'security_class2_desc,issue_date,maturity_date,'
                'outstanding_amt,issued_amt,redeemed_amt,inflation_adj_amt'
            ),
            'filter': ','.join(filter_parts),
            'sort': 'record_date,maturity_date,security_type_desc,security_class2_desc',
            'page[size]': str(CONFIG.page_size),
            'format': 'json'
        }
        
        all_rows = []
        page = 1
        
        while True:
            params['page[number]'] = str(page)
            
            try:
                data = self._fetch_with_retry(CONFIG.mspd_base_url, params)
                rows = data.get('data', [])
                
                if not rows:
                    break
                    
                all_rows.extend(rows)
                print(f"    Page {page}: {len(rows)} records (total: {len(all_rows):,})")
                page += 1
                
            except Exception as e:
                print(f"    Error on page {page}: {e}")
                break
        
        return pd.DataFrame(all_rows)
    
    def fetch_mspd_full_history(self) -> pd.DataFrame:
        """Fetch complete MSPD history using parallel requests"""
        cache_key = "mspd_full_history_" + "_".join(
            f"{s}_{e or 'now'}" for s, e in CONFIG.date_ranges
        )
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache
        if self.use_cache and self._is_cache_valid(cache_path):
            print("Loading MSPD data from cache...")
            return pd.read_parquet(cache_path)
        
        print("Fetching MSPD full history from Treasury API...")
        dfs: List[pd.DataFrame] = []
        
        # Parallel fetch
        with ThreadPoolExecutor(max_workers=len(CONFIG.date_ranges)) as executor:
            futures = {
                executor.submit(self.fetch_mspd_range, start, end): (start, end)
                for start, end in CONFIG.date_ranges
            }
            
            for future in as_completed(futures):
                start, end = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        dfs.append(df)
                        print(f"  Completed: {start} -> {end or 'latest'}: {len(df):,} records")
                except Exception as e:
                    print(f"  Failed: {start} -> {end or 'latest'}: {e}")
        
        if not dfs:
            raise RuntimeError("No MSPD data fetched from API")
        
        # Combine and deduplicate
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=['record_date', 'security_type_desc', 'security_class1_desc',
                   'security_class2_desc', 'issue_date', 'maturity_date']
        ).reset_index(drop=True)
        
        print(f"Total MSPD records: {len(combined):,}")
        
        # Cache the raw data
        if self.use_cache:
            combined.to_parquet(cache_path, index=False)
        
        return combined
    
    def fetch_tga_balance(self) -> pd.DataFrame:
        """Fetch TGA balance data"""
        cache_key = "tga_balance_latest"
        cache_path = self._get_cache_path(cache_key)
        
        if self.use_cache and self._is_cache_valid(cache_path):
            print("Loading TGA data from cache...")
            return pd.read_parquet(cache_path)
        
        print("Fetching TGA balance data...")
        
        params = {
            'fields': 'record_date,account_type,open_today_bal',
            'filter': 'account_type:eq:Treasury General Account (TGA) Closing Balance',
            'sort': '-record_date',
            'page[size]': str(CONFIG.page_size),
            'format': 'json'
        }
        
        all_rows = []
        page = 1
        max_pages = 10
        
        while page <= max_pages:
            params['page[number]'] = str(page)
            
            try:
                data = self._fetch_with_retry(CONFIG.tga_base_url, params)
                rows = data.get('data', [])
                
                if not rows:
                    break
                    
                all_rows.extend(rows)
                print(f"  Page {page}: {len(rows)} records")
                
                if len(rows) < CONFIG.page_size:
                    break
                page += 1
                
            except Exception as e:
                print(f"  Error on page {page}: {e}")
                if page == 1:
                    raise
                break
        
        df = pd.DataFrame(all_rows)
        
        if df.empty:
            raise ValueError("No TGA data fetched")
        
        # Process TGA data
        df['record_date'] = pd.to_datetime(df['record_date'])
        df['tga_balance_usd'] = pd.to_numeric(df['open_today_bal'], errors='coerce') * 1_000_000
        df = df[df['tga_balance_usd'].notna()].copy()
        
        tga = df[['record_date', 'tga_balance_usd']].copy()
        tga = tga.sort_values('record_date').drop_duplicates(
            subset=['record_date'], keep='last'
        ).reset_index(drop=True)
        
        print(f"TGA records: {len(tga):,}")
        print(f"TGA date range: {tga['record_date'].min()} to {tga['record_date'].max()}")
        
        if self.use_cache:
            tga.to_parquet(cache_path, index=False)
        
        return tga


class DataTransformer:
    """Transforms raw MSPD data into analysis-ready format"""
    
    @staticmethod
    def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw API data and convert types"""
        print("Cleaning and transforming data...")
        df = df.copy()
        
        # Handle null strings
        amount_cols = ['outstanding_amt', 'issued_amt', 'redeemed_amt', 'inflation_adj_amt']
        for col in amount_cols:
            if col in df.columns:
                df[col] = df[col].replace(['null', 'NULL', '', None], np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Scale from millions to dollars
                df[col] = df[col] * 1_000_000
        
        # Convert dates
        date_cols = ['record_date', 'issue_date', 'maturity_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean CUSIP identifier and keep only real CUSIPs. This drops the
        # subtotal rows ("Total Treasury Notes", "Total Marketable", ...) and
        # the Federal Financing Bank line, which have no CUSIP.
        df['CUSIP'] = df['security_class2_desc'].str.strip().str.upper()
        df = df[df['CUSIP'].str.match(r'^[0-9A-Z]{9}$', na=False)]

        # Remove rows with invalid critical fields. outstanding_amt is NOT
        # required: reopening tranche rows carry issued/redeemed/inflation
        # amounts but leave outstanding_amt null (only the first row of each
        # CUSIP carries the security-level outstanding).
        df = df.dropna(subset=['record_date', 'maturity_date'])
        
        print(f"  Cleaned records: {len(df):,}")
        return df
    
    @staticmethod
    def collapse_reopenings(df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse reopenings to one row per (record_date, CUSIP, maturity_date).
        
        IMPROVED LOGIC:
        - par_outstanding_usd: Use max (same security can't have different outstanding amounts)
        - issued_flow_usd: Sum only issues in the CURRENT month (not cumulative)
        - redeemed_flow_usd: Calculate as change in redemption cumulative, not raw sum
        """
        print("Collapsing reopenings...")
        df = df.copy()
        
        # Add period columns for flow calculations
        df['year_month'] = df['record_date'].dt.to_period('M')
        df['issue_year_month'] = df['issue_date'].dt.to_period('M')
        
        def aggregate_security(group):
            """Aggregate multiple records for the same CUSIP on the same date"""
            record_ym = group['year_month'].iloc[0]
            
            # Issues this month only (for flow calculation)
            current_month_issues = group[group['issue_year_month'] == record_ym]
            
            return pd.Series({
                # Stock variables (point-in-time). Only one row per CUSIP
                # carries outstanding_amt; max() skips the null tranche rows.
                'par_outstanding_usd': group['outstanding_amt'].max(),
                
                # Cumulative variables (for bridge calculation)
                'issued_cum_usd': group['issued_amt'].sum(),
                'redeemed_cum_usd': group['redeemed_amt'].sum(),  # This is cumulative redemptions
                'infl_adj_cum_usd': group['inflation_adj_amt'].sum(),
                
                # Monthly flow (issuance this month only)
                'issued_flow_usd': current_month_issues['issued_amt'].sum() if len(current_month_issues) > 0 else 0,
                
                # Metadata
                'instrument_group': group['security_class1_desc'].iloc[0],
                'security_type': group['security_type_desc'].iloc[0],
                'orig_issue_date': group['issue_date'].min(),
                'last_issue_date': group['issue_date'].max(),
                'reopen_count': max(0, group['issue_date'].nunique() - 1),
            })
        
        grouped = df.groupby(
            ['record_date', 'CUSIP', 'maturity_date'], 
            as_index=False
        ).apply(aggregate_security, include_groups=False).reset_index()
        
        # Clean up the index columns
        if 'level_0' in grouped.columns:
            grouped = grouped.drop(columns=['level_0'])
        if 'level_3' in grouped.columns:
            grouped = grouped.drop(columns=['level_3'])
        
        print(f"  Collapsed to {len(grouped):,} records")
        return grouped
    
    @staticmethod
    def compute_bridge(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute stock-flow bridge for reconciliation.
        
        IMPROVED LOGIC:
        Bridge equation: Δ Stock = Issued Flow + Inflation Adj + Redeemed Flow
        Where:
        - Δ Stock = par_outstanding(t) - par_outstanding(t-1)
        - Issued Flow = new issuance this period
        - Inflation Adj = TIPS adjustment
        - Redeemed Flow = Δ cumulative redemptions (negative)
        """
        print("Computing stock-flow bridge...")
        df = df.copy()
        df = df.sort_values(['CUSIP', 'record_date'])
        
        # Compute lagged values per CUSIP
        df['prev_par_outstanding'] = df.groupby('CUSIP')['par_outstanding_usd'].shift(1)
        df['prev_redeemed_cum'] = df.groupby('CUSIP')['redeemed_cum_usd'].shift(1)
        df['prev_infl_adj_cum'] = df.groupby('CUSIP')['infl_adj_cum_usd'].shift(1)
        
        # Stock change
        df['delta_stock'] = df['par_outstanding_usd'] - df['prev_par_outstanding']
        
        # Flow calculations
        # Redemption flow is CHANGE in cumulative redemptions (positive value = outflow).
        # First-period records (prev = NaN) are left as NaN — filling with 0 would
        # incorrectly treat the entire prior cumulative redemption balance as a
        # single-period flow.
        df['redeemed_flow_usd'] = df['redeemed_cum_usd'] - df['prev_redeemed_cum']

        # Inflation adjustment flow
        df['infl_adj_usd'] = df['infl_adj_cum_usd'] - df['prev_infl_adj_cum']

        # Bridge: sum of flows should equal stock change.
        # Redemptions reduce outstanding, so redeemed_flow_usd is subtracted.
        df['flow_sum'] = df['issued_flow_usd'] + df['infl_adj_usd'] - df['redeemed_flow_usd']
        
        # Bridge error (should be ~0 for clean data)
        df['bridge_error_usd'] = df['delta_stock'] - df['flow_sum']
        df['bridge_error_pct'] = np.where(
            df['par_outstanding_usd'].abs() > 0,
            df['bridge_error_usd'] / df['par_outstanding_usd'] * 100,
            0
        )
        
        return df
    
    @staticmethod
    def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
        """Add analytical fields for charting and analysis"""
        print("Adding derived fields...")
        df = df.copy()
        
        # Time-based calculations
        df['years_to_maturity'] = (
            (df['maturity_date'] - df['record_date']).dt.days / 365.25
        )
        df['orig_tenor_years'] = (
            (df['maturity_date'] - df['orig_issue_date']).dt.days / 365.25
        )
        df['age_years'] = (
            (df['record_date'] - df['orig_issue_date']).dt.days / 365.25
        )
        df['issue_year'] = df['orig_issue_date'].dt.year
        
        # Net issuance (for a period).
        # Redemptions reduce outstanding, so redeemed_flow_usd is subtracted.
        df['net_issuance_usd'] = (
            df['issued_flow_usd'] + df['infl_adj_usd'] - df['redeemed_flow_usd']
        )
        
        # Maturity bucket classification
        def classify_maturity_bucket(years):
            if pd.isna(years) or years < 0:
                return 'Unknown'
            for bucket, (low, high) in CONFIG.maturity_buckets.items():
                if low <= years < high:
                    return bucket
            return '20y+'  # Fallback for very long maturities
        
        df['maturity_bucket'] = df['years_to_maturity'].apply(classify_maturity_bucket)
        
        # Original tenor program classification
        def classify_tenor_program(years):
            if pd.isna(years) or years < 0:
                return 'Unknown'
            for program, (low, high) in CONFIG.tenor_programs.items():
                if low <= years < high:
                    return program
            return '30y'
        
        df['orig_tenor_program'] = df['orig_tenor_years'].apply(classify_tenor_program)
        
        # Security type flags
        df['is_bill'] = df['instrument_group'].str.contains('Bill', case=False, na=False)
        df['is_tips'] = df['instrument_group'].str.contains('TIPS', case=False, na=False)
        df['is_frn'] = df['instrument_group'].str.contains('FRN', case=False, na=False)
        
        # Year-month for aggregation
        df['year_month'] = df['record_date'].dt.to_period('M')
        
        print(f"  Final dataset: {len(df):,} records")
        print(f"  Date range: {df['record_date'].min()} to {df['record_date'].max()}")
        
        return df


def run_full_pipeline(use_cache: bool = True) -> pd.DataFrame:
    """
    Run the complete data pipeline: fetch, transform, save.
    
    Returns the final processed DataFrame.
    """
    print("=" * 80)
    print("Treasury MSPD Data Pipeline")
    print("=" * 80)
    
    # Fetch data
    fetcher = DataFetcher(use_cache=use_cache)
    raw_df = fetcher.fetch_mspd_full_history()
    
    # Transform
    transformer = DataTransformer()
    cleaned_df = transformer.clean_and_transform(raw_df)
    collapsed_df = transformer.collapse_reopenings(cleaned_df)
    bridge_df = transformer.compute_bridge(collapsed_df)
    final_df = transformer.add_derived_fields(bridge_df)
    
    # Save outputs
    parquet_path = CONFIG.data_dir / "mspd_issuance_panel.parquet"
    csv_path = CONFIG.data_dir / "mspd_issuance_panel.csv"
    
    final_df.to_parquet(parquet_path, index=False)
    final_df.to_csv(csv_path, index=False)
    
    print(f"\nData saved:")
    print(f"  Parquet: {parquet_path}")
    print(f"  CSV: {csv_path}")
    
    # Print summary
    latest = final_df[final_df['record_date'] == final_df['record_date'].max()]
    print(f"\nLatest snapshot ({latest['record_date'].iloc[0].date()}):")
    print(f"  Total Outstanding: ${latest['par_outstanding_usd'].sum()/1e12:.2f}T")
    print(f"  Number of CUSIPs: {latest['CUSIP'].nunique():,}")
    
    return final_df


def load_processed_data() -> pd.DataFrame:
    """Load previously processed data from parquet file"""
    parquet_path = CONFIG.data_dir / "mspd_issuance_panel.parquet"
    
    # Also check old location for backwards compatibility
    old_path = CONFIG.base_dir / "mspd_issuance_panel.parquet"
    
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif old_path.exists():
        return pd.read_parquet(old_path)
    else:
        raise FileNotFoundError(
            f"Processed data not found. Run `run_full_pipeline()` first.\n"
            f"Looked in: {parquet_path} and {old_path}"
        )


if __name__ == "__main__":
    # Run the full pipeline when executed directly
    df = run_full_pipeline(use_cache=True)
    print("\nPipeline complete!")
