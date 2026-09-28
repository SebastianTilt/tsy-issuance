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
import time

from config import CONFIG


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
        
        return pd.DataFrame(self._fetch_all_pages(CONFIG.mspd_base_url, params))
    
    def _fetch_all_pages(self, url: str, params: Dict[str, Any]) -> List[Dict]:
        """
        Fetch every page of a Fiscal Data query, using meta['total-pages'].
        
        Any failed page raises (after retries) instead of returning partial
        data, so an incomplete history is never cached or saved.
        """
        params = dict(params)
        all_rows: List[Dict] = []
        page = 1
        total_pages = 1
        
        while page <= total_pages:
            params['page[number]'] = str(page)
            data = self._fetch_with_retry(url, params)
            total_pages = int(data['meta']['total-pages'])
            rows = data.get('data', [])
            all_rows.extend(rows)
            print(f"    Page {page}/{total_pages}: {len(rows)} records (total: {len(all_rows):,})")
            page += 1
        
        expected = int(data['meta']['total-count'])
        if len(all_rows) != expected:
            raise RuntimeError(
                f"Fetched {len(all_rows):,} rows but the API reported {expected:,}"
            )
        return all_rows
    
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
                # Let failures propagate: a missing date range must not be
                # cached or saved as if it were the full history.
                df = future.result()
                if not df.empty:
                    dfs.append(df)
                print(f"  Completed: {start} -> {end or 'latest'}: {len(df):,} records")
        
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
        
        df = pd.DataFrame(self._fetch_all_pages(CONFIG.tga_base_url, params))
        
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
        
        - par_outstanding_usd: max of outstanding_amt (only one row per CUSIP
          carries it; the reopening tranche rows are null)
        - *_cum_usd: sum over tranches of the cumulative issued / redeemed /
          inflation-adjustment amounts. Monthly flows are derived from these
          in compute_bridge.
        """
        print("Collapsing reopenings...")
        grouped = df.groupby(
            ['record_date', 'CUSIP', 'maturity_date'], as_index=False
        ).agg(
            # Stock variable (point-in-time)
            par_outstanding_usd=('outstanding_amt', 'max'),
            
            # Cumulative variables (for flow and bridge calculation).
            # MSPD reports redeemed_amt as a NEGATIVE number, so per CUSIP
            # outstanding = issued_cum + infl_adj_cum + redeemed_cum.
            issued_cum_usd=('issued_amt', 'sum'),
            redeemed_cum_usd=('redeemed_amt', 'sum'),
            infl_adj_cum_usd=('inflation_adj_amt', 'sum'),
            
            # Metadata
            instrument_group=('security_class1_desc', 'first'),
            security_type=('security_type_desc', 'first'),
            orig_issue_date=('issue_date', 'min'),
            last_issue_date=('issue_date', 'max'),
            n_issue_dates=('issue_date', 'nunique'),
        )
        grouped['reopen_count'] = (grouped.pop('n_issue_dates') - 1).clip(lower=0)
        
        print(f"  Collapsed to {len(grouped):,} records")
        return grouped
    
    @staticmethod
    def compute_bridge(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute monthly flows and the stock-flow bridge for each CUSIP.
        
        Flows between consecutive snapshots t-1 and t:
        - issued_flow_usd   = Δ cumulative issued amount. This captures both
          reopening tranches (listed as separate rows since ~2014) and older
          reopenings that MSPD recorded by updating the original row in place.
        - infl_adj_usd      = Δ cumulative TIPS inflation adjustment
        - redeemed_flow_usd = -Δ cumulative redeemed (positive = outflow;
          buybacks and other early redemptions)
        - matured_usd       = par at t-1 of a CUSIP that has left the
          snapshot at t (it matured between the two snapshots)
        
        A CUSIP's first snapshot starts from zero (it did not exist at t-1).
        Matured CUSIPs get an "exit row" at t with par_outstanding_usd = 0,
        is_outstanding = False and matured_usd = prior par, so flow sums by
        tenor / instrument include maturities. Snapshot analytics must use
        is_outstanding rows only.
        
        Bridge: Δ par = issued + infl_adj - redeemed - matured, and
        bridge_error_usd = Δ par - that sum. Flows for the first snapshot
        date are NaN (there is no prior snapshot).
        """
        print("Computing stock-flow bridge...")
        df = df.copy()
        df['is_outstanding'] = True
        
        dates = sorted(df['record_date'].unique())
        prev_of = dict(zip(dates[1:], dates[:-1]))
        next_of = dict(zip(dates[:-1], dates[1:]))
        first_date = dates[0]
        
        # Prior-snapshot values, matched on the previous record date
        prev = df[['record_date', 'CUSIP', 'par_outstanding_usd', 'issued_cum_usd',
                   'redeemed_cum_usd', 'infl_adj_cum_usd']].rename(columns={
            'record_date': 'prev_date',
            'par_outstanding_usd': 'prev_par_outstanding',
            'issued_cum_usd': 'prev_issued_cum',
            'redeemed_cum_usd': 'prev_redeemed_cum',
            'infl_adj_cum_usd': 'prev_infl_adj_cum',
        })
        df['prev_date'] = df['record_date'].map(prev_of)
        df = df.merge(prev, on=['prev_date', 'CUSIP'], how='left')
        
        # New CUSIPs did not exist at t-1: their prior values are zero, not NaN
        prev_cols = ['prev_par_outstanding', 'prev_issued_cum',
                     'prev_redeemed_cum', 'prev_infl_adj_cum']
        is_new = df['prev_par_outstanding'].isna() & (df['record_date'] != first_date)
        df.loc[is_new, prev_cols] = 0.0
        
        df['issued_flow_usd'] = df['issued_cum_usd'] - df['prev_issued_cum']
        df['infl_adj_usd'] = df['infl_adj_cum_usd'] - df['prev_infl_adj_cum']
        df['redeemed_flow_usd'] = -(df['redeemed_cum_usd'] - df['prev_redeemed_cum'])
        df['matured_usd'] = np.where(df['record_date'] == first_date, np.nan, 0.0)
        
        # Exit rows: CUSIPs present at t-1 but gone at t
        present = df[['record_date', 'CUSIP']].assign(_present=True)
        exits = df[df['record_date'] != dates[-1]].copy()
        exits['next_date'] = exits['record_date'].map(next_of)
        exits = exits.merge(
            present.rename(columns={'record_date': 'next_date'}),
            on=['next_date', 'CUSIP'], how='left'
        )
        exits = exits[exits['_present'].isna()]
        exit_rows = pd.DataFrame({
            'record_date': exits['next_date'].values,
            'CUSIP': exits['CUSIP'].values,
            'maturity_date': exits['maturity_date'].values,
            'par_outstanding_usd': 0.0,
            'issued_cum_usd': 0.0,
            'redeemed_cum_usd': 0.0,
            'infl_adj_cum_usd': 0.0,
            'instrument_group': exits['instrument_group'].values,
            'security_type': exits['security_type'].values,
            'orig_issue_date': exits['orig_issue_date'].values,
            'last_issue_date': exits['last_issue_date'].values,
            'reopen_count': exits['reopen_count'].values,
            'is_outstanding': False,
            'prev_date': exits['record_date'].values,
            'prev_par_outstanding': exits['par_outstanding_usd'].values,
            'prev_issued_cum': exits['issued_cum_usd'].values,
            'prev_redeemed_cum': exits['redeemed_cum_usd'].values,
            'prev_infl_adj_cum': exits['infl_adj_cum_usd'].values,
            'issued_flow_usd': 0.0,
            'infl_adj_usd': 0.0,
            'redeemed_flow_usd': 0.0,
            'matured_usd': exits['par_outstanding_usd'].values,
        })
        n_early = int((exits['maturity_date'] > exits['next_date']).sum())
        print(f"  Matured CUSIPs (exit rows): {len(exit_rows):,}")
        if n_early:
            print(f"  WARNING: {n_early} CUSIPs left the snapshot before their maturity date")
        
        df = pd.concat([df, exit_rows], ignore_index=True)
        df = df.sort_values(['CUSIP', 'record_date']).reset_index(drop=True)
        
        # Bridge: sum of flows should equal the stock change
        df['delta_stock'] = df['par_outstanding_usd'] - df['prev_par_outstanding']
        df['flow_sum'] = (
            df['issued_flow_usd'] + df['infl_adj_usd']
            - df['redeemed_flow_usd'] - df['matured_usd']
        )
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
        
        # Net issuance (for a period): gross issuance plus TIPS accrual, less
        # buybacks/early redemptions and maturities. Equals the change in
        # outstanding par up to the bridge error.
        df['net_issuance_usd'] = (
            df['issued_flow_usd'] + df['infl_adj_usd']
            - df['redeemed_flow_usd'] - df['matured_usd']
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
        
        # Security type flags (exact security_class1_desc values used by MSPD)
        df['is_bill'] = df['instrument_group'] == 'Bills Maturity Value'
        df['is_tips'] = df['instrument_group'] == 'Inflation-Protected Securities'
        df['is_frn'] = df['instrument_group'] == 'Floating Rate Notes'
        
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
    
    # Monthly stock-flow reconciliation
    from analytics import PortfolioAnalytics
    recon = PortfolioAnalytics(final_df).compute_reconciliation()
    recon_path = CONFIG.data_dir / "reconciliation.csv"
    recon.to_csv(recon_path)
    print(f"  Reconciliation: {recon_path}")
    print("\nStock-flow reconciliation, last 6 months ($bn):")
    cols = ['delta_stock', 'gross_issued', 'matured', 'redeemed',
            'infl_adj', 'net_issuance', 'residual']
    print((recon[cols].tail(6) / 1e9).round(1).to_string())
    
    # Print summary
    latest = final_df[
        (final_df['record_date'] == final_df['record_date'].max()) &
        final_df['is_outstanding']
    ]
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
