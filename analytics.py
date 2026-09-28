"""
Treasury Issuance Analysis - Analytics Module
Core analytical calculations for Treasury debt analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from config import CONFIG, get_bucket_order, get_tenor_order


@dataclass
class PortfolioMetrics:
    """Container for portfolio-level metrics at a point in time"""
    record_date: pd.Timestamp
    total_outstanding: float
    cusip_count: int
    weighted_avg_maturity: float
    weighted_avg_age: float
    short_share: float  # <= 3y
    long_share: float   # >= 7y
    bill_share: float
    tips_share: float
    
    def to_dict(self) -> Dict:
        return {
            'record_date': self.record_date,
            'total_outstanding_usd': self.total_outstanding,
            'cusip_count': self.cusip_count,
            'wam_years': self.weighted_avg_maturity,
            'waa_years': self.weighted_avg_age,
            'short_share_pct': self.short_share,
            'long_share_pct': self.long_share,
            'bill_share_pct': self.bill_share,
            'tips_share_pct': self.tips_share,
        }


class PortfolioAnalytics:
    """Calculate portfolio-level analytics from MSPD data"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with processed MSPD DataFrame"""
        self.df = df.copy()
        self._validate_data()
    
    def _validate_data(self):
        """Validate required columns exist"""
        required_cols = [
            'record_date', 'CUSIP', 'par_outstanding_usd', 
            'years_to_maturity', 'maturity_bucket', 'instrument_group',
            'is_outstanding',
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def get_snapshot(self, date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Get portfolio snapshot for a specific date (default: latest).
        
        Excludes the exit rows that record maturities (is_outstanding False).
        """
        if date is None:
            date = self.df['record_date'].max()
        mask = (self.df['record_date'] == date) & self.df['is_outstanding']
        return self.df[mask].copy()
    
    def compute_portfolio_metrics(
        self, 
        date: Optional[pd.Timestamp] = None
    ) -> PortfolioMetrics:
        """Compute key portfolio metrics for a date"""
        snapshot = self.get_snapshot(date)
        
        total = snapshot['par_outstanding_usd'].sum()

        # Weighted average maturity — exclude rows with missing/non-positive
        # years_to_maturity so already-matured or data-deficient records don't
        # pull the average toward zero.
        wam_mask = (
            snapshot['years_to_maturity'].notna() &
            (snapshot['years_to_maturity'] > 0) &
            (snapshot['par_outstanding_usd'] > 0)
        )
        wam_snap = snapshot[wam_mask]
        wam = (
            np.average(wam_snap['years_to_maturity'], weights=wam_snap['par_outstanding_usd'])
            if len(wam_snap) > 0 else 0
        )

        # Weighted average age — exclude rows with missing par or age
        waa_mask = (
            snapshot['age_years'].notna() &
            (snapshot['par_outstanding_usd'] > 0)
        )
        waa_snap = snapshot[waa_mask]
        waa = (
            np.average(waa_snap['age_years'], weights=waa_snap['par_outstanding_usd'])
            if len(waa_snap) > 0 else 0
        )
        
        # Short/long shares
        short_par = snapshot[snapshot['years_to_maturity'] <= 3]['par_outstanding_usd'].sum()
        long_par = snapshot[snapshot['years_to_maturity'] >= 7]['par_outstanding_usd'].sum()
        
        # Bill and TIPS shares
        bill_par = snapshot[snapshot['is_bill']]['par_outstanding_usd'].sum()
        tips_par = snapshot[snapshot['is_tips']]['par_outstanding_usd'].sum()
        
        return PortfolioMetrics(
            record_date=snapshot['record_date'].iloc[0] if len(snapshot) > 0 else date,
            total_outstanding=total,
            cusip_count=snapshot['CUSIP'].nunique(),
            weighted_avg_maturity=wam,
            weighted_avg_age=waa,
            short_share=(short_par / total * 100) if total > 0 else 0,
            long_share=(long_par / total * 100) if total > 0 else 0,
            bill_share=(bill_par / total * 100) if total > 0 else 0,
            tips_share=(tips_par / total * 100) if total > 0 else 0,
        )
    
    def compute_metrics_timeseries(self) -> pd.DataFrame:
        """Compute portfolio metrics for all dates"""
        dates = sorted(self.df['record_date'].unique())
        metrics = [self.compute_portfolio_metrics(d).to_dict() for d in dates]
        return pd.DataFrame(metrics)
    
    def compute_composition_by_bucket(self) -> pd.DataFrame:
        """Compute outstanding by maturity bucket over time"""
        bucket_order = get_bucket_order()
        
        agg = self.df.groupby(
            ['record_date', 'maturity_bucket']
        )['par_outstanding_usd'].sum().reset_index()
        
        pivot = agg.pivot(
            index='record_date', 
            columns='maturity_bucket', 
            values='par_outstanding_usd'
        ).fillna(0)
        
        # Reorder columns
        pivot = pivot[[b for b in bucket_order if b in pivot.columns]]
        
        return pivot
    
    def compute_composition_by_instrument(self) -> pd.DataFrame:
        """Compute outstanding by instrument group over time"""
        agg = self.df.groupby(
            ['record_date', 'instrument_group']
        )['par_outstanding_usd'].sum().reset_index()
        
        return agg.pivot(
            index='record_date', 
            columns='instrument_group', 
            values='par_outstanding_usd'
        ).fillna(0)
    
    def compute_wam_by_group(self) -> pd.DataFrame:
        """Compute weighted average maturity by instrument group"""
        def weighted_avg(group):
            mask = (
                group['years_to_maturity'].notna() &
                (group['years_to_maturity'] > 0) &
                (group['par_outstanding_usd'] > 0)
            )
            g = group[mask]
            if len(g) > 0:
                return np.average(g['years_to_maturity'], weights=g['par_outstanding_usd'])
            return np.nan
        
        wam = self.df.groupby(
            ['record_date', 'instrument_group']
        ).apply(weighted_avg, include_groups=False).reset_index(name='wam')
        
        return wam.pivot(
            index='record_date', 
            columns='instrument_group', 
            values='wam'
        )
    
    def compute_net_issuance_by_tenor(self, freq: str = 'M') -> pd.DataFrame:
        """
        Compute net issuance by original tenor program at specified frequency.
        
        Net issuance = gross issuance + TIPS accrual - buybacks/early
        redemptions - maturities.

        Flows are grouped by orig_tenor_program (the tenor at which each security
        was originally issued) rather than by remaining maturity_bucket.  Using
        remaining maturity would mis-classify reopened securities — a 10y note
        reopened 5 years after original issue still belongs in the '10y' program.
        """
        df = self.df.copy()
        tenor_order = get_tenor_order()

        if freq == 'M':
            df['period'] = df['record_date'].dt.to_period('M')
        elif freq == 'Q':
            df['period'] = df['record_date'].dt.to_period('Q')
        else:
            df['period'] = df['record_date'].dt.to_period('Y')

        flows = df.groupby(['period', 'orig_tenor_program']).agg({
            'issued_flow_usd': 'sum',
            'redeemed_flow_usd': 'sum',
            'infl_adj_usd': 'sum',
            'matured_usd': 'sum',
        }).reset_index()

        flows['net_issuance'] = (
            flows['issued_flow_usd'] +
            flows['infl_adj_usd'] -
            flows['redeemed_flow_usd'] -
            flows['matured_usd']
        )

        pivot = flows.pivot(
            index='period',
            columns='orig_tenor_program',
            values='net_issuance'
        ).fillna(0)

        pivot = pivot[[b for b in tenor_order if b in pivot.columns]]
        pivot.index = pivot.index.to_timestamp()

        return pivot
    
    def compute_maturity_wall(
        self, 
        horizon_years: int = 5,
        date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """Compute maturity wall (rollover profile) by quarter"""
        snapshot = self.get_snapshot(date)
        record_date = snapshot['record_date'].iloc[0]
        
        # Add maturity quarter
        snapshot['maturity_quarter'] = snapshot['maturity_date'].dt.to_period('Q')
        
        # Aggregate by quarter and instrument
        wall = snapshot.groupby(
            ['maturity_quarter', 'instrument_group']
        )['par_outstanding_usd'].sum().reset_index()
        
        # Filter to horizon
        horizon_end = (record_date + pd.DateOffset(years=horizon_years)).to_period('Q')
        wall = wall[wall['maturity_quarter'] <= horizon_end]
        
        pivot = wall.pivot(
            index='maturity_quarter',
            columns='instrument_group',
            values='par_outstanding_usd'
        ).fillna(0)
        
        pivot.index = pivot.index.to_timestamp()
        return pivot
    
    def compute_short_long_rotation(self) -> pd.DataFrame:
        """Compute short vs long share time series for rotation chart"""
        metrics = self.compute_metrics_timeseries()
        return metrics[['record_date', 'short_share_pct', 'long_share_pct']].copy()
    
    def compute_tenor_maturity_heatmap(
        self, 
        date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """Compute original tenor vs remaining maturity heatmap"""
        snapshot = self.get_snapshot(date)
        tenor_order = get_tenor_order()
        bucket_order = get_bucket_order()
        
        heatmap = snapshot.groupby(
            ['orig_tenor_program', 'maturity_bucket']
        )['par_outstanding_usd'].sum().reset_index()
        
        pivot = heatmap.pivot(
            index='orig_tenor_program',
            columns='maturity_bucket',
            values='par_outstanding_usd'
        ).fillna(0)
        
        # Reorder
        pivot = pivot.reindex([t for t in tenor_order if t in pivot.index])
        pivot = pivot[[b for b in bucket_order if b in pivot.columns]]
        
        return pivot
    
    def compute_bill_share_timeseries(self) -> pd.DataFrame:
        """Compute bill share of stock and issuance over time"""
        # Stock share — reindex to monthly period timestamps so it aligns with
        # the issuance series (which is already period-based).
        stock = self.df.groupby(['record_date', 'is_bill'])['par_outstanding_usd'].sum()
        stock = stock.unstack(fill_value=0)
        stock['bill_stock_share'] = (
            stock.get(True, 0) / (stock.get(True, 0) + stock.get(False, 0)) * 100
        )
        stock.index = stock.index.to_period('M').to_timestamp()
        # If multiple record_dates fall in the same month, keep the last.
        stock = stock[~stock.index.duplicated(keep='last')]

        # Issuance share (monthly); the first snapshot has no flows
        df = self.df[self.df['issued_flow_usd'].notna()].copy()
        df['year_month'] = df['record_date'].dt.to_period('M')

        issuance = df.groupby(['year_month', 'is_bill'])['issued_flow_usd'].sum()
        issuance = issuance.unstack(fill_value=0)
        total_issuance = issuance.get(True, 0) + issuance.get(False, 0)
        issuance['bill_issuance_share'] = np.where(
            total_issuance > 0,
            issuance.get(True, 0) / total_issuance * 100,
            np.nan
        )
        issuance.index = issuance.index.to_timestamp()

        return pd.DataFrame({
            'bill_stock_share': stock['bill_stock_share'],
            'bill_issuance_share': issuance['bill_issuance_share'],
        })
    
    def compute_maturity_percentiles(self) -> pd.DataFrame:
        """
        Compute par-weighted maturity percentiles (p25, p50, p75) over time.
        Used for Chart W to show the distribution spread around the WAM.
        """
        dates = sorted(self.df['record_date'].unique())
        results = []

        for date in dates:
            snapshot = self.get_snapshot(date)
            snap = snapshot[
                snapshot['years_to_maturity'].notna() &
                (snapshot['years_to_maturity'] > 0) &
                (snapshot['par_outstanding_usd'] > 0)
            ].sort_values('years_to_maturity').copy()

            total = snap['par_outstanding_usd'].sum()
            if total == 0:
                continue

            snap['cum_pct'] = snap['par_outstanding_usd'].cumsum() / total * 100

            def _interp(target: float) -> float:
                below = snap[snap['cum_pct'] <= target]
                above = snap[snap['cum_pct'] > target]
                if len(below) == 0:
                    return above.iloc[0]['years_to_maturity']
                if len(above) == 0:
                    return below.iloc[-1]['years_to_maturity']
                x1, y1 = below.iloc[-1]['years_to_maturity'], below.iloc[-1]['cum_pct']
                x2, y2 = above.iloc[0]['years_to_maturity'], above.iloc[0]['cum_pct']
                if y2 == y1:
                    return (x1 + x2) / 2
                return x1 + (target - y1) * (x2 - x1) / (y2 - y1)

            results.append({
                'record_date': date,
                'p25_years': _interp(25),
                'p50_years': _interp(50),
                'p75_years': _interp(75),
            })

        return pd.DataFrame(results)

    def compute_reconciliation(self) -> pd.DataFrame:
        """
        Monthly stock-flow reconciliation of total marketable debt.
        
        delta_stock (change in total par outstanding) is compared with net
        issuance built up from the flows; residual = delta_stock - net.
        Bills issued and matured inside the same month never appear in a
        month-end snapshot. They are missing from gross issuance and from
        maturities alike, so they cannot create a residual.
        """
        df = self.df
        monthly = df.groupby('record_date').agg(
            total_outstanding=('par_outstanding_usd', 'sum'),
            gross_issued=('issued_flow_usd', 'sum'),
            infl_adj=('infl_adj_usd', 'sum'),
            redeemed=('redeemed_flow_usd', 'sum'),
            matured=('matured_usd', 'sum'),
        )
        monthly['net_issuance'] = (
            monthly['gross_issued'] + monthly['infl_adj']
            - monthly['redeemed'] - monthly['matured']
        )
        monthly['delta_stock'] = monthly['total_outstanding'].diff()
        monthly['residual'] = monthly['delta_stock'] - monthly['net_issuance']
        # No flows exist for the first snapshot
        monthly = monthly.iloc[1:]
        return monthly
    
    def compute_bridge_diagnostics(self) -> Dict[str, pd.DataFrame]:
        """Compute bridge error diagnostics"""
        df = self.df[self.df['bridge_error_usd'].notna()].copy()
        df['bridge_error_abs'] = df['bridge_error_usd'].abs()
        
        # Summary by date
        by_date = df.groupby('record_date').agg({
            'bridge_error_usd': ['sum', 'mean', 'std'],
            'bridge_error_abs': 'sum',
            'par_outstanding_usd': 'sum',
        })
        by_date.columns = ['error_sum', 'error_mean', 'error_std', 'error_abs', 'total_par']
        by_date['error_pct'] = by_date['error_sum'] / by_date['total_par'] * 100
        
        # Large errors
        large_errors = df[df['bridge_error_usd'].abs() > 1e9].copy()
        
        return {
            'summary': by_date,
            'large_errors': large_errors,
        }


class TGAAnalytics:
    """Compute TGA coverage analytics"""
    
    def __init__(self, mspd_df: pd.DataFrame, tga_df: pd.DataFrame):
        self.mspd_df = mspd_df.copy()
        self.tga_df = tga_df.copy()
    
    def compute_coverage(
        self, 
        horizons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute maturities-to-TGA ratios for various horizons.
        
        maturities_to_tga_{h}d_pct = (par maturing within h days) / TGA balance * 100
        
        Above 100% the maturing debt exceeds the cash in the TGA, so it must
        be rolled rather than paid down from cash. (This is the inverse of a
        coverage ratio.)
        """
        if horizons is None:
            horizons = CONFIG.tga_horizons
        
        print("Computing maturities-to-TGA ratios...")
        
        # Find overlapping date range
        mspd_min = self.mspd_df['record_date'].min()
        mspd_max = self.mspd_df['record_date'].max()
        tga_min = self.tga_df['record_date'].min()
        tga_max = self.tga_df['record_date'].max()
        
        overlap_start = max(mspd_min, tga_min)
        overlap_end = min(mspd_max, tga_max)
        
        print(f"  Overlap period: {overlap_start.date()} to {overlap_end.date()}")
        
        if overlap_start > overlap_end:
            raise ValueError("No overlapping dates between MSPD and TGA data")
        
        # Filter MSPD to overlap
        mspd = self.mspd_df[
            (self.mspd_df['record_date'] >= overlap_start) &
            (self.mspd_df['record_date'] <= overlap_end) &
            self.mspd_df['is_outstanding']
        ]
        
        # Get unique MSPD dates
        mspd_dates = sorted(mspd['record_date'].unique())
        
        # Match TGA to MSPD dates (backward-looking)
        dates_df = pd.DataFrame({'record_date': mspd_dates})
        merged = pd.merge_asof(
            dates_df.sort_values('record_date'),
            self.tga_df.sort_values('record_date'),
            on='record_date',
            direction='backward',
            tolerance=pd.Timedelta(days=10)
        )
        
        # Build results
        results = []
        valid_dates = merged[merged['tga_balance_usd'].notna()]['record_date'].tolist()
        tga_lookup = dict(zip(merged['record_date'], merged['tga_balance_usd']))
        
        for i, current_date in enumerate(valid_dates):
            if i % 5 == 0:
                print(f"  Processing {i+1}/{len(valid_dates)}: {current_date.date()}")
            
            tga_balance = tga_lookup.get(current_date, np.nan)
            if pd.isna(tga_balance) or tga_balance <= 0:
                continue
            
            outstanding = mspd[mspd['record_date'] == current_date]
            if len(outstanding) == 0:
                continue
            
            row = {
                'record_date': current_date,
                'tga_balance_usd': tga_balance,
            }
            
            for horizon in horizons:
                horizon_date = current_date + pd.Timedelta(days=horizon)
                
                maturing = outstanding[
                    (outstanding['maturity_date'] > current_date) &
                    (outstanding['maturity_date'] <= horizon_date)
                ]
                
                maturing_par = maturing['par_outstanding_usd'].sum()
                ratio = (maturing_par / tga_balance * 100) if tga_balance > 0 else np.nan
                
                row[f'maturing_{horizon}d_usd'] = maturing_par
                row[f'maturities_to_tga_{horizon}d_pct'] = ratio
            
            results.append(row)
        
        coverage_df = pd.DataFrame(results)
        print(f"  Computed ratios for {len(coverage_df)} dates")
        
        return coverage_df
    
    def compute_coverage_by_instrument(
        self, 
        horizon_days: int = 90
    ) -> pd.DataFrame:
        """Compute maturities-to-TGA breakdown by instrument group"""
        print(f"Computing maturities-to-TGA by instrument ({horizon_days}d horizon)...")
        
        # Use subset of dates for performance
        mspd_dates = sorted(self.mspd_df['record_date'].unique())[::3]
        
        results = []
        for current_date in mspd_dates:
            tga_row = self.tga_df[self.tga_df['record_date'] <= current_date].tail(1)
            if len(tga_row) == 0:
                continue
            
            tga_balance = tga_row.iloc[0]['tga_balance_usd']
            if pd.isna(tga_balance) or tga_balance <= 0:
                continue
            
            outstanding = self.mspd_df[self.mspd_df['record_date'] == current_date]
            horizon_date = current_date + pd.Timedelta(days=horizon_days)
            
            maturing = outstanding[
                (outstanding['maturity_date'] > current_date) &
                (outstanding['maturity_date'] <= horizon_date)
            ]
            
            by_instrument = maturing.groupby('instrument_group')['par_outstanding_usd'].sum()
            
            for instrument, par in by_instrument.items():
                results.append({
                    'record_date': current_date,
                    'instrument_group': instrument,
                    'maturing_par_usd': par,
                    'maturities_to_tga_pct': par / tga_balance * 100,
                })
        
        return pd.DataFrame(results)


def compute_refinancing_halflife(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute portfolio refinancing half-life over time.
    
    Half-life = time for 50% of current portfolio to mature
    """
    print("Computing refinancing half-life...")
    
    dates = sorted(df['record_date'].unique())
    results = []
    
    for date in dates:
        snapshot = df[df['record_date'] == date]
        # Same filter as WAM: drop missing/non-positive maturities (this also
        # drops the maturity exit rows, which have par 0 and negative years)
        snapshot = snapshot[
            snapshot['years_to_maturity'].notna() &
            (snapshot['years_to_maturity'] > 0) &
            (snapshot['par_outstanding_usd'] > 0)
        ].sort_values('years_to_maturity').copy()
        
        total_par = snapshot['par_outstanding_usd'].sum()
        if total_par == 0:
            continue
        
        # Cumulative sum
        snapshot['cum_par'] = snapshot['par_outstanding_usd'].cumsum()
        snapshot['cum_pct'] = snapshot['cum_par'] / total_par * 100
        
        # Find half-life (interpolate)
        below_50 = snapshot[snapshot['cum_pct'] <= 50]
        above_50 = snapshot[snapshot['cum_pct'] > 50]
        
        if len(below_50) > 0 and len(above_50) > 0:
            # Linear interpolation
            x1 = below_50.iloc[-1]['years_to_maturity']
            y1 = below_50.iloc[-1]['cum_pct']
            x2 = above_50.iloc[0]['years_to_maturity']
            y2 = above_50.iloc[0]['cum_pct']
            
            if y2 != y1:
                half_life = x1 + (50 - y1) * (x2 - x1) / (y2 - y1)
            else:
                half_life = (x1 + x2) / 2
        elif len(above_50) > 0:
            half_life = above_50.iloc[0]['years_to_maturity']
        else:
            half_life = snapshot['years_to_maturity'].max()
        
        results.append({
            'record_date': date,
            'half_life_years': half_life,
        })
    
    return pd.DataFrame(results)


def compute_issuance_wam(df: pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
    """
    Compute weighted average remaining maturity of NEW issuance by period.
    
    Uses years to maturity at month-end weighted by gross issuance, the same
    measure as the stock WAM, so the two can be compared directly. (Original
    tenor would overstate reopenings, which are sold with less than the
    original term remaining.) This shows whether Treasury is issuing shorter
    or longer than the existing stock.
    """
    print("Computing issuance WAM...")
    df = df.copy()
    
    if freq == 'M':
        df['period'] = df['record_date'].dt.to_period('M')
    else:
        df['period'] = df['record_date'].dt.to_period('Q')
    
    # Only include securities with positive issuance this period
    issuance = df[df['issued_flow_usd'] > 0].copy()
    
    def calc_wam(group):
        mask = (
            group['years_to_maturity'].notna() &
            (group['years_to_maturity'] > 0) &
            (group['issued_flow_usd'] > 0)
        )
        g = group[mask]
        if len(g) > 0:
            return np.average(g['years_to_maturity'], weights=g['issued_flow_usd'])
        return np.nan
    
    wam = issuance.groupby('period').apply(calc_wam, include_groups=False).reset_index(name='issuance_wam')
    wam['period'] = wam['period'].dt.to_timestamp()

    return wam


def compute_rolling_rollover_rate(
    df: pd.DataFrame, window_months: int = 12
) -> pd.DataFrame:
    """
    Compute trailing N-month gross rollover rate as % of total outstanding.

    Rollover rate = trailing-window gross issuance / total outstanding × 100.
    A value of 100 % means Treasury must refinance the entire portfolio within
    the window.  The first snapshot (no flows) is dropped and the rate is
    only reported once a full window of months is available.
    """
    print("Computing rolling rollover rate...")
    d = df[df['issued_flow_usd'].notna()].copy()
    d['period'] = d['record_date'].dt.to_period('M')

    monthly = d.groupby('period').agg(
        gross_issued=('issued_flow_usd', 'sum'),
        total_outstanding=('par_outstanding_usd', 'sum'),
    ).reset_index()
    monthly = monthly.sort_values('period')
    monthly['period_ts'] = monthly['period'].dt.to_timestamp()

    monthly['rolling_issued'] = (
        monthly['gross_issued'].rolling(window_months, min_periods=window_months).sum()
    )
    monthly['rollover_rate_pct'] = np.where(
        monthly['total_outstanding'] > 0,
        monthly['rolling_issued'] / monthly['total_outstanding'] * 100,
        np.nan,
    )

    return monthly


if __name__ == "__main__":
    # Quick test
    from data_ingest import load_processed_data
    
    df = load_processed_data()
    analytics = PortfolioAnalytics(df)
    
    metrics = analytics.compute_portfolio_metrics()
    print(f"\nLatest Portfolio Metrics:")
    print(f"  Total Outstanding: ${metrics.total_outstanding/1e12:.2f}T")
    print(f"  WAM: {metrics.weighted_avg_maturity:.2f} years")
    print(f"  Short Share: {metrics.short_share:.1f}%")
    print(f"  Long Share: {metrics.long_share:.1f}%")
