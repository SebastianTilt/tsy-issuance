"""
Treasury Issuance Analysis - Charting Module
Generate static charts for Treasury debt analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from config import CONFIG, get_bucket_order, get_tenor_order, format_currency

# Caveat for charts built on gross issuance, which misses intramonth bills
BILL_ISSUANCE_CAVEAT = (
    "Month-end MSPD snapshots miss bills issued and matured within the same month "
    "(4/6/8-week bills, CMBs),\nso bill gross issuance is understated."
)
from analytics import (
    PortfolioAnalytics,
    TGAAnalytics,
    compute_refinancing_halflife,
    compute_issuance_wam,
    compute_rolling_rollover_rate,
)


class ChartGenerator:
    """Generate all analysis charts"""
    
    def __init__(self, df: pd.DataFrame, output_dir: Optional[Path] = None):
        self.df = df.copy()
        self.output_dir = output_dir or CONFIG.charts_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analytics = PortfolioAnalytics(df)
        
        # Setup style
        CONFIG.setup_matplotlib_style()
    
    def _save_figure(self, name: str, fig: plt.Figure):
        """Save figure and close"""
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=CONFIG.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path.name}")
    
    def _format_yaxis_currency(self, ax, scale: str = 'T'):
        """Format y-axis as currency"""
        sign = lambda x: '-' if x < 0 else ''
        if scale == 'T':
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'{sign(x)}${abs(x)/1e12:.1f}T')
            )
        elif scale == 'B':
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'{sign(x)}${abs(x)/1e9:,.0f}B')
            )
        else:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'{sign(x)}${abs(x)/1e6:,.0f}M')
            )
    
    # =========================================================================
    # COMPOSITION CHARTS (A, B)
    # =========================================================================
    
    def chart_instrument_composition(self):
        """Chart A: Outstanding by instrument group"""
        print("Creating Chart A: Instrument composition...")
        
        pivot = self.analytics.compute_composition_by_instrument()
        
        # A1: Absolute
        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        pivot.plot(kind='area', stacked=True, ax=ax, alpha=0.8)
        ax.set_title('Outstanding Treasury Securities by Instrument Group', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Outstanding Par (USD)')
        ax.legend(title='Instrument Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        self._format_yaxis_currency(ax, 'T')
        plt.tight_layout()
        self._save_figure('A1_instrument_group_absolute', fig)
        
        # A2: Percentage
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        pivot_pct.plot(kind='area', stacked=True, ax=ax, alpha=0.8)
        ax.set_title('Outstanding Treasury Securities by Instrument Group (% of Total)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Percentage of Total')
        ax.legend(title='Instrument Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        plt.tight_layout()
        self._save_figure('A2_instrument_group_percentage', fig)
    
    def chart_maturity_bucket_composition(self):
        """Chart B: Outstanding by maturity bucket"""
        print("Creating Chart B: Maturity bucket composition...")
        
        pivot = self.analytics.compute_composition_by_bucket()
        
        # B1: Absolute
        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        pivot.plot(kind='area', stacked=True, ax=ax, alpha=0.8)
        ax.set_title('Outstanding Treasury Securities by Maturity Bucket', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Outstanding Par (USD)')
        ax.legend(title='Maturity Bucket', bbox_to_anchor=(1.05, 1), loc='upper left')
        self._format_yaxis_currency(ax, 'T')
        plt.tight_layout()
        self._save_figure('B1_maturity_bucket_absolute', fig)
        
        # B2: Percentage
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        pivot_pct.plot(kind='area', stacked=True, ax=ax, alpha=0.8)
        ax.set_title('Outstanding Treasury Securities by Maturity Bucket (% of Total)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Percentage of Total')
        ax.legend(title='Maturity Bucket', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        plt.tight_layout()
        self._save_figure('B2_maturity_bucket_percentage', fig)
    
    # =========================================================================
    # WAM CHARTS (C)
    # =========================================================================
    
    def chart_weighted_average_maturity(self):
        """Chart C: Weighted Average Maturity"""
        print("Creating Chart C: Weighted Average Maturity...")
        
        metrics = self.analytics.compute_metrics_timeseries()
        wam_by_group = self.analytics.compute_wam_by_group()
        
        # C1: Total WAM
        fig, ax = plt.subplots(figsize=CONFIG.figsize_medium)
        ax.plot(metrics['record_date'], metrics['wam_years'], 
               linewidth=2.5, color='navy')
        ax.set_title('Weighted Average Maturity of Outstanding Treasury Portfolio', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('WAM (Years)')
        ax.grid(True, alpha=0.3)
        
        # Add trend annotation
        start_wam = metrics['wam_years'].iloc[0]
        end_wam = metrics['wam_years'].iloc[-1]
        change = end_wam - start_wam
        ax.annotate(
            f'Change: {change:+.2f}y',
            xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8)
        )
        
        plt.tight_layout()
        self._save_figure('C1_wam_total', fig)
        
        # C2: WAM by instrument group
        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        for col in wam_by_group.columns:
            ax.plot(wam_by_group.index, wam_by_group[col], 
                   linewidth=2, label=col, alpha=0.8)
        ax.set_title('Weighted Average Maturity by Instrument Group', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('WAM (Years)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_figure('C2_wam_by_group', fig)
    
    # =========================================================================
    # ISSUANCE CHARTS (D)
    # =========================================================================
    
    def chart_net_issuance(self):
        """Chart D: Net issuance by original tenor program"""
        print("Creating Chart D: Net issuance...")
        
        pivot = self.analytics.compute_net_issuance_by_tenor(freq='M')
        
        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        pivot.plot(kind='bar', stacked=True, ax=ax, width=0.8, alpha=0.8)
        ax.set_title('Net Issuance by Original Tenor (Monthly)\n'
                    'gross issuance + TIPS accrual − buybacks − maturities', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Net Issuance (USD)')
        ax.legend(title='Original Tenor', bbox_to_anchor=(1.05, 1), loc='upper left')
        self._format_yaxis_currency(ax, 'B')
        ax.axhline(0, color='black', linewidth=0.8)
        
        # Reduce x-tick clutter
        n_ticks = len(pivot)
        step = max(1, n_ticks // 20)
        ax.set_xticks(range(0, n_ticks, step))
        ax.set_xticklabels(
            [pivot.index[i].strftime('%Y-%m') for i in range(0, n_ticks, step)],
            rotation=45, ha='right'
        )
        
        plt.tight_layout()
        self._save_figure('D_net_issuance_by_tenor', fig)
    
    # =========================================================================
    # BILL SHARE (E)
    # =========================================================================
    
    def chart_bill_share(self):
        """Chart E: Bill share of stock and issuance"""
        print("Creating Chart E: Bill share...")
        
        bill_data = self.analytics.compute_bill_share_timeseries()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # E1: Stock share
        ax1.plot(bill_data.index, bill_data['bill_stock_share'], 
                linewidth=2, color='darkred')
        ax1.set_title('Bill Share of Outstanding Treasury Securities', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('Bill Share (%)')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        # E2: Issuance share
        ax2.plot(bill_data.index, bill_data['bill_issuance_share'], 
                linewidth=2, color='darkblue')
        ax2.set_title('Bill Share of Gross Issuance', 
                     fontsize=12, fontweight='bold')
        ax2.text(0.01, 0.02, BILL_ISSUANCE_CAVEAT, transform=ax2.transAxes,
                 fontsize=8, color='#555555', va='bottom', fontstyle='italic')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Bill Share (%)')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        plt.tight_layout()
        self._save_figure('E_bill_share', fig)
    
    # =========================================================================
    # MATURITY WALL (F)
    # =========================================================================
    
    def chart_maturity_wall(self):
        """Chart F: Maturity wall / rollover profile"""
        print("Creating Chart F: Maturity wall...")
        
        pivot = self.analytics.compute_maturity_wall(horizon_years=5)
        latest_date = self.df['record_date'].max()
        
        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        pivot.plot(kind='bar', stacked=True, ax=ax, width=0.8, alpha=0.8)
        ax.set_title(f'Treasury Maturity Wall (as of {latest_date.strftime("%Y-%m-%d")})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Maturity Quarter')
        ax.set_ylabel('Par Maturing (USD)')
        ax.legend(title='Instrument Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        self._format_yaxis_currency(ax, 'B')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_figure('F_maturity_wall', fig)
    
    # =========================================================================
    # HEATMAP (G)
    # =========================================================================
    
    def chart_tenor_maturity_heatmap(self):
        """Chart G: Original tenor vs remaining maturity heatmap"""
        print("Creating Chart G: Tenor-maturity heatmap...")
        
        pivot = self.analytics.compute_tenor_maturity_heatmap()
        latest_date = self.df['record_date'].max()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pivot.values / 1e9, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        
        ax.set_xlabel('Remaining Maturity Bucket', fontsize=12)
        ax.set_ylabel('Original Tenor Program', fontsize=12)
        ax.set_title(
            f'Treasury Portfolio: Original Tenor vs Remaining Maturity\n'
            f'(as of {latest_date.strftime("%Y-%m-%d")})', 
            fontsize=14, fontweight='bold'
        )
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Outstanding Par ($B)', rotation=270, labelpad=20)
        
        # Text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if val > 0:
                    ax.text(j, i, f'{val/1e9:.0f}',
                           ha="center", va="center", 
                           color="white" if val > pivot.values.max()/2 else "black",
                           fontsize=8)
        
        plt.tight_layout()
        self._save_figure('G_tenor_maturity_heatmap', fig)
    
    # =========================================================================
    # SHORT VS LONG ROTATION (N)
    # =========================================================================
    
    def chart_short_long_rotation(self):
        """Chart N: Short vs Long rotation diagram"""
        print("Creating Chart N: Short vs Long rotation...")
        
        rotation = self.analytics.compute_short_long_rotation()
        
        n = len(rotation)
        if n < 2:
            print("  Skipping - need at least 2 data points")
            return
        
        t_norm = np.linspace(0, 1, n)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Path line
        ax.plot(
            rotation['short_share_pct'],
            rotation['long_share_pct'],
            color='lightgray',
            linewidth=1.3,
            zorder=1,
        )
        
        # Colored scatter
        sc = ax.scatter(
            rotation['short_share_pct'],
            rotation['long_share_pct'],
            c=t_norm,
            cmap='viridis',
            s=40,
            alpha=0.9,
            edgecolor='black',
            linewidth=0.4,
            zorder=2,
        )
        
        # Start/end markers
        start = rotation.iloc[0]
        end = rotation.iloc[-1]
        
        ax.scatter(
            [start['short_share_pct']], [start['long_share_pct']],
            s=90, facecolors='white', edgecolors='black',
            linewidth=1.3, zorder=3, label='Start'
        )
        ax.scatter(
            [end['short_share_pct']], [end['long_share_pct']],
            s=90, facecolors='gold', edgecolors='black',
            linewidth=1.3, zorder=3, label='Latest'
        )
        
        # Annotations
        ax.text(
            start['short_share_pct'] - 0.2, start['long_share_pct'] - 0.2,
            rotation['record_date'].iloc[0].strftime('%Y-%m'),
            fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8)
        )
        ax.text(
            end['short_share_pct'] + 0.2, end['long_share_pct'] + 0.2,
            rotation['record_date'].iloc[-1].strftime('%Y-%m'),
            fontsize=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8)
        )
        
        ax.set_title(
            'Portfolio Rotation: Short (≤3y) vs Long (≥7y) Share',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('Short Share (≤3y) % of Total')
        ax.set_ylabel('Long Share (≥7y) % of Total')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='lower left', fontsize=8)
        
        # Colorbar with date labels
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Time (earlier → later)', rotation=270, labelpad=15)
        
        tick_positions = np.linspace(0, n - 1, 5).astype(int)
        cbar.set_ticks(tick_positions / (n - 1))
        cbar.set_ticklabels(
            rotation['record_date'].iloc[tick_positions].dt.strftime('%Y-%m')
        )
        
        plt.tight_layout()
        self._save_figure('N_short_vs_long_rotation', fig)
    
    # =========================================================================
    # REFINANCING HALF-LIFE (J)
    # =========================================================================
    
    def chart_refinancing_halflife(self):
        """Chart J: Portfolio refinancing half-life"""
        print("Creating Chart J: Refinancing half-life...")
        
        halflife = compute_refinancing_halflife(self.df)
        
        fig, ax = plt.subplots(figsize=CONFIG.figsize_medium)
        ax.plot(halflife['record_date'], halflife['half_life_years'], 
               linewidth=2, color='darkgreen')
        ax.set_title('Portfolio Refinancing Half-Life', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Half-Life (Years)')
        ax.grid(True, alpha=0.3)
        
        # Annotation
        latest = halflife.iloc[-1]
        ax.annotate(
            f'Current: {latest["half_life_years"]:.2f}y',
            xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8)
        )
        
        plt.tight_layout()
        self._save_figure('J_refinancing_halflife', fig)
    
    # =========================================================================
    # ISSUANCE WAM (U)
    # =========================================================================
    
    def chart_issuance_wam(self):
        """Chart U: WAM of new issuance vs WAM of the outstanding stock"""
        print("Creating Chart U: Issuance WAM vs stock WAM...")
        
        issuance_wam = compute_issuance_wam(self.df, freq='M')
        metrics = self.analytics.compute_metrics_timeseries()
        
        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        
        ax.plot(issuance_wam['period'], issuance_wam['issuance_wam'], 
               linewidth=1, color='blue', alpha=0.35, label='New issuance WAM (monthly)')
        ax.plot(issuance_wam['period'], issuance_wam['issuance_wam'].rolling(12).mean(),
               linewidth=2, color='blue', label='New issuance WAM (12m avg)')
        ax.plot(metrics['record_date'], metrics['wam_years'], 
               linewidth=2, color='red', label='Outstanding stock WAM')
        
        ax.set_title('Remaining Maturity: New Issuance vs Outstanding Stock\n'
                    '(par-weighted years to maturity at month-end)', 
                    fontsize=14, fontweight='bold')
        fig.text(0.01, 0.005, BILL_ISSUANCE_CAVEAT +
                 ' Issuance WAM is biased upward as a result.',
                 fontsize=8, color='#555555', fontstyle='italic')
        ax.set_xlabel('Date')
        ax.set_ylabel('Years')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        self._save_figure('U_issuance_wam_vs_stock_wam', fig)
    
    # =========================================================================
    # BRIDGE DIAGNOSTICS (O)
    # =========================================================================
    
    def chart_bridge_diagnostics(self):
        """Chart O: Stock-flow bridge error diagnostics"""
        print("Creating Chart O: Bridge diagnostics...")
        
        diagnostics = self.analytics.compute_bridge_diagnostics()
        summary = diagnostics['summary']
        
        if len(summary) == 0:
            print("  No bridge data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # O1: Net and absolute error over time
        ax1.bar(summary.index, summary['error_sum'] / 1e9, width=25,
               alpha=0.7, color='steelblue', label='Net error (sum)')
        ax1.plot(summary.index, summary['error_abs'] / 1e9,
                linewidth=1, color='darkorange', label='Absolute error (sum of |error|)')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.set_title('Stock-Flow Bridge Error by Date\n'
                     'Δ par − (issued + TIPS accrual − buybacks − maturities)', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('Bridge Error ($B)')
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # O2: Error as % of total
        ax2.plot(summary.index, summary['error_pct'], 
                linewidth=2, color='darkred')
        ax2.set_title('Bridge Error as % of Total Outstanding', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Error (%)')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure('O_bridge_diagnostics', fig)
    
    # =========================================================================
    # GROSS ISSUANCE vs REDEMPTIONS (P)
    # =========================================================================

    def chart_gross_issuance_redemptions(self):
        """
        Chart P: Quarterly butterfly chart — gross issuance (above zero) and
        gross outflows (maturities + buybacks, below zero), stacked by
        instrument group, with a separate net issuance strip below.

        Quarterly aggregation keeps ~60 bars over the full history so each bar
        is clearly visible.  Issuance and redemptions use the same colour per
        instrument but different opacity so the two halves read as a mirror.
        """
        print("Creating Chart P: Gross issuance vs maturities and redemptions...")

        df = self.df.copy()
        df['period'] = df['record_date'].dt.to_period('Q')
        # Gross outflow = maturities + buybacks/early redemptions
        df['outflow_usd'] = df['matured_usd'] + df['redeemed_flow_usd']

        # ── per-instrument quarterly totals ──────────────────────────────────
        issued_raw = (
            df.groupby(['period', 'instrument_group'])['issued_flow_usd']
            .sum().reset_index()
        )
        issued_pivot = issued_raw.pivot(
            index='period', columns='instrument_group', values='issued_flow_usd'
        ).fillna(0)
        issued_pivot.index = issued_pivot.index.to_timestamp()

        redeemed_raw = (
            df.groupby(['period', 'instrument_group'])['outflow_usd']
            .sum().reset_index()
        )
        redeemed_pivot = redeemed_raw.pivot(
            index='period', columns='instrument_group', values='outflow_usd'
        ).fillna(0)
        redeemed_pivot.index = redeemed_pivot.index.to_timestamp()

        # net per quarter
        net_q = df.groupby('period').agg(
            net=('net_issuance_usd', 'sum'),
        ).reset_index()
        net_q['period_ts'] = net_q['period'].dt.to_timestamp()

        # ── consistent colour map across both halves ──────────────────────────
        all_instruments = list(
            issued_pivot.columns.union(redeemed_pivot.columns)
        )
        palette = plt.cm.tab10(np.linspace(0, 1, min(len(all_instruments), 10)))
        color_map = {inst: palette[i % 10] for i, inst in enumerate(all_instruments)}

        # ── layout: tall main panel + slim net strip ──────────────────────────
        fig, (ax_main, ax_net) = plt.subplots(
            2, 1,
            figsize=(16, 11),
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08},
            sharex=True,
        )

        bar_width = 65  # days — fills ~¾ of each quarter slot

        # ── above zero: gross issuance stacked by instrument ─────────────────
        bottoms = np.zeros(len(issued_pivot))
        for inst in all_instruments:
            if inst not in issued_pivot.columns:
                continue
            vals = issued_pivot[inst].values
            ax_main.bar(
                issued_pivot.index, vals, bottom=bottoms,
                width=bar_width, color=color_map[inst],
                alpha=0.88, label=inst,
            )
            bottoms += vals

        # ── below zero: gross redemptions stacked, lighter shade ─────────────
        bottoms_neg = np.zeros(len(redeemed_pivot))
        for inst in all_instruments:
            if inst not in redeemed_pivot.columns:
                continue
            vals = redeemed_pivot[inst].values
            ax_main.bar(
                redeemed_pivot.index, -vals, bottom=-bottoms_neg,
                width=bar_width, color=color_map[inst],
                alpha=0.38,  # lighter: same hue, reads as mirror
            )
            bottoms_neg += vals

        ax_main.axhline(0, color='#333333', linewidth=1.2)
        ax_main.set_title(
            'Gross Treasury Issuance vs Maturities and Redemptions by Instrument  (Quarterly)',
            fontsize=13, fontweight='bold', pad=10,
        )
        ax_main.set_ylabel('Amount', fontsize=11)
        self._format_yaxis_currency(ax_main, 'B')
        ax_main.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)

        # Labels inside the chart to mark each half
        ax_main.text(0.012, 0.97, 'GROSS ISSUANCE',
                     transform=ax_main.transAxes,
                     fontsize=9, color='#555555', va='top', fontstyle='italic')
        ax_main.text(0.012, 0.03, 'MATURITIES + BUYBACKS',
                     transform=ax_main.transAxes,
                     fontsize=9, color='#555555', va='bottom', fontstyle='italic')

        # Legend: one entry per instrument (issuance shade only)
        ax_main.legend(
            title='Instrument Group', bbox_to_anchor=(1.01, 1),
            loc='upper left', fontsize=9, framealpha=0.9,
        )

        # ── net strip ─────────────────────────────────────────────────────────
        ax_net.plot(net_q['period_ts'], net_q['net'],
                    color='#222222', linewidth=1.4, zorder=3)
        ax_net.fill_between(
            net_q['period_ts'], net_q['net'], 0,
            where=(net_q['net'] >= 0),
            color='#2ca02c', alpha=0.55, label='Net positive',
        )
        ax_net.fill_between(
            net_q['period_ts'], net_q['net'], 0,
            where=(net_q['net'] < 0),
            color='#d62728', alpha=0.55, label='Net negative',
        )
        ax_net.axhline(0, color='#333333', linewidth=0.9)
        ax_net.set_ylabel('Net Issuance', fontsize=10)
        ax_net.set_xlabel('Date', fontsize=11)
        self._format_yaxis_currency(ax_net, 'B')
        ax_net.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax_net.spines['top'].set_visible(False)
        ax_net.spines['right'].set_visible(False)
        ax_net.legend(loc='upper left', fontsize=9, framealpha=0.9)

        ax_net.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_net.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax_net.get_xticklabels(), rotation=0, ha='center')

        # No tight_layout: the gridspec sets hspace explicitly, and
        # savefig(bbox_inches='tight') trims the margins.
        self._save_figure('P_gross_issuance_vs_outflows', fig)

    # =========================================================================
    # 12-MONTH FORWARD MATURITY CALENDAR (Q)
    # =========================================================================

    def chart_forward_maturity_calendar(self):
        """Chart Q: Monthly maturity calendar for the next 12 months"""
        print("Creating Chart Q: 12-month forward maturity calendar...")

        latest_date = self.df['record_date'].max()
        snapshot = self.analytics.get_snapshot(latest_date)

        horizon_end = latest_date + pd.DateOffset(months=12)
        maturing = snapshot[
            (snapshot['maturity_date'] > latest_date) &
            (snapshot['maturity_date'] <= horizon_end)
        ].copy()

        if maturing.empty:
            print("  No maturities in next 12 months — skipping")
            return

        maturing['maturity_month'] = maturing['maturity_date'].dt.to_period('M')
        grouped = maturing.groupby(
            ['maturity_month', 'instrument_group']
        )['par_outstanding_usd'].sum().reset_index()

        pivot = grouped.pivot(
            index='maturity_month',
            columns='instrument_group',
            values='par_outstanding_usd',
        ).fillna(0)
        pivot.index = pivot.index.to_timestamp()

        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        pivot.plot(kind='bar', stacked=True, ax=ax, alpha=0.85, width=0.7)
        ax.set_title(
            f'12-Month Forward Maturity Calendar\n'
            f'(as of {latest_date.strftime("%Y-%m-%d")})',
            fontsize=14, fontweight='bold',
        )
        ax.set_xlabel('Maturity Month')
        ax.set_ylabel('Par Maturing (USD)')
        self._format_yaxis_currency(ax, 'B')
        ax.legend(title='Instrument', bbox_to_anchor=(1.05, 1), loc='upper left')
        n = len(pivot)
        ax.set_xticks(range(n))
        ax.set_xticklabels(
            [pivot.index[i].strftime('%b %Y') for i in range(n)],
            rotation=45, ha='right',
        )
        plt.tight_layout()
        self._save_figure('Q_forward_maturity_calendar', fig)

    # =========================================================================
    # DEBT VINTAGE COHORT COMPOSITION (R)
    # =========================================================================

    def chart_vintage_cohort_composition(self):
        """Chart R: Outstanding par grouped by issuance vintage (year)"""
        print("Creating Chart R: Vintage cohort composition...")

        df = self.df.copy()
        df = df[df['issue_year'].notna()]
        df['issue_year'] = df['issue_year'].astype(int)

        agg = df.groupby(
            ['record_date', 'issue_year']
        )['par_outstanding_usd'].sum().reset_index()

        pivot = agg.pivot(
            index='record_date',
            columns='issue_year',
            values='par_outstanding_usd',
        ).fillna(0)

        n_vintages = len(pivot.columns)
        cmap = plt.cm.get_cmap('tab20', n_vintages)

        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        pivot.plot(kind='area', stacked=True, ax=ax, alpha=0.82,
                   color=[cmap(i) for i in range(n_vintages)])
        ax.set_title('Outstanding Treasury Securities by Issuance Vintage',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Outstanding Par (USD)')
        self._format_yaxis_currency(ax, 'T')

        # Thin the legend if there are many vintages
        handles, labels = ax.get_legend_handles_labels()
        step = max(1, len(labels) // 15)
        ax.legend(
            handles[::step], labels[::step],
            title='Issue Year', bbox_to_anchor=(1.05, 1),
            loc='upper left', fontsize=8,
        )
        plt.tight_layout()
        self._save_figure('R_vintage_cohort_composition', fig)

    # =========================================================================
    # MATURITY CDF (S)
    # =========================================================================

    def chart_maturity_cdf(self):
        """Chart S: Cumulative maturity distribution across selected snapshots"""
        print("Creating Chart S: Maturity CDF...")

        all_dates = sorted(self.df['record_date'].unique())
        n = len(all_dates)
        # 5 evenly-spaced historical dates + the latest
        indices = sorted(set(
            list(np.linspace(0, n - 1, min(5, n)).astype(int)) + [n - 1]
        ))
        snapshot_dates = [all_dates[i] for i in indices]

        colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_dates)))

        fig, ax = plt.subplots(figsize=CONFIG.figsize_medium)

        for date, color in zip(snapshot_dates, colors):
            snapshot = self.analytics.get_snapshot(date)
            snap = snapshot[
                snapshot['years_to_maturity'].notna() &
                (snapshot['years_to_maturity'] > 0) &
                (snapshot['par_outstanding_usd'] > 0)
            ].sort_values('years_to_maturity').copy()

            total = snap['par_outstanding_usd'].sum()
            if total == 0:
                continue

            snap['cum_pct'] = snap['par_outstanding_usd'].cumsum() / total * 100
            ax.plot(snap['years_to_maturity'], snap['cum_pct'],
                    label=date.strftime('%Y-%m'), color=color, linewidth=2)

        ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(0.3, 51, '50%', fontsize=8, color='gray')
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 100)
        ax.set_title('Cumulative Maturity Distribution (Selected Snapshots)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Years to Maturity')
        ax.set_ylabel('Cumulative % of Par Outstanding')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        ax.legend(title='As Of', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_figure('S_maturity_cdf', fig)

    # =========================================================================
    # ROLLING GROSS ROLLOVER RATE (T)
    # =========================================================================

    def chart_rolling_rollover_rate(self):
        """Chart T: Trailing 12-month gross rollover rate as % of outstanding"""
        print("Creating Chart T: Rolling rollover rate...")

        monthly = compute_rolling_rollover_rate(self.df, window_months=12)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Top panel: rolling 12-month gross issuance in $T
        ax1.fill_between(monthly['period_ts'],
                         monthly['rolling_issued'],
                         alpha=0.6, color='steelblue', label='Trailing 12-month Gross Issuance')
        ax1.plot(monthly['period_ts'], monthly['total_outstanding'],
                 color='black', linewidth=2, label='Total Outstanding')
        ax1.set_ylabel('Amount')
        ax1.set_title('Trailing 12-Month Gross Issuance vs Total Outstanding',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        self._format_yaxis_currency(ax1, 'T')

        # Bottom panel: rollover rate %
        ax2.plot(monthly['period_ts'], monthly['rollover_rate_pct'],
                 linewidth=2, color='darkred')
        ax2.axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                    label='100% (full portfolio turnover)')
        ax2.set_ylabel('Rollover Rate (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('Trailing 12-Month Gross Rollover Rate (% of Outstanding)',
                      fontsize=12, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.xticks(rotation=45)

        plt.tight_layout()
        self._save_figure('T_rolling_rollover_rate', fig)

    # =========================================================================
    # ISSUANCE SEASONALITY HEATMAP (V)
    # =========================================================================

    def chart_issuance_seasonality(self):
        """Chart V: Gross issuance heatmap by year × calendar month"""
        print("Creating Chart V: Issuance seasonality heatmap...")

        # The first snapshot has no flows (NaN); leave it and months not yet
        # published blank rather than showing them as zero issuance.
        df = self.df[self.df['issued_flow_usd'].notna()].copy()
        df['year'] = df['record_date'].dt.year
        df['month'] = df['record_date'].dt.month

        monthly = df.groupby(['year', 'month'])['issued_flow_usd'].sum().reset_index()
        pivot = monthly.pivot(
            index='year', columns='month', values='issued_flow_usd'
        ).reindex(columns=range(1, 13))
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig, ax = plt.subplots(figsize=(13, max(6, len(pivot) * 0.4 + 2)))
        im = ax.imshow(pivot.values / 1e9, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(12))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.grid(False)
        vmax = np.nanmax(pivot.values)
        for i in range(len(pivot.index)):
            for j in range(12):
                val = pivot.values[i, j]
                if pd.notna(val) and val > 0:
                    text_color = 'white' if val > vmax * 0.6 else 'black'
                    ax.text(j, i, f'{val / 1e9:.0f}',
                            ha='center', va='center',
                            fontsize=7, color=text_color)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Gross Issuance ($B)', rotation=270, labelpad=20)
        ax.set_title('Treasury Gross Issuance Seasonality (Year × Month)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        fig.text(0.01, 0.005, BILL_ISSUANCE_CAVEAT, fontsize=8,
                 color='#555555', fontstyle='italic')
        self._save_figure('V_issuance_seasonality', fig)

    # =========================================================================
    # MATURITY DISTRIBUTION PERCENTILE BAND (W)
    # =========================================================================

    def chart_maturity_percentile_band(self):
        """Chart W: WAM with par-weighted p25/p75 maturity band"""
        print("Creating Chart W: Maturity percentile band...")

        metrics = self.analytics.compute_metrics_timeseries()
        pctiles = self.analytics.compute_maturity_percentiles()

        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)

        # Shaded interquartile band
        ax.fill_between(
            pctiles['record_date'],
            pctiles['p25_years'],
            pctiles['p75_years'],
            alpha=0.25, color='steelblue', label='IQR (p25–p75)',
        )
        ax.plot(pctiles['record_date'], pctiles['p50_years'],
                linewidth=1.5, linestyle='--', color='steelblue',
                alpha=0.8, label='Median (p50)')
        ax.plot(metrics['record_date'], metrics['wam_years'],
                linewidth=2.5, color='navy', label='WAM (par-weighted mean)')

        ax.set_title('Treasury Portfolio Maturity: WAM and Distribution Band',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Years to Maturity')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        self._save_figure('W_maturity_percentile_band', fig)

    # =========================================================================
    # TIPS INFLATION ACCRUAL (X)
    # =========================================================================

    def chart_tips_inflation_accrual(self):
        """Chart X: TIPS outstanding — nominal par vs inflation accrual"""
        print("Creating Chart X: TIPS inflation accrual...")

        tips = self.df[self.df['is_tips']].copy()
        if tips.empty:
            print("  No TIPS data — skipping")
            return

        # Aggregate per date
        by_date = tips.groupby('record_date').agg(
            par_outstanding=('par_outstanding_usd', 'sum'),
            infl_accrual=('infl_adj_cum_usd', 'sum'),
        ).reset_index()

        # Original par = CPI-adjusted outstanding minus cumulative accrual
        by_date['original_par'] = (
            by_date['par_outstanding'] - by_date['infl_accrual'].clip(lower=0)
        )
        by_date['accrual'] = by_date['infl_accrual'].clip(lower=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Top panel: stacked area of original par + inflation accrual
        ax1.stackplot(
            by_date['record_date'],
            by_date['original_par'],
            by_date['accrual'],
            labels=['Original Par', 'Inflation Accrual'],
            colors=['steelblue', 'tomato'],
            alpha=0.8,
        )
        ax1.set_ylabel('Amount')
        ax1.set_title('TIPS Outstanding: Original Par vs Inflation Accrual',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        self._format_yaxis_currency(ax1, 'T')

        # Bottom panel: accrual as % of total TIPS outstanding
        by_date['accrual_pct'] = np.where(
            by_date['par_outstanding'] > 0,
            by_date['accrual'] / by_date['par_outstanding'] * 100,
            0,
        )
        ax2.plot(by_date['record_date'], by_date['accrual_pct'],
                 linewidth=2, color='darkred')
        ax2.fill_between(by_date['record_date'], by_date['accrual_pct'],
                         alpha=0.2, color='darkred')
        ax2.set_ylabel('Inflation Accrual as % of TIPS Par')
        ax2.set_xlabel('Date')
        ax2.set_title('Cumulative Inflation Accrual as % of TIPS Outstanding',
                      fontsize=12, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.xticks(rotation=45)

        plt.tight_layout()
        self._save_figure('X_tips_inflation_accrual', fig)

    # =========================================================================
    # GENERATE ALL CHARTS
    # =========================================================================

    def generate_all_charts(self):
        """Generate all standard charts"""
        print("\n" + "=" * 60)
        print("Generating All Charts")
        print("=" * 60)
        
        self.chart_instrument_composition()
        self.chart_maturity_bucket_composition()
        self.chart_weighted_average_maturity()
        self.chart_net_issuance()
        self.chart_bill_share()
        self.chart_maturity_wall()
        self.chart_tenor_maturity_heatmap()
        self.chart_short_long_rotation()
        self.chart_refinancing_halflife()
        self.chart_issuance_wam()
        self.chart_bridge_diagnostics()
        # New charts
        self.chart_gross_issuance_redemptions()
        self.chart_forward_maturity_calendar()
        self.chart_vintage_cohort_composition()
        self.chart_maturity_cdf()
        self.chart_rolling_rollover_rate()
        self.chart_issuance_seasonality()
        self.chart_maturity_percentile_band()
        self.chart_tips_inflation_accrual()
        
        print("\n" + "=" * 60)
        print(f"All charts saved to: {self.output_dir}")
        print("=" * 60)


class TGAChartGenerator:
    """Generate TGA coverage charts"""
    
    def __init__(
        self, 
        mspd_df: pd.DataFrame, 
        tga_df: pd.DataFrame,
        output_dir: Optional[Path] = None
    ):
        self.mspd_df = mspd_df.copy()
        self.tga_df = tga_df.copy()
        self.output_dir = output_dir or CONFIG.charts_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analytics = TGAAnalytics(mspd_df, tga_df)
        
        CONFIG.setup_matplotlib_style()
    
    def _save_figure(self, name: str, fig: plt.Figure):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=CONFIG.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path.name}")
    
    def chart_tga_coverage_ratios(self, coverage_df: pd.DataFrame):
        """Upcoming maturities as a % of the TGA balance, by horizon"""
        print("Creating maturities-to-TGA chart...")
        
        horizons = CONFIG.tga_horizons
        horizon_labels = {30: '30d', 60: '60d', 90: '90d', 180: '6m', 365: '1y'}
        
        fig, ax = plt.subplots(figsize=CONFIG.figsize_large)
        
        for horizon in horizons:
            col = f'maturities_to_tga_{horizon}d_pct'
            if col in coverage_df.columns:
                ax.plot(coverage_df['record_date'], coverage_df[col],
                       linewidth=2, label=horizon_labels.get(horizon, f'{horizon}d'),
                       alpha=0.8)
        
        ax.set_title('Upcoming Maturities as % of TGA Balance', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Maturing Par / TGA Balance (%)')
        ax.axhline(100, color='red', linestyle='--', linewidth=1, alpha=0.5,
                   label='100% = maturities equal TGA balance')
        ax.legend(title='Horizon', loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure('TGA_maturities_to_tga', fig)
    
    def chart_tga_vs_maturities(self, coverage_df: pd.DataFrame):
        """TGA balance vs maturing debt"""
        print("Creating TGA vs maturities chart...")
        
        horizons = [30, 90, 365]
        horizon_labels = {30: '30-Day', 90: '90-Day', 365: '1-Year'}
        
        fig, axes = plt.subplots(len(horizons), 1, 
                                 figsize=(14, 4*len(horizons)), sharex=True)
        
        for i, horizon in enumerate(horizons):
            ax = axes[i]
            mat_col = f'maturing_{horizon}d_usd'
            
            if mat_col not in coverage_df.columns:
                continue
            
            ax.plot(coverage_df['record_date'], 
                   coverage_df['tga_balance_usd']/1e12,
                   linewidth=2.5, label='TGA Balance', color='navy')
            ax.plot(coverage_df['record_date'], 
                   coverage_df[mat_col]/1e12,
                   linewidth=2.5, 
                   label=f'{horizon_labels[horizon]} Maturities', 
                   color='darkred')
            
            # Fill between
            ax.fill_between(
                coverage_df['record_date'],
                coverage_df['tga_balance_usd']/1e12,
                coverage_df[mat_col]/1e12,
                where=coverage_df['tga_balance_usd'] > coverage_df[mat_col],
                alpha=0.3, color='green', label='Surplus'
            )
            ax.fill_between(
                coverage_df['record_date'],
                coverage_df['tga_balance_usd']/1e12,
                coverage_df[mat_col]/1e12,
                where=coverage_df['tga_balance_usd'] <= coverage_df[mat_col],
                alpha=0.3, color='red', label='Deficit'
            )
            
            ax.set_ylabel('Amount ($T)')
            ax.set_title(f'{horizon_labels[horizon]} Horizon', fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        plt.suptitle('TGA Balance vs Debt Maturities by Horizon', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        self._save_figure('TGA_vs_maturities', fig)
    
    def generate_all_tga_charts(self):
        """Generate all TGA-related charts"""
        print("\nGenerating TGA Charts...")
        
        coverage_df = self.analytics.compute_coverage()
        
        if len(coverage_df) == 0:
            print("  No TGA coverage data available")
            return
        
        # Save coverage data
        coverage_df.to_parquet(
            CONFIG.data_dir / "tga_coverage.parquet", 
            index=False
        )
        
        self.chart_tga_coverage_ratios(coverage_df)
        self.chart_tga_vs_maturities(coverage_df)
        
        print(f"\nTGA charts saved to: {self.output_dir}")


if __name__ == "__main__":
    from data_ingest import load_processed_data, DataFetcher
    
    # Load data
    df = load_processed_data()
    
    # Generate standard charts
    generator = ChartGenerator(df)
    generator.generate_all_charts()
    
    # Generate TGA charts if data available
    try:
        fetcher = DataFetcher()
        tga_df = fetcher.fetch_tga_balance()
        tga_generator = TGAChartGenerator(df, tga_df)
        tga_generator.generate_all_tga_charts()
    except Exception as e:
        print(f"TGA charts skipped: {e}")
