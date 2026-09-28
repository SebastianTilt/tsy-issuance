"""
Treasury Issuance Analysis - Live Dashboard
Interactive Tkinter dashboard with embedded matplotlib charts
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable
import threading
import queue
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG, get_bucket_order, format_currency
from analytics import PortfolioAnalytics


class MetricCard(ttk.Frame):
    """A single metric display card"""
    
    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.value_var = tk.StringVar(value="--")
        self.change_var = tk.StringVar(value="")
        
        # Title
        ttk.Label(self, text=title, font=('Helvetica', 9)).pack(anchor='w')
        
        # Value
        ttk.Label(
            self, 
            textvariable=self.value_var,
            font=('Helvetica', 18, 'bold')
        ).pack(anchor='w')
        
        # Change indicator
        self.change_label = ttk.Label(
            self, 
            textvariable=self.change_var,
            font=('Helvetica', 9)
        )
        self.change_label.pack(anchor='w')
    
    def set_value(self, value: str, change: str = "", change_positive: bool = True):
        """Update the metric value and change"""
        self.value_var.set(value)
        self.change_var.set(change)
        
        # Color code the change
        if change:
            color = 'green' if change_positive else 'red'
            self.change_label.configure(foreground=color)


class ChartPanel(ttk.Frame):
    """A panel containing a matplotlib chart"""
    
    def __init__(self, parent, title: str, figsize=(6, 4), **kwargs):
        super().__init__(parent, **kwargs)
        
        self.title = title
        
        # Create figure
        self.fig = Figure(figsize=figsize, dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=2.0)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def clear(self):
        """Clear the chart"""
        self.ax.clear()
    
    def refresh(self):
        """Refresh the canvas"""
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
    
    def show_loading(self):
        """Show loading message"""
        self.clear()
        self.ax.text(
            0.5, 0.5, 'Loading...', 
            ha='center', va='center',
            fontsize=14, color='gray',
            transform=self.ax.transAxes
        )
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.refresh()


class DashboardApp:
    """Main dashboard application"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Treasury Issuance Analysis Dashboard")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Data
        self.df: Optional[pd.DataFrame] = None
        self.analytics: Optional[PortfolioAnalytics] = None
        self.metrics_ts: Optional[pd.DataFrame] = None
        self.available_dates: List[pd.Timestamp] = []
        self.current_date: Optional[pd.Timestamp] = None
        
        # Queue for thread communication
        self.data_queue = queue.Queue()
        
        # Chart panels
        self.chart_panels: Dict[str, ChartPanel] = {}
        self.metric_cards: Dict[str, MetricCard] = {}
        
        # Setup UI
        self._setup_styles()
        self._create_layout()
        self._create_menu()
        
        # Load data in background
        self.root.after(100, self._start_data_load)
        
        # Periodic update check
        self.root.after(100, self._check_data_queue)
    
    def _setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Card.TFrame', relief='solid', borderwidth=1)
    
    def _create_layout(self):
        """Create main layout"""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self._create_header()
        
        # Content area with notebook
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create tabs
        self._create_overview_tab()
        self._create_composition_tab()
        self._create_flows_tab()
        self._create_maturity_tab()
        
        # Status bar
        self._create_status_bar()
    
    def _create_header(self):
        """Create header with title and controls"""
        header = ttk.Frame(self.main_frame)
        header.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        ttk.Label(
            header, 
            text="📊 Treasury Debt Analysis Dashboard",
            style='Title.TLabel'
        ).pack(side=tk.LEFT)
        
        # Controls frame
        controls = ttk.Frame(header)
        controls.pack(side=tk.RIGHT)
        
        # Date selector
        ttk.Label(controls, text="As of:").pack(side=tk.LEFT, padx=5)
        
        self.date_var = tk.StringVar(value="Loading...")
        self.date_combo = ttk.Combobox(
            controls, 
            textvariable=self.date_var,
            state='readonly',
            width=15
        )
        self.date_combo.pack(side=tk.LEFT, padx=5)
        self.date_combo.bind('<<ComboboxSelected>>', self._on_date_change)
        
        # Reload button (re-reads the processed panel from disk; it does not
        # call the API. Run `python main.py --pipeline` for new data.)
        self.refresh_btn = ttk.Button(
            controls, 
            text="🔄 Reload from disk",
            command=self._refresh_data
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=10)
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar(value="Loading data...")
        status_bar = ttk.Label(
            self.main_frame, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export All Charts", command=self._export_charts)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh All Charts", command=self._refresh_all_charts)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    # =========================================================================
    # TABS
    # =========================================================================
    
    def _create_overview_tab(self):
        """Create overview tab"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="📈 Overview")
        
        # Top: Metrics
        metrics_frame = ttk.LabelFrame(tab, text="Key Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        metrics_config = [
            ('total_outstanding', 'Total Outstanding'),
            ('cusip_count', 'Securities'),
            ('wam', 'WAM (Years)'),
            ('short_share', 'Short (≤3y)'),
            ('long_share', 'Long (≥7y)'),
            ('bill_share', 'Bills'),
        ]
        
        for i, (key, title) in enumerate(metrics_config):
            card = MetricCard(metrics_frame, title)
            card.grid(row=0, column=i, padx=10, pady=5, sticky='nsew')
            metrics_frame.columnconfigure(i, weight=1)
            self.metric_cards[key] = card
        
        # Bottom: Charts
        charts_frame = ttk.Frame(tab)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left chart: WAM over time
        left_frame = ttk.LabelFrame(charts_frame, text="Weighted Average Maturity")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.chart_panels['wam'] = ChartPanel(left_frame, "WAM", figsize=(5, 3.5))
        self.chart_panels['wam'].pack(fill=tk.BOTH, expand=True)
        
        # Right chart: Composition
        right_frame = ttk.LabelFrame(charts_frame, text="Portfolio Composition")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.chart_panels['composition'] = ChartPanel(right_frame, "Composition", figsize=(5, 3.5))
        self.chart_panels['composition'].pack(fill=tk.BOTH, expand=True)
    
    def _create_composition_tab(self):
        """Create composition analysis tab"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="🥧 Composition")
        
        # Left: By instrument
        left_frame = ttk.LabelFrame(tab, text="By Instrument Group")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.chart_panels['instrument'] = ChartPanel(left_frame, "Instrument", figsize=(6, 5))
        self.chart_panels['instrument'].pack(fill=tk.BOTH, expand=True)
        
        # Right: By maturity bucket
        right_frame = ttk.LabelFrame(tab, text="By Maturity Bucket")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.chart_panels['bucket'] = ChartPanel(right_frame, "Bucket", figsize=(6, 5))
        self.chart_panels['bucket'].pack(fill=tk.BOTH, expand=True)
    
    def _create_flows_tab(self):
        """Create issuance flows tab"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="💰 Issuance")
        
        # Top: Net issuance
        top_frame = ttk.LabelFrame(tab, text="Net Issuance by Original Tenor")
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.chart_panels['net_issuance'] = ChartPanel(top_frame, "Net Issuance", figsize=(10, 4))
        self.chart_panels['net_issuance'].pack(fill=tk.BOTH, expand=True)
        
        # Bottom: Bill share
        bottom_frame = ttk.LabelFrame(tab, text="Bill Share Analysis")
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.chart_panels['bill_share'] = ChartPanel(bottom_frame, "Bill Share", figsize=(10, 3))
        self.chart_panels['bill_share'].pack(fill=tk.BOTH, expand=True)
    
    def _create_maturity_tab(self):
        """Create maturity analysis tab"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="📅 Maturity")
        
        # Top: Maturity wall
        top_frame = ttk.LabelFrame(tab, text="Maturity Wall (Next 5 Years)")
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.chart_panels['maturity_wall'] = ChartPanel(top_frame, "Maturity Wall", figsize=(10, 4))
        self.chart_panels['maturity_wall'].pack(fill=tk.BOTH, expand=True)
        
        # Bottom: Short vs Long rotation
        bottom_frame = ttk.LabelFrame(tab, text="Short vs Long Rotation")
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.chart_panels['rotation'] = ChartPanel(bottom_frame, "Rotation", figsize=(6, 4))
        self.chart_panels['rotation'].pack(fill=tk.BOTH, expand=True)
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def _start_data_load(self):
        """Start loading data in background thread"""
        self.status_var.set("Loading data from disk...")
        
        # Show loading on all charts
        for panel in self.chart_panels.values():
            panel.show_loading()
        
        # Start background thread
        thread = threading.Thread(target=self._load_data_thread, daemon=True)
        thread.start()
    
    def _load_data_thread(self):
        """Background thread for data loading"""
        try:
            # Try to load from disk first
            from data_ingest import load_processed_data
            df = load_processed_data()
            self.data_queue.put(('success', df))
        except FileNotFoundError:
            self.data_queue.put(('not_found', None))
        except Exception as e:
            self.data_queue.put(('error', str(e)))
    
    def _check_data_queue(self):
        """Check queue for data loading results"""
        try:
            while True:
                status, data = self.data_queue.get_nowait()
                
                if status == 'success':
                    self._on_data_loaded(data)
                elif status == 'not_found':
                    self.status_var.set("No data found. Please run data pipeline first.")
                    messagebox.showwarning(
                        "Data Not Found",
                        "No processed data found.\n\n"
                        "Please run the data pipeline first:\n"
                        "python main.py --pipeline"
                    )
                elif status == 'error':
                    self.status_var.set(f"Error: {data}")
                    messagebox.showerror("Error", f"Failed to load data:\n{data}")
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._check_data_queue)
    
    def _on_data_loaded(self, df: pd.DataFrame):
        """Handle successful data load"""
        self.df = df
        self.analytics = PortfolioAnalytics(df)
        # Time series don't depend on the selected date: compute once per load
        self.metrics_ts = self.analytics.compute_metrics_timeseries()
        
        # Get available dates
        self.available_dates = sorted(df['record_date'].unique())
        self.current_date = self.available_dates[-1]
        
        # Update date selector
        date_strings = [d.strftime('%Y-%m-%d') for d in self.available_dates]
        self.date_combo['values'] = date_strings
        self.date_combo.set(date_strings[-1])
        
        # Update status
        self.status_var.set(
            f"Data loaded: {len(self.available_dates)} dates, "
            f"{df['record_date'].min().strftime('%Y-%m')} to "
            f"{df['record_date'].max().strftime('%Y-%m')}"
        )
        
        # Refresh all charts
        self._refresh_all_charts()
    
    # =========================================================================
    # CHART UPDATES
    # =========================================================================
    
    def _refresh_all_charts(self):
        """Refresh all charts"""
        if self.df is None:
            return
        
        self._update_metrics()
        self._update_wam_chart()
        self._update_composition_chart()
        self._update_instrument_chart()
        self._update_bucket_chart()
        self._update_net_issuance_chart()
        self._update_bill_share_chart()
        self._update_maturity_wall_chart()
        self._update_rotation_chart()
    
    def _refresh_snapshot_charts(self):
        """Redraw only the views that depend on the selected date"""
        if self.df is None:
            return
        
        self._update_metrics()
        self._update_wam_chart()  # cheap: uses cached time series, moves marker
        self._update_composition_chart()
        self._update_maturity_wall_chart()
    
    def _update_metrics(self):
        """Update metric cards"""
        if self.analytics is None:
            return
        
        metrics = self.analytics.compute_portfolio_metrics(self.current_date)
        
        self.metric_cards['total_outstanding'].set_value(
            format_currency(metrics.total_outstanding)
        )
        self.metric_cards['cusip_count'].set_value(
            f"{metrics.cusip_count:,}"
        )
        self.metric_cards['wam'].set_value(
            f"{metrics.weighted_avg_maturity:.2f}"
        )
        self.metric_cards['short_share'].set_value(
            f"{metrics.short_share:.1f}%"
        )
        self.metric_cards['long_share'].set_value(
            f"{metrics.long_share:.1f}%"
        )
        self.metric_cards['bill_share'].set_value(
            f"{metrics.bill_share:.1f}%"
        )
    
    def _update_wam_chart(self):
        """Update WAM chart"""
        panel = self.chart_panels['wam']
        panel.clear()
        
        metrics = self.metrics_ts
        
        panel.ax.plot(metrics['record_date'], metrics['wam_years'], 
                     linewidth=2, color='navy')
        panel.ax.set_xlabel('Date')
        panel.ax.set_ylabel('WAM (Years)')
        panel.ax.grid(True, alpha=0.3)
        
        # Mark current date
        if self.current_date is not None:
            current_wam = metrics[metrics['record_date'] == self.current_date]['wam_years'].iloc[0]
            panel.ax.axhline(current_wam, color='red', linestyle='--', alpha=0.5)
        
        panel.refresh()
    
    def _update_composition_chart(self):
        """Update composition pie chart"""
        panel = self.chart_panels['composition']
        panel.clear()
        
        snapshot = self.analytics.get_snapshot(self.current_date)
        by_group = snapshot.groupby('instrument_group')['par_outstanding_usd'].sum()
        
        # Filter small slices
        total = by_group.sum()
        significant = by_group[by_group / total >= 0.02]
        other = by_group[by_group / total < 0.02].sum()
        if other > 0:
            significant['Other'] = other
        
        panel.ax.pie(
            significant.values,
            labels=significant.index,
            autopct='%1.1f%%',
            startangle=90
        )
        panel.ax.set_title(f'As of {self.current_date.strftime("%Y-%m-%d")}')
        
        panel.refresh()
    
    def _update_instrument_chart(self):
        """Update instrument group stacked area"""
        panel = self.chart_panels['instrument']
        panel.clear()
        
        pivot = self.analytics.compute_composition_by_instrument()
        
        pivot.plot(kind='area', stacked=True, ax=panel.ax, alpha=0.8, legend=True)
        panel.ax.set_xlabel('Date')
        panel.ax.set_ylabel('Outstanding ($)')
        panel.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x/1e12:.1f}T')
        )
        panel.ax.legend(loc='upper left', fontsize=8)
        
        panel.refresh()
    
    def _update_bucket_chart(self):
        """Update maturity bucket stacked area"""
        panel = self.chart_panels['bucket']
        panel.clear()
        
        pivot = self.analytics.compute_composition_by_bucket()
        
        pivot.plot(kind='area', stacked=True, ax=panel.ax, alpha=0.8, legend=True)
        panel.ax.set_xlabel('Date')
        panel.ax.set_ylabel('Outstanding ($)')
        panel.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x/1e12:.1f}T')
        )
        panel.ax.legend(loc='upper left', fontsize=8)
        
        panel.refresh()
    
    def _update_net_issuance_chart(self):
        """Update net issuance bar chart"""
        panel = self.chart_panels['net_issuance']
        panel.clear()
        
        pivot = self.analytics.compute_net_issuance_by_tenor(freq='M')
        
        # Only show recent data for readability
        if len(pivot) > 36:
            pivot = pivot.iloc[-36:]
        
        pivot.plot(kind='bar', stacked=True, ax=panel.ax, width=0.8, alpha=0.8)
        panel.ax.set_xlabel('Date')
        panel.ax.set_ylabel('Net Issuance ($)')
        panel.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x/1e9:.0f}B')
        )
        panel.ax.axhline(0, color='black', linewidth=0.8)
        panel.ax.legend(loc='upper left', fontsize=7)
        
        # Clean up x-axis
        n = len(pivot)
        step = max(1, n // 12)
        panel.ax.set_xticks(range(0, n, step))
        panel.ax.set_xticklabels(
            [pivot.index[i].strftime('%Y-%m') for i in range(0, n, step)],
            rotation=45, ha='right', fontsize=8
        )
        
        panel.refresh()
    
    def _update_bill_share_chart(self):
        """Update bill share chart"""
        panel = self.chart_panels['bill_share']
        panel.clear()
        
        bill_data = self.analytics.compute_bill_share_timeseries()
        
        panel.ax.plot(bill_data.index, bill_data['bill_stock_share'], 
                     linewidth=2, label='Stock Share', color='darkred')
        panel.ax.set_xlabel('Date')
        panel.ax.set_ylabel('Bill Share (%)')
        panel.ax.legend(loc='upper right')
        panel.ax.grid(True, alpha=0.3)
        panel.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x:.0f}%')
        )
        
        panel.refresh()
    
    def _update_maturity_wall_chart(self):
        """Update maturity wall chart"""
        panel = self.chart_panels['maturity_wall']
        panel.clear()
        
        pivot = self.analytics.compute_maturity_wall(horizon_years=5, date=self.current_date)
        
        if len(pivot) == 0:
            panel.ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            panel.refresh()
            return
        
        pivot.plot(kind='bar', stacked=True, ax=panel.ax, width=0.8, alpha=0.8)
        panel.ax.set_xlabel('Maturity Quarter')
        panel.ax.set_ylabel('Par Maturing ($)')
        panel.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x/1e9:.0f}B')
        )
        panel.ax.legend(loc='upper right', fontsize=7)
        
        # Clean up x-axis
        plt.setp(panel.ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        panel.refresh()
    
    def _update_rotation_chart(self):
        """Update short vs long rotation chart"""
        panel = self.chart_panels['rotation']
        panel.clear()
        
        rotation = self.metrics_ts[['record_date', 'short_share_pct', 'long_share_pct']]
        n = len(rotation)
        
        if n < 2:
            panel.ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
            panel.refresh()
            return
        
        t_norm = np.linspace(0, 1, n)
        
        # Path
        panel.ax.plot(
            rotation['short_share_pct'],
            rotation['long_share_pct'],
            color='lightgray',
            linewidth=1,
            zorder=1
        )
        
        # Scatter
        sc = panel.ax.scatter(
            rotation['short_share_pct'],
            rotation['long_share_pct'],
            c=t_norm,
            cmap='viridis',
            s=20,
            alpha=0.8,
            zorder=2
        )
        
        # Start/end markers
        start = rotation.iloc[0]
        end = rotation.iloc[-1]
        
        panel.ax.scatter([start['short_share_pct']], [start['long_share_pct']],
                        s=60, facecolors='white', edgecolors='black', zorder=3)
        panel.ax.scatter([end['short_share_pct']], [end['long_share_pct']],
                        s=60, facecolors='gold', edgecolors='black', zorder=3)
        
        panel.ax.set_xlabel('Short Share (≤3y) %')
        panel.ax.set_ylabel('Long Share (≥7y) %')
        panel.ax.grid(True, alpha=0.2)
        
        panel.refresh()
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_date_change(self, event):
        """Handle date selection change"""
        date_str = self.date_var.get()
        self.current_date = pd.Timestamp(date_str)
        self._refresh_snapshot_charts()
    
    def _refresh_data(self):
        """Reload the processed panel from disk"""
        self.status_var.set("Reloading data from disk...")
        self._start_data_load()
    
    def _export_charts(self):
        """Export all charts to files"""
        if self.df is None:
            messagebox.showwarning("No Data", "Please load data first.")
            return
        
        folder = filedialog.askdirectory(title="Select Export Folder")
        if not folder:
            return
        
        folder = Path(folder)
        
        for name, panel in self.chart_panels.items():
            path = folder / f"{name}.png"
            panel.fig.savefig(path, dpi=150, bbox_inches='tight')
        
        messagebox.showinfo("Export Complete", f"Charts exported to:\n{folder}")
    
    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Treasury Issuance Analysis Dashboard\n\n"
            "Analyzes U.S. Treasury debt issuance using\n"
            "data from the Treasury Fiscal Data API.\n\n"
            "Version 2.0"
        )


def main():
    """Main entry point"""
    # Configure matplotlib for threading
    plt.switch_backend('TkAgg')
    
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
