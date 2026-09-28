"""
Treasury Issuance Analysis - Configuration Module
Centralized configuration for all project settings
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import matplotlib.pyplot as plt


@dataclass
class ProjectConfig:
    """Central configuration for the Treasury analysis project"""
    
    # Directory paths - resolved relative to this file's location
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    
    @property
    def charts_dir(self) -> Path:
        return self.base_dir / "charts"
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"
    
    @property
    def cache_dir(self) -> Path:
        return self.base_dir / "cache"
    
    # API Configuration
    mspd_base_url: str = (
        "https://api.fiscaldata.treasury.gov/services/api/"
        "fiscal_service/v1/debt/mspd/mspd_table_3_market"
    )
    tga_base_url: str = (
        "https://api.fiscaldata.treasury.gov/services/api/"
        "fiscal_service/v1/accounting/dts/operating_cash_balance"
    )
    page_size: int = 5000
    request_timeout: int = 60
    max_retries: int = 3
    
    # Date ranges for historical data fetch
    date_ranges: List[tuple] = field(default_factory=lambda: [
        ("2009-12-31", "2015-01-01"),
        ("2015-01-01", "2020-01-01"),
        ("2020-01-01", None),
    ])
    
    # Maturity bucket definitions (years)
    maturity_buckets: Dict[str, tuple] = field(default_factory=lambda: {
        "0-1y": (0, 1),
        "1-3y": (1, 3),
        "3-5y": (3, 5),
        "5-7y": (5, 7),
        "7-10y": (7, 10),
        "10-20y": (10, 20),
        "20y+": (20, float('inf')),
    })
    
    # Original tenor program definitions (years)
    tenor_programs: Dict[str, tuple] = field(default_factory=lambda: {
        "Bills": (0, 1),
        "2y": (1, 2.5),
        "3y": (2.5, 4),
        "5y": (4, 6),
        "7y": (6, 8),
        "10y": (8, 15),
        "20y": (15, 25),
        "30y": (25, float('inf')),
    })
    
    # TGA Coverage horizons (days)
    tga_horizons: List[int] = field(default_factory=lambda: [30, 60, 90, 180, 365])
    
    # Chart configuration
    figsize_large: tuple = (14, 8)
    figsize_medium: tuple = (12, 7)
    figsize_small: tuple = (10, 6)
    dpi: int = 300
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [self.base_dir, self.charts_dir, self.data_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_matplotlib_style(self):
        """Configure matplotlib style for all charts"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            # Fallback for older matplotlib versions
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                plt.style.use('ggplot')
        
        plt.rcParams.update({
            'figure.figsize': self.figsize_medium,
            'figure.dpi': 100,
            'savefig.dpi': self.dpi,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
        })


# Global configuration instance
CONFIG = ProjectConfig()


def get_bucket_order() -> List[str]:
    """Return maturity buckets in logical order"""
    return ["0-1y", "1-3y", "3-5y", "5-7y", "7-10y", "10-20y", "20y+"]


def get_tenor_order() -> List[str]:
    """Return tenor programs in logical order"""
    return ["Bills", "2y", "3y", "5y", "7y", "10y", "20y", "30y"]


def format_currency(value: float, decimals: int = 1) -> str:
    """Format large currency values with appropriate suffix"""
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_val >= 1e12:
        return f"{sign}${abs_val/1e12:.{decimals}f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.{decimals}f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.{decimals}f}M"
    else:
        return f"{sign}${abs_val:,.0f}"
