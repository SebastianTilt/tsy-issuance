# Treasury Issuance Analysis

A comprehensive Python toolkit for analyzing U.S. Treasury debt issuance, portfolio composition, and maturity profiles using data from the Treasury Fiscal Data API.

## Features

- **Data Pipeline**: Automated fetching, caching, and transformation of MSPD (Monthly Statement of Public Debt) data
- **Portfolio Analytics**: WAM, maturity distribution, short/long rotation, refinancing half-life
- **TGA Coverage Analysis**: Treasury General Account coverage ratios vs upcoming maturities
- **Static Charts**: 20+ publication-quality visualizations
- **Live Dashboard**: Interactive Tkinter dashboard with real-time chart updates

## Installation

### Prerequisites

```bash
pip install pandas numpy matplotlib requests
```

For the dashboard:
```bash
pip install tkinter  # Usually included with Python
```

### Setup

1. Clone or copy the project files to your desired location
2. Update the `base_dir` path in `config.py`:

```python
@dataclass
class ProjectConfig:
    base_dir: Path = field(default_factory=lambda: Path(r"C:\Users\st\Downloads\TsyIssuance"))
```

## Usage

### Quick Start

```bash
# Run full analysis (pipeline + charts + summary)
python main.py --all

# Or run with fresh data (ignores cache)
python main.py --all --refresh
```

### Command Line Options

```bash
python main.py --pipeline    # Only run data pipeline
python main.py --charts      # Generate static charts
python main.py --tga         # Generate TGA coverage charts
python main.py --summary     # Print portfolio summary
python main.py --dashboard   # Launch interactive dashboard
python main.py --refresh     # Force API refresh (ignore cache)
```

### Interactive Dashboard

```bash
python main.py --dashboard
# Or directly:
python dashboard.py
```

The dashboard provides:
- **Overview Tab**: Key metrics, WAM chart, composition pie
- **Composition Tab**: Instrument and maturity bucket trends
- **Issuance Tab**: Net issuance flows, bill share analysis
- **Maturity Tab**: Maturity wall, short/long rotation diagram

### Programmatic Usage

```python
from data_ingest import run_full_pipeline, load_processed_data
from analytics import PortfolioAnalytics
from charts import ChartGenerator

# Run pipeline
df = run_full_pipeline()

# Or load existing data
df = load_processed_data()

# Compute analytics
analytics = PortfolioAnalytics(df)
metrics = analytics.compute_portfolio_metrics()
print(f"WAM: {metrics.weighted_avg_maturity:.2f} years")

# Generate specific charts
generator = ChartGenerator(df)
generator.chart_short_long_rotation()
```

## Project Structure

```
TsyIssuance/
├── config.py          # Central configuration
├── data_ingest.py     # API fetching and data transformation
├── analytics.py       # Core analytical calculations
├── charts.py          # Static chart generation
├── dashboard.py       # Interactive Tkinter dashboard
├── main.py           # CLI entry point
├── README.md
├── data/             # Processed data files
│   └── mspd_issuance_panel.parquet
├── cache/            # API response cache
└── charts/           # Generated chart images
```

## Key Improvements Over Original

### Logic Corrections

1. **Flow Calculations**: Fixed the stock-flow bridge calculation:
   - Issued flow now correctly captures only current-period issuance
   - Redeemed flow computed as change in cumulative redemptions
   - Proper handling of inflation adjustments for TIPS

2. **Maturity Bucket Classification**: Consistent application of bucket boundaries:
   - 0-1y: `[0, 1)`
   - 1-3y: `[1, 3)`
   - etc.

3. **WAM Calculation**: Uses proper par-weighted averaging with NaN handling

### Architecture Improvements

1. **Modular Design**: Separated concerns into distinct modules
2. **Caching**: API responses cached to avoid redundant fetches
3. **Configuration**: Centralized settings in `config.py`
4. **Error Handling**: Retry logic for API calls, graceful degradation

### New Features

1. **Interactive Dashboard**: Live Tkinter GUI with embedded matplotlib
2. **Parallel Fetching**: Multi-threaded API requests for faster data loading
3. **Date Navigation**: Browse historical snapshots in dashboard
4. **Export Functionality**: Save charts to specified folder

## Charts Generated

### Core Charts (A-J)
- **A**: Instrument group composition (absolute & %)
- **B**: Maturity bucket composition (absolute & %)
- **C**: Weighted Average Maturity (total & by group)
- **D**: Net issuance by maturity bucket
- **E**: Bill share of stock and flows
- **F**: Maturity wall (rollover profile)
- **G**: Original tenor vs remaining maturity heatmap
- **J**: Portfolio refinancing half-life

### Extended Charts (N-U)
- **N**: Short vs Long rotation diagram
- **O**: Bridge error diagnostics
- **U**: Issuance WAM vs portfolio half-life

### TGA Charts
- Coverage ratios by horizon
- TGA balance vs maturing debt
- Coverage by instrument group

## Data Sources

- **MSPD Table 3**: Monthly Statement of Public Debt
  - Endpoint: `api.fiscaldata.treasury.gov/.../mspd_table_3_market`
  - Fields: CUSIP, outstanding amounts, issue/maturity dates, flows

- **TGA Balance**: Daily Treasury Statement
  - Endpoint: `api.fiscaldata.treasury.gov/.../operating_cash_balance`
  - Treasury General Account closing balance

## Configuration Options

Edit `config.py` to customize:

```python
@dataclass
class ProjectConfig:
    # Paths
    base_dir: Path = ...
    
    # API settings
    page_size: int = 5000
    request_timeout: int = 60
    
    # Date ranges for historical fetch
    date_ranges: List[tuple] = [
        ("2009-12-31", "2015-01-01"),
        ("2015-01-01", "2020-01-01"),
        ("2020-01-01", None),  # None = latest
    ]
    
    # Maturity buckets (years)
    maturity_buckets: Dict[str, tuple] = {
        "0-1y": (0, 1),
        "1-3y": (1, 3),
        ...
    }
    
    # TGA horizons (days)
    tga_horizons: List[int] = [30, 60, 90, 180, 365]
```

## Troubleshooting

### "No data found" error
Run the pipeline first:
```bash
python main.py --pipeline
```

### API timeouts
Increase timeout in `config.py`:
```python
request_timeout: int = 120  # seconds
```

### Cache issues
Force refresh:
```bash
python main.py --refresh
```

Or delete the `cache/` folder manually.

### Dashboard not launching
Ensure Tkinter is installed:
```python
import tkinter
tkinter._test()  # Should open a window
```

## License

MIT License - Free for personal and commercial use.

## Acknowledgments

- Data sourced from [Treasury Fiscal Data](https://fiscaldata.treasury.gov/)
- Built with pandas, matplotlib, and Tkinter
