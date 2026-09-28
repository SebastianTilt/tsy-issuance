"""
Treasury Issuance Analysis - Main Entry Point
Run this script to execute the full pipeline and generate charts
"""

import sys
import argparse
from pathlib import Path

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG


def run_pipeline(use_cache: bool = True):
    """Run the full data pipeline"""
    from data_ingest import run_full_pipeline
    return run_full_pipeline(use_cache=use_cache)


def generate_charts(df=None):
    """Generate all static charts"""
    from data_ingest import load_processed_data
    from charts import ChartGenerator
    
    if df is None:
        df = load_processed_data()
    
    generator = ChartGenerator(df)
    generator.generate_all_charts()


def generate_tga_charts():
    """Generate TGA coverage charts"""
    from data_ingest import load_processed_data, DataFetcher
    from charts import TGAChartGenerator
    
    df = load_processed_data()
    fetcher = DataFetcher()
    tga_df = fetcher.fetch_tga_balance()
    
    generator = TGAChartGenerator(df, tga_df)
    generator.generate_all_tga_charts()


def launch_dashboard():
    """Launch the interactive dashboard"""
    from dashboard import main as dashboard_main
    dashboard_main()


def print_summary():
    """Print portfolio summary statistics"""
    from data_ingest import load_processed_data
    from analytics import PortfolioAnalytics
    
    df = load_processed_data()
    analytics = PortfolioAnalytics(df)
    metrics = analytics.compute_portfolio_metrics()
    
    print("\n" + "=" * 60)
    print("TREASURY PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"\nAs of: {metrics.record_date.strftime('%Y-%m-%d')}")
    print(f"\nTotal Outstanding:     ${metrics.total_outstanding/1e12:.2f}T")
    print(f"Number of Securities:  {metrics.cusip_count:,}")
    print(f"\nWeighted Avg Maturity: {metrics.weighted_avg_maturity:.2f} years")
    print(f"Weighted Avg Age:      {metrics.weighted_avg_age:.2f} years")
    print(f"\nShort Share (≤3y):     {metrics.short_share:.1f}%")
    print(f"Long Share (≥7y):      {metrics.long_share:.1f}%")
    print(f"Bill Share:            {metrics.bill_share:.1f}%")
    print(f"TIPS Share:            {metrics.tips_share:.1f}%")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Treasury Issuance Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all              # Run full pipeline, charts, and summary
  python main.py --pipeline         # Only run data pipeline
  python main.py --charts           # Only generate charts
  python main.py --dashboard        # Launch interactive dashboard
  python main.py --summary          # Print portfolio summary
  python main.py --refresh          # Force refresh data from API
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run pipeline, generate charts, print summary')
    parser.add_argument('--pipeline', action='store_true',
                       help='Run data ingestion pipeline')
    parser.add_argument('--charts', action='store_true',
                       help='Generate static charts')
    parser.add_argument('--tga', action='store_true',
                       help='Generate TGA coverage charts')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch interactive dashboard')
    parser.add_argument('--summary', action='store_true',
                       help='Print portfolio summary')
    parser.add_argument('--refresh', action='store_true',
                       help='Force refresh data (ignore cache)')
    
    args = parser.parse_args()
    
    # Default to --all if no arguments
    if not any([args.all, args.pipeline, args.charts, args.tga, 
                args.dashboard, args.summary]):
        args.all = True
    
    use_cache = not args.refresh
    
    try:
        if args.all:
            print("Running full analysis...")
            df = run_pipeline(use_cache=use_cache)
            generate_charts(df)
            print_summary()
            
        else:
            if args.pipeline:
                run_pipeline(use_cache=use_cache)
            
            if args.charts:
                generate_charts()
            
            if args.tga:
                generate_tga_charts()
            
            if args.summary:
                print_summary()
            
            if args.dashboard:
                launch_dashboard()
                
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the pipeline first:")
        print("  python main.py --pipeline")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
