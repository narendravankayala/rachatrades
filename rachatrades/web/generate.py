"""Website generator - creates static HTML from scan results."""

import json
import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


def generate_website(
    scan_results_path: str = "data/latest_scan.json",
    db_path: str = "data/positions.db",
    output_dir: str = "web/public",
):
    """
    Generate static website from scan results.

    Args:
        scan_results_path: Path to latest scan JSON
        db_path: Path to positions database
        output_dir: Output directory for HTML files
    """
    from rachatrades.core.signals import PositionTracker

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load scan results
    scan_data = {}
    if Path(scan_results_path).exists():
        with open(scan_results_path) as f:
            scan_data = json.load(f)

    # Get position data
    tracker = PositionTracker(db_path)
    open_positions = tracker.get_open_positions()
    closed_positions = tracker.get_closed_positions()
    stats = tracker.get_stats()

    # Setup Jinja2
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))

    # Generate index.html
    template = env.get_template("index.html")
    html = template.render(
        scan=scan_data,
        open_positions=open_positions,
        closed_positions=closed_positions[:20],  # Last 20
        stats=stats,
        zone_counts=scan_data.get("zone_counts", {"LONG": 0, "FLAT": 0, "SHORT": 0}),
        generated_at=datetime.now().isoformat(),
    )

    index_path = output_path / "index.html"
    with open(index_path, "w") as f:
        f.write(html)

    logger.info(f"Generated {index_path}")

    # Copy static assets
    static_src = Path(__file__).parent / "static"
    if static_src.exists():
        import shutil
        static_dst = output_path / "static"
        if static_dst.exists():
            shutil.rmtree(static_dst)
        shutil.copytree(static_src, static_dst)
        logger.info(f"Copied static assets to {static_dst}")

    logger.info(f"Website generated in {output_path}")


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    generate_website()


if __name__ == "__main__":
    main()
