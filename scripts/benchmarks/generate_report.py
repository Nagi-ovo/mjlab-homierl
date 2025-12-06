#!/usr/bin/env python3
"""Generate benchmark report from WandB runs.

This script fetches training metrics from WandB and generates a static HTML
dashboard similar to ASV (airspeed-velocity) for tracking RL training performance
over time.

The dashboard shows:
- Metrics organized by category (Train/, Loss/, Perf/, etc.)
- Each metric shows a timeline of final values across runs
- Easy to spot regressions at a glance

Usage:
    python scripts/benchmarks/generate_report.py --run-path <entity/project/run_id>
    python scripts/benchmarks/generate_report.py --project mjlab --tag nightly
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import wandb

# Metric categories to fetch (prefix matching)
METRIC_CATEGORIES = [
  "Train/",
  "Metrics/",
  "Episode_Reward/",
  "Episode_Termination/",
  "Loss/",
  "Perf/",
  "Policy/",
]


def fetch_run_data(run_path: str) -> dict:
  """Fetch final metrics from a single WandB run (summary only, no history).

  Args:
      run_path: Full run path in format "entity/project/run_id"

  Returns:
      Dictionary containing run metadata and final metric values
  """
  api = wandb.Api()
  run = api.run(run_path)

  # Get summary (final values) - this is fast
  summary = dict(run.summary)

  # Filter to only metrics in our categories
  filtered_summary = {
    k: v
    for k, v in summary.items()
    if any(k.startswith(cat) for cat in METRIC_CATEGORIES)
  }

  print(f"  Found {len(filtered_summary)} metrics")

  return {
    "id": run.id,
    "name": run.name,
    "path": run_path,
    "state": run.state,
    "created_at": run.created_at,
    "config": dict(run.config),
    "summary": filtered_summary,
    "tags": run.tags,
    "url": run.url,
  }


def fetch_project_runs(
  entity: str,
  project: str,
  tag: str | None = None,
  limit: int = 30,
) -> list[dict]:
  """Fetch multiple runs from a WandB project.

  Args:
      entity: WandB entity (username or team)
      project: WandB project name
      tag: Optional tag to filter runs
      limit: Maximum number of runs to fetch

  Returns:
      List of run data dictionaries
  """
  api = wandb.Api()

  filters = {}
  if tag:
    filters["tags"] = tag

  runs = api.runs(
    f"{entity}/{project}",
    filters=filters,
    order="-created_at",
  )

  run_data = []
  for i, run in enumerate(runs):
    if i >= limit:
      break
    if run.state == "finished":
      run_data.append(fetch_run_data(f"{entity}/{project}/{run.id}"))

  return run_data


def generate_html_report(runs: list[dict], output_dir: Path) -> None:
  """Generate static HTML dashboard from run data.

  Args:
      runs: List of run data dictionaries
      output_dir: Directory to write HTML files to
  """
  output_dir.mkdir(parents=True, exist_ok=True)

  # Save run data as JSON for the dashboard
  data_dir = output_dir / "data"
  data_dir.mkdir(exist_ok=True)

  # Collect all metrics across all runs, grouped by category
  all_metrics: dict[str, set[str]] = defaultdict(set)
  for run in runs:
    for key in run.get("summary", {}).keys():
      for cat in METRIC_CATEGORIES:
        if key.startswith(cat):
          category = cat.rstrip("/")
          short_name = key.split("/", 1)[-1] if "/" in key else key
          all_metrics[category].add(short_name)
          break

  # Process runs into format needed by dashboard
  processed_runs = []
  for run in runs:
    # Group this run's metrics by category
    metrics_by_category: dict[str, dict[str, float]] = defaultdict(dict)
    for key, value in run.get("summary", {}).items():
      for cat in METRIC_CATEGORIES:
        if key.startswith(cat):
          category = cat.rstrip("/")
          short_name = key.split("/", 1)[-1] if "/" in key else key
          if isinstance(value, (int, float)) and value == value:  # not NaN
            metrics_by_category[category][short_name] = value
          break

    processed_runs.append(
      {
        "id": run["id"],
        "name": run["name"],
        "url": run["url"],
        "created_at": run["created_at"],
        "tags": run.get("tags", []),
        "metrics": dict(metrics_by_category),
      }
    )

  # Write processed data
  with open(data_dir / "runs.json", "w") as f:
    json.dump(processed_runs, f, indent=2, default=str)

  # Generate index.html
  html_content = generate_dashboard_html(
    processed_runs, {k: sorted(v) for k, v in all_metrics.items()}
  )
  with open(output_dir / "index.html", "w") as f:
    f.write(html_content)

  print(f"Report generated at {output_dir / 'index.html'}")
  print(f"Data saved to {data_dir / 'runs.json'}")


def generate_dashboard_html(runs: list[dict], metrics_by_category: dict) -> str:
  """Generate the HTML dashboard content."""
  runs_json = json.dumps(runs, default=str)
  metrics_json = json.dumps(metrics_by_category, default=str)
  timestamp = datetime.now().isoformat()

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MJLab Nightly Benchmarks</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --border: #30363d;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}

        h1 {{
            font-size: 1.5rem;
            font-weight: 600;
        }}

        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 1rem;
        }}

        .card-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .card-value {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }}

        .card-value.positive {{
            color: var(--accent-green);
        }}

        .card-value.negative {{
            color: var(--accent-red);
        }}

        /* Category sections */
        .category-section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }}

        .category-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
            cursor: pointer;
        }}

        .category-header:hover {{
            background: #282e36;
        }}

        .category-title {{
            font-size: 1rem;
            font-weight: 600;
        }}

        .category-toggle {{
            color: var(--text-secondary);
            transition: transform 0.2s;
        }}

        .category-section.collapsed .category-toggle {{
            transform: rotate(-90deg);
        }}

        .category-section.collapsed .category-charts {{
            display: none;
        }}

        .category-charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1rem;
            padding: 1.5rem;
        }}

        .metric-chart {{
            background: var(--bg-primary);
            border-radius: 4px;
            padding: 1rem;
        }}

        .metric-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }}

        .metric-title {{
            font-size: 0.875rem;
            font-weight: 500;
        }}

        .metric-value {{
            font-size: 0.875rem;
            color: var(--text-secondary);
        }}

        .metric-delta {{
            font-size: 0.75rem;
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
        }}

        .metric-delta.positive {{
            background: rgba(63, 185, 80, 0.2);
            color: var(--accent-green);
        }}

        .metric-delta.negative {{
            background: rgba(248, 81, 73, 0.2);
            color: var(--accent-red);
        }}

        .chart-wrapper {{
            height: 150px;
        }}

        /* Runs table */
        .runs-table {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
        }}

        .runs-table h2 {{
            font-size: 0.875rem;
            font-weight: 600;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        tr:hover {{
            background: var(--bg-tertiary);
        }}

        a {{
            color: var(--accent);
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        .empty-state {{
            text-align: center;
            padding: 3rem;
            color: var(--text-secondary);
        }}

        .chart-subtitle {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}

        @media (max-width: 768px) {{
            .category-charts {{
                grid-template-columns: 1fr;
            }}

            .summary-cards {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>MJLab Nightly Benchmarks</h1>
                <div class="chart-subtitle">Tracking metrics across runs</div>
            </div>
            <span class="timestamp">Updated: {timestamp[:10]}</span>
        </header>

        <div class="summary-cards" id="summary-cards"></div>

        <div id="categories-container"></div>

        <div class="runs-table">
            <h2>All Runs</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Run Name</th>
                        <th>Train/mean_reward</th>
                        <th>Train/mean_episode_length</th>
                    </tr>
                </thead>
                <tbody id="runs-tbody"></tbody>
            </table>
        </div>
    </div>

    <script>
        const runs = {runs_json};
        const metricsByCategory = {metrics_json};

        // Sort runs by date (oldest first)
        runs.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));

        // Colors for charts
        const chartColors = {{
            'Train': '#3fb950',
            'Metrics': '#58a6ff',
            'Episode_Reward': '#a371f7',
            'Episode_Termination': '#f85149',
            'Loss': '#f0883e',
            'Perf': '#79c0ff',
            'Policy': '#d2a8ff'
        }};

        // Chart.js defaults
        Chart.defaults.color = '#8b949e';
        Chart.defaults.borderColor = '#30363d';

        function getMetricData(category, metricName) {{
            return runs.map(run => ({{
                x: new Date(run.created_at),
                y: run.metrics[category]?.[metricName] ?? null,
                name: run.name,
                url: run.url
            }})).filter(d => d.y !== null);
        }}

        function calculateDelta(data) {{
            if (data.length < 2) return null;
            const latest = data[data.length - 1].y;
            const previous = data[data.length - 2].y;
            if (previous === 0) return null;
            return ((latest - previous) / Math.abs(previous) * 100).toFixed(1);
        }}

        function createMetricChart(container, category, metricName, color) {{
            const data = getMetricData(category, metricName);
            if (data.length === 0) return;

            const delta = calculateDelta(data);
            const latestValue = data[data.length - 1]?.y;

            const chartDiv = document.createElement('div');
            chartDiv.className = 'metric-chart';
            chartDiv.innerHTML = `
                <div class="metric-header">
                    <span class="metric-title">${{metricName}}</span>
                    <div>
                        <span class="metric-value">${{latestValue?.toFixed(2) ?? 'N/A'}}</span>
                        ${{delta ? `<span class="metric-delta ${{delta > 0 ? 'positive' : 'negative'}}">
                            ${{delta > 0 ? '+' : ''}}${{delta}}%
                        </span>` : ''}}
                    </div>
                </div>
                <div class="chart-wrapper">
                    <canvas></canvas>
                </div>
            `;
            container.appendChild(chartDiv);

            const ctx = chartDiv.querySelector('canvas').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    datasets: [{{
                        data: data,
                        borderColor: color,
                        backgroundColor: color + '20',
                        borderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        pointBackgroundColor: color,
                        pointBorderColor: '#0d1117',
                        pointBorderWidth: 1,
                        tension: 0.1,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                title: (items) => items[0]?.raw?.name || '',
                                label: (item) => [
                                    `${{metricName}}: ${{item.raw.y?.toFixed(4)}}`,
                                    `Date: ${{new Date(item.raw.x).toLocaleDateString()}}`
                                ]
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{ unit: 'day' }},
                            display: true,
                            ticks: {{ maxTicksLimit: 4, font: {{ size: 10 }} }}
                        }},
                        y: {{
                            display: true,
                            ticks: {{ maxTicksLimit: 4, font: {{ size: 10 }} }}
                        }}
                    }},
                    onClick: (event, elements) => {{
                        if (elements.length > 0) {{
                            const url = elements[0].element.$context.raw.url;
                            if (url) window.open(url, '_blank');
                        }}
                    }}
                }}
            }});
        }}

        function initCategories() {{
            const container = document.getElementById('categories-container');

            // Sort categories in preferred order
            const categoryOrder = ['Train', 'Loss', 'Perf', 'Metrics', 'Episode_Reward', 'Episode_Termination', 'Policy'];
            const sortedCategories = Object.keys(metricsByCategory).sort((a, b) => {{
                const aIdx = categoryOrder.indexOf(a);
                const bIdx = categoryOrder.indexOf(b);
                if (aIdx === -1 && bIdx === -1) return a.localeCompare(b);
                if (aIdx === -1) return 1;
                if (bIdx === -1) return -1;
                return aIdx - bIdx;
            }});

            sortedCategories.forEach(category => {{
                const metrics = metricsByCategory[category];
                if (!metrics || metrics.length === 0) return;

                const color = chartColors[category] || '#58a6ff';

                const section = document.createElement('div');
                section.className = 'category-section';
                section.innerHTML = `
                    <div class="category-header" onclick="this.parentElement.classList.toggle('collapsed')">
                        <span class="category-title">${{category}}</span>
                        <span class="category-toggle">▼</span>
                    </div>
                    <div class="category-charts"></div>
                `;
                container.appendChild(section);

                const chartsContainer = section.querySelector('.category-charts');
                metrics.forEach(metric => {{
                    createMetricChart(chartsContainer, category, metric, color);
                }});
            }});
        }}

        function initSummary() {{
            const container = document.getElementById('summary-cards');
            if (runs.length === 0) {{
                container.innerHTML = '';
                return;
            }}

            const latestRun = runs[runs.length - 1];
            const previousRun = runs[runs.length - 2];

            const latestReward = latestRun?.metrics?.Train?.mean_reward;
            const previousReward = previousRun?.metrics?.Train?.mean_reward;
            const rewardDelta = latestReward && previousReward
                ? ((latestReward - previousReward) / Math.abs(previousReward) * 100).toFixed(1)
                : null;

            container.innerHTML = `
                <div class="card">
                    <div class="card-label">Total Runs</div>
                    <div class="card-value">${{runs.length}}</div>
                </div>
                <div class="card">
                    <div class="card-label">Latest Run</div>
                    <div class="card-value" style="font-size: 1rem;">${{latestRun?.name || 'N/A'}}</div>
                </div>
                <div class="card">
                    <div class="card-label">Mean Reward</div>
                    <div class="card-value">${{latestReward?.toFixed(2) || 'N/A'}}</div>
                </div>
                <div class="card">
                    <div class="card-label">vs Previous</div>
                    <div class="card-value ${{rewardDelta > 0 ? 'positive' : rewardDelta < 0 ? 'negative' : ''}}">
                        ${{rewardDelta ? (rewardDelta > 0 ? '+' : '') + rewardDelta + '%' : 'N/A'}}
                    </div>
                </div>
            `;
        }}

        function initTable() {{
            const tbody = document.getElementById('runs-tbody');
            if (runs.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No runs found</td></tr>';
                return;
            }}

            // Show newest first
            const sortedRuns = [...runs].reverse();

            tbody.innerHTML = sortedRuns.map(run => `
                <tr>
                    <td>${{new Date(run.created_at).toLocaleDateString()}}</td>
                    <td><a href="${{run.url}}" target="_blank">${{run.name}}</a></td>
                    <td>${{run.metrics?.Train?.mean_reward?.toFixed(2) || 'N/A'}}</td>
                    <td>${{run.metrics?.Train?.mean_episode_length?.toFixed(1) || 'N/A'}}</td>
                </tr>
            `).join('');
        }}

        // Initialize
        initSummary();
        initCategories();
        initTable();
    </script>
</body>
</html>
"""


def main():
  parser = argparse.ArgumentParser(
    description="Generate benchmark report from WandB runs"
  )
  parser.add_argument(
    "--run-path",
    type=str,
    action="append",
    dest="run_paths",
    help="Run path (entity/project/run_id). Can be specified multiple times.",
  )
  parser.add_argument(
    "--entity",
    type=str,
    default="rll_humanoid",
    help="WandB entity",
  )
  parser.add_argument(
    "--project",
    type=str,
    default="mjlab",
    help="WandB project name",
  )
  parser.add_argument(
    "--tag",
    type=str,
    help="Filter runs by tag (e.g., 'nightly')",
  )
  parser.add_argument(
    "--limit",
    type=int,
    default=30,
    help="Maximum number of runs to fetch",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("benchmark_results"),
    help="Output directory for generated report",
  )

  args = parser.parse_args()

  if args.run_paths:
    # Fetch specified runs
    runs = []
    for run_path in args.run_paths:
      print(f"Fetching run: {run_path}")
      runs.append(fetch_run_data(run_path))
  else:
    # Fetch from project
    print(f"Fetching runs from {args.entity}/{args.project}")
    if args.tag:
      print(f"Filtering by tag: {args.tag}")
    runs = fetch_project_runs(
      args.entity,
      args.project,
      args.tag,
      args.limit,
    )

  print(f"Found {len(runs)} run(s)")
  generate_html_report(runs, args.output_dir)


if __name__ == "__main__":
  main()
