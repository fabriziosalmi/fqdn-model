import json
import argparse
from typing import List, Dict, Any
from rich.progress import track
from rich.console import Console
from rich.table import Table
from collections import defaultdict

def merge_json_data(json_data_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merges a list of JSON data (lists of dictionaries) into a single list of dictionaries.
    Handles potential key collisions and missing keys gracefully.
    """
    merged_data: Dict[str, Dict[str, Any]] = {}

    for json_data in json_data_list:
        if not isinstance(json_data, list):
            raise TypeError("Each element in json_data_list must be a list.")

        for entry in json_data:
            if not isinstance(entry, dict):
                raise TypeError("Each element within the inner lists must be a dictionary.")
            if "FQDN" not in entry:
                raise ValueError("Each dictionary must contain an 'FQDN' key.")

            fqdn = entry["FQDN"]
            if fqdn in merged_data:
                merged_data[fqdn].update(entry)
            else:
                merged_data[fqdn] = entry.copy()

    return list(merged_data.values())



def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Loads JSON data from a file, handling potential errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"The JSON file {filepath} must contain a list of objects.")
            return data
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {filepath}: {e}")
        exit(1)
    except ValueError as e:
        print(f"Error in {filepath}: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        exit(1)



def save_json_file(filepath: str, data: List[Dict[str, Any]]):
    """Saves JSON data to a file, handling potential errors."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")
        exit(1)


def calculate_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates overall statistics from the merged data."""
    stats = {
        "total_fqdns": len(data),
        "value_counts": defaultdict(lambda: {"0": 0, "1": 0, "2": 0})
    }

    for entry in data:
        for key, value in entry.items():
            if key != "FQDN":
                if isinstance(value, int) and str(value) in ("0", "1", "2"):
                    stats["value_counts"][key][str(value)] += 1

    for key, counts in stats["value_counts"].items():
        total_count = sum(counts.values())
        stats["value_counts"][key]["0_percent"] = (counts["0"] / total_count) * 100 if total_count else 0
        stats["value_counts"][key]["1_percent"] = (counts["1"] / total_count) * 100 if total_count else 0
        stats["value_counts"][key]["2_percent"] = (counts["2"] / total_count) * 100 if total_count else 0

    return stats

def display_statistics(stats: Dict[str, Any], console: Console):
    """Displays the calculated statistics in a formatted table."""

    console.print("\n[bold underline]Overall Statistics:[/bold underline]")
    console.print(f"Total FQDNs: [bold]{stats['total_fqdns']}[/bold]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field")
    table.add_column("Count (0)", justify="right")
    table.add_column("Percent (0)", justify="right")
    table.add_column("Count (1)", justify="right")
    table.add_column("Percent (1)", justify="right")
    table.add_column("Count (2)", justify="right")
    table.add_column("Percent (2)", justify="right")

    for field, counts in stats['value_counts'].items():
        table.add_row(
            f"[cyan]{field}[/cyan]",
            f"[green]{counts['0']}[/green]",
            f"[green]{counts['0_percent']:.2f}%[/green]",
            f"[yellow]{counts['1']}[/yellow]",
            f"[yellow]{counts['1_percent']:.2f}%[/yellow]",
            f"[red]{counts['2']}[/red]",
            f"[red]{counts['2_percent']:.2f}%[/red]"
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Merge multiple JSON files containing website data.")
    parser.add_argument('input_files', nargs='+', help='Paths to the input JSON files.')
    parser.add_argument('-o', '--output', required=True, help='Path to the output JSON file.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.  Includes FQDN count.') # Added to description
    parser.add_argument('--show-table', action='store_true', help='Show a table of merged data before saving.')
    parser.add_argument('--show-stats', action='store_true', help='Show overall statistics.')

    args = parser.parse_args()

    console = Console()

    all_json_data = []
    for filename in track(args.input_files, description="Loading files...", console=console):
        all_json_data.append(load_json_file(filename))

    try:
        merged_data = merge_json_data(all_json_data)
    except (TypeError, ValueError) as e:
        console.print(f"[red]Error during merging:[/red] {e}")
        exit(1)

    # FQDN Count Display (outside --verbose)
    console.print(f"Total FQDNs processed: [bold]{len(merged_data)}[/bold]")

    if args.verbose:
        console.print(f"[green]Successfully merged {len(args.input_files)} files.[/green]")
        # Removed redundant FQDN count here

    if args.show_table:
        table = Table(show_header=True, header_style="bold magenta")
        if merged_data:
            for key in merged_data[0].keys():
                table.add_column(key)
            for row in merged_data:
                table.add_row(*[str(value) for value in row.values()])
            console.print(table)
        else:
            console.print("[yellow]No data to display in the table.[/yellow]")

    if args.show_stats:
        stats = calculate_statistics(merged_data)
        display_statistics(stats, console)

    save_json_file(args.output, merged_data)
    console.print(f"[green]Merged data saved to {args.output}[/green]")

if __name__ == "__main__":
    main()