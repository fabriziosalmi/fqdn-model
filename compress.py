import joblib
import argparse
import os  # For checking file existence
from rich.console import Console
from rich.panel import Panel
from rich.style import Style

console = Console()


def compress_joblib_model(input_file, output_file=None, compression_level=3, overwrite=False):
    """
    Compresses a joblib model using joblib.dump with rich visualization.

    Args:
        input_file (str): Path to the input joblib model file.
        output_file (str, optional): Path to the output compressed joblib model file.
            If None, defaults to input_file_compressed.joblib.
        compression_level (int, optional): Compression level to use (0-9). Defaults to 3.
        overwrite (bool, optional): Whether to overwrite the output file if it already exists. Defaults to False.

    Raises:
        FileNotFoundError: If the input file does not exist.
        FileExistsError: If the output file already exists and overwrite is False.
        ValueError: If the compression level is not within the valid range (0-9).
        Exception: For other errors during loading or saving.
    """

    # Input validation
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if compression_level not in range(0, 10):
        raise ValueError(f"Compression level must be between 0 and 9 (inclusive), but got {compression_level}")

    if output_file is None:
        output_file = input_file.replace(".joblib", "_compressed.joblib")
        if output_file == input_file:
          output_file += "_compressed.joblib"  # Handle case with no .joblib extension

    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_file}. Use --overwrite to allow overwriting.")

    try:
        # Load the model
        with console.status("[bold blue]Loading model...[/]") as status:
            model = joblib.load(input_file)
            input_size = os.path.getsize(input_file)
            status.update("[bold green]Model loaded successfully![/]")

        # Dump the model with compression
        with console.status(f"[bold blue]Compressing model with level {compression_level}...[/]") as status:
            joblib.dump(model, output_file, compress=compression_level)
            output_size = os.path.getsize(output_file)
            status.update("[bold green]Model compressed successfully![/]")

        # Calculate compression ratio
        compression_ratio = (1 - (output_size / input_size)) * 100

        # Print summary using rich
        panel_content = f"""
        [bold]Input File:[/bold]  [yellow]{input_file}[/]
        [bold]Output File:[/bold] [yellow]{output_file}[/]
        [bold]Compression Level:[/bold] [green]{compression_level}[/]
        [bold]Original Size:[/bold] [cyan]{input_size:,} bytes[/]
        [bold]Compressed Size:[/bold] [cyan]{output_size:,} bytes[/]
        [bold]Compression Ratio:[/bold] [magenta]{compression_ratio:.2f}%[/]
        """
        panel = Panel(panel_content, title="[bold green]Compression Summary[/]", border_style="green")
        console.print(panel)

    except Exception as e:
        console.print_exception(show_locals=True)  # Print exception with locals for debugging
        raise Exception(f"An error occurred during compression: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compress a joblib model with rich visualization.")
    parser.add_argument("input_file", help="Path to the input joblib model file.")
    parser.add_argument(
        "-o", "--output_file", help="Path to the output compressed joblib model file. Defaults to input_file_compressed.joblib",
        default=None
    )
    parser.add_argument(
        "-c", "--compression_level", type=int, default=3, help="Compression level to use (0-9). Defaults to 3."
    )
    parser.add_argument(
        "-w", "--overwrite", action="store_true", help="Overwrite the output file if it already exists."
    )

    args = parser.parse_args()

    try:
        compress_joblib_model(
            args.input_file, args.output_file, args.compression_level, args.overwrite
        )
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        console.print(f"[bold red]Error:[/] {e}")  # Print specific error message for known errors
    except Exception as e:
        console.print_exception(show_locals=True)
        console.print(f"[bold red]An unexpected error occurred:[/] {e}") # Catch all other errors


if __name__ == "__main__":
    main()