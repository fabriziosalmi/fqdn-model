import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def process_batch(batch_df, other_df):
    """
    Processes a batch of FQDNs, prioritizing 'Is_Bad' = 0.

    Args:
        batch_df: DataFrame containing a batch of FQDNs.
        other_df: The DataFrame to merge against (prioritized for 'Is_Bad' = 0).

    Returns:
        A DataFrame containing the processed batch.
    """
    # Concatenate the batch with the other DataFrame
    combined_df = pd.concat([batch_df, other_df], ignore_index=True)

    # Sort by FQDN and Is_Bad (ascending) to prioritize Is_Bad = 0
    combined_df = combined_df.sort_values(by=['FQDN', 'Is_Bad'])

    # Drop duplicates, keeping the first occurrence (Is_Bad = 0 if available)
    return combined_df.drop_duplicates(subset='FQDN', keep='first')

def merge_csv_prioritize_good_parallel(file1_path, file2_path, output_path, batch_size=1000, num_workers=None):
    """
    Merges two CSV files, prioritizing rows where 'Is_Bad' is 0,
    using parallel processing and displaying progress.

    Args:
        file1_path: Path to the first CSV file.
        file2_path: Path to the second CSV file.
        output_path: Path to save the merged CSV file.
        batch_size: Number of rows to process in each batch.
        num_workers: Number of worker processes to use (defaults to number of CPU cores).
    """
    try:
        # Read the smaller CSV file into memory (assuming file2 is smaller)
        # We'll use this one to prioritize 'Is_Bad' = 0 during merging
        df2 = pd.read_csv(file2_path)
        df2 = df2.sort_values(by=['FQDN', 'Is_Bad'])  # Pre-sort df2 for efficiency
        df2 = df2.drop_duplicates(subset='FQDN', keep='first') # Drop duplicates for best performace

        # Get the total number of rows in the larger file (file1) for the progress bar
        with open(file1_path, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # Subtract 1 for the header row
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            print(Fore.GREEN + "Starting parallel merge..." + Style.RESET_ALL)

            # Create an iterator for reading file1 in batches
            for chunk in pd.read_csv(file1_path, chunksize=batch_size):
                # Submit each batch for processing
                future = executor.submit(process_batch, chunk, df2)
                futures.append(future)

            # Collect results with a progress bar
            results = []
            with tqdm(total=total_rows, desc=Fore.CYAN + "Merging" + Style.RESET_ALL, unit="rows", colour="green") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(len(result))  # Update based on processed rows in the result
                    except Exception as e:
                        print(Fore.RED + f"Error processing batch: {e}" + Style.RESET_ALL)

            # Concatenate all processed batches
            print(Fore.GREEN + "Concatenating results..." + Style.RESET_ALL)
            df_merged = pd.concat(results, ignore_index=True)
            df_merged = df_merged.sort_values(by=['FQDN', 'Is_Bad'])
            df_merged = df_merged.drop_duplicates(subset='FQDN', keep='first')

            # Save the merged DataFrame to a new CSV file
            df_merged.to_csv(output_path, index=False)
            print(Fore.GREEN + f"Files merged successfully. Output saved to: {output_path}" + Style.RESET_ALL)

    except FileNotFoundError:
        print(Fore.RED + "Error: One or both of the input CSV files were not found." + Style.RESET_ALL)
    except pd.errors.EmptyDataError:
        print(Fore.RED + "Error: One or both of the input CSV files are empty." + Style.RESET_ALL)
    except pd.errors.ParserError:
        print(Fore.RED + "Error: There was a problem parsing one of the CSV files.  Check for formatting issues." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred: {e}" + Style.RESET_ALL)



def main():
    parser = argparse.ArgumentParser(description="Merge two CSV files, prioritizing rows with 'Is_Bad' = 0.")
    parser.add_argument("file1", help="Path to the first CSV file (larger file).")
    parser.add_argument("file2", help="Path to the second CSV file (smaller file).")
    parser.add_argument("output", help="Path to save the merged CSV file.")
    parser.add_argument("-b", "--batch_size", type=int, default=1000, help="Batch size for processing (default: 1000).")
    parser.add_argument("-n", "--num_workers", type=int, default=None, help="Number of worker processes (default: number of CPU cores).")
    args = parser.parse_args()

    merge_csv_prioritize_good_parallel(args.file1, args.file2, args.output, args.batch_size, args.num_workers)

if __name__ == "__main__":
    main()