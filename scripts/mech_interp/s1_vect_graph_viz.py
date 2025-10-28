import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--jsonl_file", type=str, default=None, help="Path to the JSONL file with projection data")
parser.add_argument("--base_folder", type=str, default=None, help="Base folder where subfolders will be created for each line")
parser.add_argument("--limit", type=int, default=None, help="Number of lines to process (None for all lines)")
parser.add_argument("--show_plots", type=bool, default=False, help="Whether to display plots (not recommended for many lines)")
parser.add_argument("--detailed_stats", type=bool, default=False, help="Whether to print detailed statistics for each line")
args = parser.parse_args()

def parse_single_jsonl_line(line: str) -> Tuple[List[List[float]], List[List[float]], List[str]]:
    """
    Parse a single JSONL line containing sentence projection data.

    Args:
        line: A single JSON line string

    Returns:
        Tuple of (compliance_projections, harmfulness_projections, sentence_texts)
    """
    data = json.loads(line.strip())

    # Extract sentence differences data
    sentence_diffs = data.get("sentence_diffs", {})
    compliance_data = sentence_diffs.get("compliance", {})
    harmfulness_data = sentence_diffs.get("harmfulness", {})

    # Extract projections for each sentence
    compliance_projections = []
    harmfulness_projections = []
    sentence_texts = []

    # Process each sentence in the data
    for sentence, comp_values in compliance_data.items():
        if sentence in harmfulness_data:
            harm_values = harmfulness_data[sentence]

            compliance_projections.append(comp_values)
            harmfulness_projections.append(harm_values)
            sentence_texts.append(sentence)

    return compliance_projections, harmfulness_projections, sentence_texts

def get_jsonl_line_count(file_path: str) -> int:
    """
    Count the number of lines in a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        Number of lines in the file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)

def calculate_averages(projections: List[List[float]]) -> List[float]:
    """
    Calculate average of each projection vector.

    Args:
        projections: List of projection vectors

    Returns:
        List of average values
    """
    return [np.mean(proj) for proj in projections]

def validate_data_consistency(compliance_sentences: List[str],
                            harmfulness_sentences: List[str]) -> bool:
    """
    Validate that both datasets have the same sentences in the same order.

    Args:
        compliance_sentences: Sentences from compliance data
        harmfulness_sentences: Sentences from harmfulness data

    Returns:
        True if data is consistent, False otherwise
    """
    if len(compliance_sentences) != len(harmfulness_sentences):
        print(f"Warning: Different number of sentences - Compliance: {len(compliance_sentences)}, Harmfulness: {len(harmfulness_sentences)}")
        return False

    for i, (comp_sent, harm_sent) in enumerate(zip(compliance_sentences, harmfulness_sentences)):
        if comp_sent != harm_sent:
            print(f"Warning: Sentence {i} differs between datasets:")
            print(f"  Compliance: {comp_sent[:100]}...")
            print(f"  Harmfulness: {harm_sent[:100]}...")

    return True

def create_visualization(compliance_avgs: List[float],
                        harmfulness_avgs: List[float],
                        sentence_texts: List[str],
                        title: str = "Compliance vs Harmfulness Analysis",
                        figsize: Tuple[int, int] = (14, 8)):
    """
    Create matplotlib visualization of the projection data.

    Args:
        compliance_avgs: List of compliance averages
        harmfulness_avgs: List of harmfulness averages
        sentence_texts: List of sentence texts
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create x-axis labels
    x_labels = [f'S{i}' for i in range(len(compliance_avgs))]
    x_positions = range(len(compliance_avgs))

    # Plot the lines
    ax.plot(x_positions, compliance_avgs, 'o-', color='#2196F3',
            linewidth=2, markersize=6, label='Compliance', alpha=0.8)
    ax.plot(x_positions, harmfulness_avgs, 's-', color='#F44336',
            linewidth=2, markersize=6, label='Harmfulness', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('Sentence Number', fontsize=12)
    ax.set_ylabel('Average Projection Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 10 else 0)

    # Add some styling
    plt.tight_layout()

    return fig

def print_statistics(compliance_avgs: List[float], harmfulness_avgs: List[float]):
    """
    Print statistical summary of the data.

    Args:
        compliance_avgs: List of compliance averages
        harmfulness_avgs: List of harmfulness averages
    """
    print(f"\nStatistics:")
    print("=" * 50)
    print(f"Number of sentences: {len(compliance_avgs)}")
    print(f"Average Compliance: {np.mean(compliance_avgs):.3f}")
    print(f"Average Harmfulness: {np.mean(harmfulness_avgs):.3f}")
    print(f"Peak Compliance: {np.max(compliance_avgs):.3f} (Sentence {np.argmax(compliance_avgs)})")
    print(f"Lowest Compliance: {np.min(compliance_avgs):.3f} (Sentence {np.argmin(compliance_avgs)})")
    print(f"Peak Harmfulness: {np.max(harmfulness_avgs):.3f} (Sentence {np.argmax(harmfulness_avgs)})")
    print(f"Lowest Harmfulness: {np.min(harmfulness_avgs):.3f} (Sentence {np.argmin(harmfulness_avgs)})")

    # Correlation analysis
    if len(compliance_avgs) > 1:
        correlation = np.corrcoef(compliance_avgs, harmfulness_avgs)[0, 1]
        print(f"Correlation between compliance and harmfulness: {correlation:.3f}")

def print_sentence_summary(sentence_texts: List[str], max_length: int = 80):
    """
    Print a summary of all sentences with truncation.

    Args:
        sentence_texts: List of sentence texts
        max_length: Maximum length for each sentence display
    """
    print("\nSentence Summary:")
    print("=" * 50)
    for i, sentence in enumerate(sentence_texts):
        # Truncate if too long
        if len(sentence) > max_length:
            sentence = sentence[:max_length-3] + "..."

        print(f"S{i:2d}: {sentence}")

def save_figure(fig, figure_folder: str, filename: str = "compliance_harmfulness_analysis.png"):
    """
    Save the figure to the specified folder.

    Args:
        fig: matplotlib figure object
        figure_folder: Path to the folder where the figure should be saved
        filename: Name of the file to save
    """
    # Create folder if it doesn't exist
    Path(figure_folder).mkdir(parents=True, exist_ok=True)

    # Construct full path
    full_path = os.path.join(figure_folder, filename)

    # Save figure with high quality
    fig.savefig(full_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Figure saved to: {full_path}")

def analyze_patterns(compliance_avgs: List[float], harmfulness_avgs: List[float]):
    """
    Analyze patterns in the data and print insights.

    Args:
        compliance_avgs: List of compliance averages
        harmfulness_avgs: List of harmfulness averages
    """
    print("\nPattern Analysis:")
    print("=" * 50)

    # Find phase changes (large differences between consecutive points)
    if len(compliance_avgs) > 1:
        compliance_diffs = np.diff(compliance_avgs)
        harmfulness_diffs = np.diff(harmfulness_avgs)

        large_compliance_changes = np.where(np.abs(compliance_diffs) > np.std(compliance_diffs) * 2)[0]
        large_harmfulness_changes = np.where(np.abs(harmfulness_diffs) > np.std(harmfulness_diffs) * 2)[0]

        print(f"Large compliance changes at sentences: {large_compliance_changes + 1}")
        print(f"Large harmfulness changes at sentences: {large_harmfulness_changes + 1}")

def save_data_to_csv(compliance_avgs: List[float],
                     harmfulness_avgs: List[float],
                     sentence_texts: List[str],
                     figure_folder: str,
                     filename: str = "projection_analysis.csv"):
    """
    Save the analysis data to a CSV file in the figure folder.

    Args:
        compliance_avgs: List of compliance averages
        harmfulness_avgs: List of harmfulness averages
        sentence_texts: List of sentence texts
        figure_folder: Folder to save the CSV
        filename: CSV filename
    """
    try:
        import pandas as pd

        # Create folder if it doesn't exist
        Path(figure_folder).mkdir(parents=True, exist_ok=True)

        # Calculate harmfulness changes (differences between consecutive values)
        harmfulness_changes = []
        for i in range(len(harmfulness_avgs)):
            if i == 0:
                # First sentence has no previous value, so change is 0
                harmfulness_changes.append(0.0)
            else:
                # Calculate change from previous sentence (rounded to 2 significant figures)
                change = harmfulness_avgs[i] - harmfulness_avgs[i-1]
                # Round to 2 significant figures
                if change != 0:
                    import math
                    rounded_change = round(change, -int(math.floor(math.log10(abs(change)))) + 1)
                else:
                    rounded_change = 0.0
                harmfulness_changes.append(rounded_change)

        df = pd.DataFrame({
            'sentence_id': range(len(compliance_avgs)),
            'compliance_avg': compliance_avgs,
            'harmfulness_avg': harmfulness_avgs,
            'sentence_text': sentence_texts,
            'harmfulness_change': harmfulness_changes
        })

        csv_path = os.path.join(figure_folder, filename)
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

    except ImportError:
        print("pandas not available. Skipping CSV export.")
        print("Install with: pip install pandas")

def process_single_line(line: str, line_number: int, jsonl_subfolder: str, show_plot: bool = False):
    """
    Process a single JSONL line and save results to a subfolder.

    Args:
        line: The JSONL line to process
        line_number: Line number (0-based)
        jsonl_subfolder: Subfolder based on JSONL filename where prompt subfolders will be created
        show_plot: Whether to display the plot

    Returns:
        Tuple of (figure, compliance_averages, harmfulness_averages, sentence_texts)
    """
    # Create subfolder for this prompt
    subfolder = os.path.join(jsonl_subfolder, f"prompt_{line_number:03d}")
    Path(subfolder).mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing prompt {line_number}...")
    print(f"Output folder: {subfolder}")

    # Parse the single line
    compliance_projections, harmfulness_projections, sentence_texts = parse_single_jsonl_line(line)

    if not compliance_projections:
        print(f"No data found in prompt {line_number}, skipping...")
        return None, None, None, None

    # Calculate averages
    compliance_avgs = calculate_averages(compliance_projections)
    harmfulness_avgs = calculate_averages(harmfulness_projections)

    print(f"Processed {len(compliance_avgs)} sentences")

    # Create visualization
    title = f"Compliance vs Harmfulness Analysis - Prompt {line_number} ({len(sentence_texts)} sentences)"
    fig = create_visualization(compliance_avgs, harmfulness_avgs, sentence_texts, title)

    # Save figure
    save_figure(fig, subfolder, filename=f"analysis_prompt_{line_number:03d}.png")

    # Save CSV data
    save_data_to_csv(compliance_avgs, harmfulness_avgs, sentence_texts, subfolder,
                    filename=f"analysis_prompt_{line_number:03d}.csv")

    # Print brief statistics
    print(f"  - Average Compliance: {np.mean(compliance_avgs):.3f}")
    print(f"  - Average Harmfulness: {np.mean(harmfulness_avgs):.3f}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # Close figure to save memory

    return fig, compliance_avgs, harmfulness_avgs, sentence_texts

def main(jsonl_file: str, base_folder: str, limit: int = None,
         show_plots: bool = False, detailed_stats: bool = False):
    """
    Main function to process each line of the JSONL file separately.

    Args:
        jsonl_file: Path to the JSONL file with projection data
        base_folder: Base folder where subfolders will be created for each line
        limit: Number of lines to process (None for all lines)
        show_plots: Whether to display plots (not recommended for many lines)
        detailed_stats: Whether to print detailed statistics for each line

    Returns:
        List of results for each processed line
    """
    print(f"Processing JSONL file: {jsonl_file}")

    # Extract filename from JSONL file path (without extension)
    jsonl_filename = os.path.splitext(os.path.basename(jsonl_file))[0]
    print(f"JSONL filename: {jsonl_filename}")

    # Create subfolder based on JSONL filename
    jsonl_subfolder = os.path.join(base_folder, jsonl_filename)
    Path(jsonl_subfolder).mkdir(parents=True, exist_ok=True)
    print(f"Created subfolder: {jsonl_subfolder}")

    # Get total line count
    total_lines = get_jsonl_line_count(jsonl_file)
    print(f"Total lines in file: {total_lines}")

    if limit is not None:
        lines_to_process = min(limit, total_lines)
        print(f"Processing first {lines_to_process} lines")
    else:
        lines_to_process = total_lines
        print(f"Processing all {lines_to_process} lines")

    results = []

    # Process each line
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file):
            if limit is not None and line_number >= limit:
                break

            try:
                result = process_single_line(line, line_number, jsonl_subfolder, show_plots)
                results.append((line_number, result))

                if detailed_stats and result[0] is not None:
                    fig, compliance_avgs, harmfulness_avgs, sentence_texts = result
                    print(f"\nDetailed Statistics for Prompt {line_number}:")
                    print_statistics(compliance_avgs, harmfulness_avgs)
                    analyze_patterns(compliance_avgs, harmfulness_avgs)
                    print_sentence_summary(sentence_texts, max_length=60)

            except Exception as e:
                print(f"Error processing prompt {line_number}: {e}")
                results.append((line_number, (None, None, None, None)))

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Processed {len(results)} prompts")
    print(f"Results saved in: {jsonl_subfolder}")

    # Print summary
    successful_results = [r for r in results if r[1][0] is not None]
    print(f"Successful analyses: {len(successful_results)}")

    if successful_results:
        all_compliance_avgs = []
        all_harmfulness_avgs = []
        for _, (_, comp_avgs, harm_avgs, _) in successful_results:
            if comp_avgs and harm_avgs:
                all_compliance_avgs.extend(comp_avgs)
                all_harmfulness_avgs.extend(harm_avgs)

        if all_compliance_avgs and all_harmfulness_avgs:
            print(f"\nOverall Summary Across All Prompts:")
            print(f"Total sentences processed: {len(all_compliance_avgs)}")
            print(f"Overall average compliance: {np.mean(all_compliance_avgs):.3f}")
            print(f"Overall average harmfulness: {np.mean(all_harmfulness_avgs):.3f}")

    return results
# Utility function for processing all lines
def process_all_lines(jsonl_file: str, base_folder: str):
    """
    Convenience function to process all lines in the JSONL file.

    Args:
        jsonl_file: Path to the JSONL file
        base_folder: Base folder for output subfolders
    """
    return main(jsonl_file, base_folder, limit=None, show_plots=False, detailed_stats=False)

# Utility function for processing with detailed output
def process_with_details(jsonl_file: str, base_folder: str, limit: int = 5):
    """
    Convenience function to process lines with detailed statistics output.

    Args:
        jsonl_file: Path to the JSONL file
        base_folder: Base folder for output subfolders
        limit: Number of lines to process
    """
    return main(jsonl_file, base_folder, limit=limit, show_plots=False, detailed_stats=True)

########### MAIN FUNCTION ###########
# Example usage
if __name__ == "__main__":
    
    # File paths - updated to process the specific JSONL file
    jsonl_file = args.jsonl_file # <--- update this (after running s1_vect_graph-final.sh)
    base_folder = args.base_folder # <--- update this (folder to store the result figure and data)

    try:
        # Run the analysis - process each line separately
        results = main(
            jsonl_file=jsonl_file,
            base_folder=base_folder,
            limit=args.limit,  # Process only first 5 lines as requested (set to None for all lines)
            show_plots=args.show_plots,  # Set to True if you want to see plots (not recommended for many lines)
            detailed_stats=args.detailed_stats  # Set to True for detailed statistics per line
        )

        print(f"\nAnalysis complete! Check the '{base_folder}' folder for saved files.")
        print(f"Results are organized as: {base_folder}/sentence_projection-<jsonl_file_name>-<timestamp>/")
        print(f"Each prompt has been processed into its own subfolder (prompt_000, prompt_001, etc.)")
        print(f"Each subfolder contains:")
        print(f"  - analysis_prompt_XXX.png (visualization)")
        print(f"  - analysis_prompt_XXX.csv (data with harmfulness_change column)")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check that the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

