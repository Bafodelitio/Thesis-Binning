import os
import pandas as pd


def process_quality_reports(
    base_directory="E:/FCT/Tese/Code/checkm2/checkm2/tests/adapted_pca",#"/mnt/mydata/checkm2/checkm2/tests/adapted_pca",
    excel_filename="hyperparameter_tuning_results.xlsx",
    recount_all=False
):
    # Load or create the Excel file
    excel_directory = os.path.join(base_directory,excel_filename)
    if os.path.exists(excel_directory):
        results_df = pd.read_excel(excel_directory)
    else:
        print(f"Excel file '{excel_filename}' not found.")
        return

    # Add columns for "Complete" and "Contaminated" if they do not exist
    if "Complete" not in results_df.columns:
        results_df["Complete"] = 0

    if "Contaminated" not in results_df.columns:
        results_df["Contaminated"] = 0
    
    if "HQ Contigs" not in results_df.columns:
        results_df["HQ Contigs"] = 0
    # Iterate over each subdirectory in the base directory
    for folder_name in os.listdir(base_directory):
        if folder_name != excel_filename:
            folder_path = os.path.join(base_directory, folder_name, "checkm2_result")
            quality_report_path = os.path.join(folder_path, "quality_report.tsv")

            # Check if the quality_report.tsv file exists
            if not os.path.isfile(quality_report_path):
                if folder_name != "losses":
                    print(f"quality_report.tsv not found in {folder_path}")
                continue

            # Find the row in the Excel file that matches the folder name
            row_index = results_df[results_df["Folder Name"] == folder_name].index

            # Check if we should recount everything or only empty columns
            if not results_df[results_df["Folder Name"] == folder_name].empty:
                if not recount_all and results_df.at[row_index[0], "Complete"] != 0 and results_df.at[row_index[0], "Contaminated"] != 0:
                    print(f"Skipping folder '{folder_name}' as it already has counts.")
                    continue     

            # Read the quality_report.tsv file
            try:
                quality_report_df = pd.read_csv(quality_report_path, sep='\t')
            except Exception as e:
                print(f"Error reading {quality_report_path}: {e}")
                continue

            # Count the "Complete" and "Contaminated" rows
            complete_count = quality_report_df[quality_report_df['Completeness'] > 90].shape[0]
            contaminated_count = quality_report_df[quality_report_df['Contamination'] > 5].shape[0]
            hq_count = quality_report_df[(quality_report_df['Completeness'] > 90) & (quality_report_df['Contamination'] < 5)].shape[0]

            
            # If matching row found, update the values
            if not row_index.empty:
                results_df.at[row_index[0], "Complete"] = complete_count
                results_df.at[row_index[0], "Contaminated"] = contaminated_count
                results_df.at[row_index[0], "HQ Contigs"] = hq_count
                print(f"COUNTED '{folder_name}'")
            else:
                print(f"No matching folder name '{folder_name}' found in the Excel file.")

    # Save the updated Excel file
    results_df.to_excel(excel_directory, index=False)
    print(f"Updated Excel file saved as '{excel_directory}'")
    
process_quality_reports(recount_all=True)