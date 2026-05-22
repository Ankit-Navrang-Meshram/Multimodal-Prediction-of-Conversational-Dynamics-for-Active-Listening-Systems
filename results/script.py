import os
import pandas as pd
import re

# Path to your results directory
RESULTS_DIR = "./"   # change if needed

data = []

def extract_metrics(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    result = {}

    # Model name from file
    filename = os.path.basename(file_path)
    result["file_name"] = filename

    # Extract fusion type (e.g., Cross_Modal_Attention)
    result["model"] = filename.split("_no")[0].replace("_results.txt", "").replace("_", " ")

    # Extract ablation
    ablation_match = re.search(r"Ablation: (.*)", content)
    result["ablation"] = ablation_match.group(1) if ablation_match else "None"

    # Overall Accuracy
    acc_match = re.search(r"Overall Accuracy:\s*([0-9.]+)", content)
    result["accuracy"] = float(acc_match.group(1)) if acc_match else None

    # Macro metrics
    result["macro_precision"] = float(re.search(r"Macro-Averaged Metrics:.*?Precision:\s*([0-9.]+)", content, re.S).group(1))
    result["macro_recall"] = float(re.search(r"Macro-Averaged Metrics:.*?Recall:\s*([0-9.]+)", content, re.S).group(1))
    result["macro_f1"] = float(re.search(r"Macro-Averaged Metrics:.*?F1-Score:\s*([0-9.]+)", content, re.S).group(1))

    # Weighted metrics
    result["weighted_precision"] = float(re.search(r"Weighted-Averaged Metrics:.*?Precision:\s*([0-9.]+)", content, re.S).group(1))
    result["weighted_recall"] = float(re.search(r"Weighted-Averaged Metrics:.*?Recall:\s*([0-9.]+)", content, re.S).group(1))
    result["weighted_f1"] = float(re.search(r"Weighted-Averaged Metrics:.*?F1-Score:\s*([0-9.]+)", content, re.S).group(1))

    # Per-class metrics
    classes = ["keep", "turn", "bc"]
    for cls in classes:
        pattern = rf"{cls}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
        match = re.search(pattern, content)
        if match:
            result[f"{cls}_precision"] = float(match.group(1))
            result[f"{cls}_recall"] = float(match.group(2))
            result[f"{cls}_f1"] = float(match.group(3))

    return result


# Iterate over all files
for file in os.listdir(RESULTS_DIR):
    if file.endswith(".txt"):
        file_path = os.path.join(RESULTS_DIR, file)
        try:
            metrics = extract_metrics(file_path)
            data.append(metrics)
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Create DataFrame
df = pd.DataFrame(data)

# Sort (optional)
df = df.sort_values(by=["model", "ablation"])

# Save to Excel
output_file = "all_results.xlsx"
df.to_excel(output_file, index=False)

print(f"✅ Excel file created: {output_file}")