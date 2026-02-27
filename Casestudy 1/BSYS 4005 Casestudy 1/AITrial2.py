import pandas as pd
import re
import os

# 1. Setup - Processes all 44 files in your current VS Code folder
current_dir = os.getcwd() 
output_file = 'CaseStudy1_Final_Data.csv'

# Patterns to find dates and transaction pairs (Number + Code)
date_regex = r'(\d{2}/\d{2}/\d{2})'
# This pattern grabs the number and the 2-3 letter code (like DP, LCP, AS)
trans_regex = r'(\d[\d,.]*)\s+([A-Z]{2,3})'

all_data = []

for filename in os.listdir(current_dir):
    if filename.endswith(".127"):
        current_date = None
        
        with open(os.path.join(current_dir, filename), 'r') as file:
            for line in file:
                # A. Update Date (Fill-Down Logic) [cite: 4]
                date_match = re.search(date_regex, line)
                if date_match:
                    current_date = date_match.group(1)

                # B. Filter for Transaction Lines only 
                if "111-1" in line:
                    # Widened slices to catch the million-dollar amounts [cite: 12, 78]
                    slices = [
                        (line[10:40], "DEBIT"),  # Column 1
                        (line[40:62], "DEBIT"),  # Column 2
                        (line[62:85], "CREDIT"), # Column 3
                        (line[85:],    "CREDIT")  # Column 4
                    ]

                    for text_chunk, trans_type in slices:
                        match = re.search(trans_regex, text_chunk)
                        if match and current_date:
                            # Clean commas and convert to float
                            val = float(match.group(1).replace(',', ''))
                            
                            # C. Signage Rule: Credits must be negative 
                            final_val = val if trans_type == "DEBIT" else -val
                            
                            all_data.append({
                                "Date": current_date,
                                "Transaction Code": match.group(2),
                                "Transaction Type": trans_type,
                                "Amount": final_val
                            })

# 2. Final DataFrame and Export
df = pd.DataFrame(all_data)

# Optional: Sum up transactions by date/code if they repeat [cite: 6]
# df = df.groupby(['Date', 'Transaction Code', 'Transaction Type'], sort=False).sum().reset_index()

df.to_csv(output_file, index=False)
print(f"Extraction complete. {len(df)} transactions found across all files.")