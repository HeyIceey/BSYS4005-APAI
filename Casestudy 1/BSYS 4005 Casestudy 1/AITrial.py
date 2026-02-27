# -------------------------------
# IMPORT REQUIRED LIBRARIES
# -------------------------------

# pandas is used to store data in table form and export to CSV
import pandas as pd

# re (regular expressions) helps us find patterns like dates and amounts
import re

# os lets us work with files and folders
import os


# -------------------------------
# BASIC SETUP
# -------------------------------

# Get the folder where this Python file is currently running
# This assumes your .127 files are in the same folder
current_dir = os.getcwd()

# Name of the final output CSV file
output_file = "CaseStudy1_Final_Data.csv"


# -------------------------------
# PATTERNS WE WANT TO FIND
# -------------------------------

# This pattern matches dates like 01/15/15
date_pattern = r"\d{2}/\d{2}/\d{2}"

# This pattern matches:
# - a number (with optional commas)
# - followed by a transaction code (2–3 letters)
# Example: 12,345 DP
amount_code_pattern = r"(\d[\d,]*)\s+([A-Z]{2,3})"


# -------------------------------
# PLACE TO STORE ALL TRANSACTIONS
# -------------------------------

# We will store each extracted transaction as a dictionary
rows = []


# -------------------------------
# LOOP THROUGH ALL FILES
# -------------------------------

# Look at every file in the current folder
for filename in os.listdir(current_dir):

    # Only process files that end with .127
    if not filename.endswith(".127"):
        continue   # skip everything else

    # This variable will hold the most recent date we saw
    # (used for fill-down logic)
    current_date = None

    # Create the full path to the file
    file_path = os.path.join(current_dir, filename)

    # Open the file so we can read it line by line
    with open(file_path, "r") as file:

        # Read each line in the file
        for line in file:

            # -------------------------------
            # STEP 1: CHECK IF THE LINE HAS A DATE
            # -------------------------------

            # Search the line for a date
            date_match = re.search(date_pattern, line)

            # If a date is found, store it
            if date_match:
                current_date = date_match.group()

            # -------------------------------
            # STEP 2: CHECK IF THIS IS A TRANSACTION LINE
            # -------------------------------

            # We only care about lines that contain "111-1"
            # Also, if we don't have a date yet, skip
            if "111-1" not in line or current_date is None:
                continue


            # -------------------------------
            # STEP 3: SPLIT THE LINE INTO FIXED COLUMNS
            # -------------------------------

            # The file uses fixed-width columns
            # We manually slice the line into debit and credit sections
            columns = [
                (line[10:40], "DEBIT"),   # First debit column
                (line[40:62], "DEBIT"),   # Second debit column
                (line[62:85], "CREDIT"),  # First credit column
                (line[85:],   "CREDIT")   # Second credit column
            ]


            # -------------------------------
            # STEP 4: EXTRACT DATA FROM EACH COLUMN
            # -------------------------------

            # Go through each column we just defined
            for text, txn_type in columns:

                # Look for an amount + transaction code in the column
                match = re.search(amount_code_pattern, text)

                # If nothing is found, move to the next column
                if not match:
                    continue

                # Extract the amount and remove commas
                amount = float(match.group(1).replace(",", ""))

                # Extract the transaction code (DP, AS, etc.)
                code = match.group(2)

                # -------------------------------
                # STEP 5: APPLY BUSINESS RULES
                # -------------------------------

                # Credits must be negative
                if txn_type == "CREDIT":
                    amount = -amount


                # -------------------------------
                # STEP 6: STORE THE TRANSACTION
                # -------------------------------

                # Save the transaction as a dictionary
                rows.append({
                    "Date": current_date,
                    "Transaction Code": code,
                    "Transaction Type": txn_type,
                    "Amount": amount
                })


# -------------------------------
# STEP 7: CREATE A TABLE (DATAFRAME)
# -------------------------------

# Convert the list of dictionaries into a pandas DataFrame
df = pd.DataFrame(rows)


# -------------------------------
# STEP 8: SUMMARIZE BY DAY
# -------------------------------

# Group transactions by Date, Code, and Type
# Then sum the amounts
summary_df = (
    df.groupby(
        ["Date", "Transaction Code", "Transaction Type"],
        as_index=False
    )["Amount"]
    .sum()
)


# -------------------------------
# STEP 9: EXPORT FINAL CSV
# -------------------------------

# Save the summarized table to a CSV file
summary_df.to_csv(output_file, index=False)

# Print confirmation
print("Processing complete.")
print(f"Final file created: {output_file}")
print(f"Number of summarized rows: {len(summary_df)}")
