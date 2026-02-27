# BSYS 4005 Applied AIM - Casestudy 1 - January 20, 2026
# CS1-1: Himanish Tripathi, Ho Ngoc An, Bhavay Partap

# Imports
import os # to work with files
import re # for regular expressions (thats what RE stands for)
import pandas as pd # for data storing and CSV writing

# Linking  w current directory/folder + Output file name
folder = os.getcwd() # gets current working folder (all 44 files)
outputfile = 'Final Output.csv' # output file name

# Regex Patterns --------------------------------
# datepattern = r'(\d{2}/\d{2}/\d{2})' # To find dates in the format mm/dd/yy
# transactionpattern = r'(\d[\d,.]*)\s*([A-Z]{2,3})' # To look for numbers & commas & currency codes
# -----------------------------------------------
# Version 2
date = re.compile(r"\b\d{2}/\d{2}/\d{2}\b") #date pattern
amount = re.compile(r"(\d[\d,]*)\s+([A-Z]{2,3})") #transaction pattern
rows = [] # to store all extracted data

# looks for header text, "CREDIT"/"DEBIT" 
def findcreditstart(lines):
    for line in lines:
        if "DEBIT" in line and "CREDIT" in line:
            return line.find("CREDIT")
    # incase nothing found, return a default    
    return 60

# Processing Files -----------------------------------------
for filename in os.listdir(folder):
    if not filename.endswith('.127'):
        continue # skip non .127 files
    # read all files
    with open(os.path.join(folder, filename), 'r', errors="replace") as f:
        lines = f.readlines()  
    # find credit vals (where they start in each file)
    findcredit = findcreditstart(lines)
    # this stores recent - (Filling down logic)
    current_date = None

# Processing Lines -----------------------------------------
    for line in lines:
        # 1 Date
        date_match = date.search(line)
        if date_match:
            current_date = date_match.group()
        # if no date
        if current_date is None:
            continue
        # Transaction (ONLY)
        if "111-1" not in line:
            continue
        # 2 Amounts + Codes 
        for match in amount.finditer(line):
            amnt = float(match.group(1).replace(",","")) # extract amount
            code = match.group(2) # extract the currency code
            # 3 Credit & Debit
            if match.start() >= findcredit:
                txn_type = "CREDIT"
                amnt = -amnt # credit as negative value
            else:
                txn_type = "DEBIT"
        # 4 Save & Store
        rows.append({
            "Date": current_date,
            "Amount": amount,
            "Transaction Type": txn_type,
            "Transaction Code": code,
            "Source File": filename
        })

# Summarize & Export CSV ------------------------------
df = pd.DataFrame(rows)
summary_df = (
    df.groupby(
        ['Date', 'Transaction Code', 'Transaction Type'], 
        as_index=False
    )['Amount'].sum()
)
summary_df.to_csv(outputfile, index=False)
print("Done! File Created: Final Output.csv")
print("Summarized Rows:", len(summary_df))







'''
# Loop & Read Files ------------------------------
# looks for files with .127 extension in current folder/directory
for filename in os.listdir(folder):
    if filename.endswith('.127'): #127 extension - itll look for the transactions files nicely
        
        current_date = None

        with open(os.path.join(folder, filename), 'r') as file:
            for line in file:

                #1 find + store date (fill it down) [AI HELPED]
                date_match = re.search(datepattern, line)
                # if date is matched, its stored
                if date_match:
                    current_date = date_match.group(1) # update current date

                #2 transaction lines only
                if '111-1' in line:
                    # the transaction areas? hard to be certain abt the #
                    slices = [
                        (line[10:40], "DEBIT"),  # Column 1
                        (line[40:62], "DEBIT 2"),  # Column 2
                        (line[62:85], "CREDIT"), # Column 3
                        (line[85:],    "CREDIT 2")  # Column 4
                    ]
'''