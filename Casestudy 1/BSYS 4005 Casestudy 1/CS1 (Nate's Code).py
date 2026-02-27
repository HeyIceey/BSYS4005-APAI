from pathlib import Path
import re
import csv
from datetime import datetime
import os

# folder = Path(r"C:\Users\MEHoc\OneDrive\Documents\BCIT\BSYS4005\BSYS4005_CaseStudy1_AIB_SampleData_201501-201502")
# out_file = folder / "parsed_data6.csv"



folder= os.getcwd() # gets current working folder (all 44 files)
outputfile = 'Final Output.csv' # output file name


DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{2}\b")
AMT_CODE_RE = re.compile(r"(?P<amt>\d{1,3}(?:,\d{3})*\.\d{2})\s+(?P<code>[A-Z]{2,4})")

def find_credit_start(lines):
    for line in lines:
        if "EFF DATE" in line and "DEBIT" in line and "CREDIT" in line:
            return line.find("CREDIT")
    return 60

rows = []

for p in folder.glob("*"):
    if not p.is_file():
        continue

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    credit_start = find_credit_start(lines)

    for line in lines:
        date_match = DATE_RE.search(line)
        if not date_match:
            continue

        # data fix *noticed it converted wrong in excel -
        raw_date = date_match.group(0)
        date_obj = datetime.strptime(raw_date, "%m/%d/%y")
        date = date_obj.strftime("%Y-%m-%d")
        

        for m in AMT_CODE_RE.finditer(line):
            amount = float(m.group("amt").replace(",", ""))
            code = m.group("code")

            if m.start() >= credit_start:
                txn_type = "CREDIT"
                amount = -amount     # credit = negative
            else:
                txn_type = "DEBIT"
                amount = amount      # debit = positive

            rows.append([
                date,
                amount,
                txn_type,
                code,
                p.name
            ])

# write CSV
with out_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Date",
        "Amount",
        "Transaction Type",
        "Transaction Code",
        "Source File"
    ])
    writer.writerows(rows)

print(f"CSV created: {out_file}")
print(f"Rows written: {len(rows)}")
