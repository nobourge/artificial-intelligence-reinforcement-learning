import os
import pdfplumber
from difflib import unified_diff

from extract_pdf_txt import extract_text_from_pdf

def compare_pdfs(pdf1_path, pdf2_path):
    pdf1_text = extract_text_from_pdf(pdf1_path)
    pdf2_text = extract_text_from_pdf(pdf2_path)

    pdf1_lines = pdf1_text.splitlines()
    pdf2_lines = pdf2_text.splitlines()

    diff = list(unified_diff(pdf1_lines, pdf2_lines, lineterm=""))

    return "\n".join(diff)


DIRECTORY_TO_WATCH = os.path.dirname(os.path.abspath(__file__))

print("DIRECTORY_TO_WATCH:", DIRECTORY_TO_WATCH)
# Replace with your PDF file paths
pdf1_path = r"doc\in\2023-2024 projet RL-1.pdf"
pdf2_path = r"doc\in\2023-2024 projet RL-v2.pdf"
differences = compare_pdfs(pdf1_path, pdf2_path)
print(differences)

pdf1_name = os.path.basename(pdf1_path)
pdf2_name = os.path.basename(pdf2_path)
diff_filename = f"{pdf1_name}_vs_{pdf2_name}_diff.txt"
diff_path = os.path.join(DIRECTORY_TO_WATCH, diff_filename)
with open(diff_path, "w", encoding="utf-8") as diff_file:
    diff_file.write(differences)