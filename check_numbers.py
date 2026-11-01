"""Number check (PLAN.md, step 6.2): every number in the exported report must come from a generated table.

Run: python check_numbers.py ataskaita.docx [--from "4. PRAKTINĖ DALIS"]
Reads the .docx (Word export of the Google Doc), takes every number with a decimal comma ("0,903", "−0,006",
"92,4") and every number with thousands groups ("66 958 086", "2 000"), and looks each one up among the numbers
of results/tables/*.csv (headers included). Prints only the numbers it cannot find, with their section and the
text around them; those are checked by hand. A number found with the other sign counts as found ("+0,046" in the
text, "0,046 [0,043; 0,049]" in a table). Numbers worked out in the text (a difference of two table values,
ms → s) are expected among the misses. Whole numbers without a comma (years, section and formula numbers, counts)
are not checked: too many of them are not results. Notes in square brackets ("[A · ~1 300 ženklų] …") are skipped,
since they are removed before the report is handed in.
"""
import argparse
import csv
import glob
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import config

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
# a decimal number with a comma, or a whole number with groups of three digits separated by spaces
NUMBER = re.compile(r"(?<![\w,.])[−+-]?(?:\d{1,3}(?:[  ]\d{3})+(?:,\d+)?|\d+,\d+)(?![\w,]|[  ]\d)")


def normal(number):
    """One spelling for comparison: "-" for the minus sign, no "+", no spaces in the thousands groups."""
    return number.replace("−", "-").replace("+", "").replace(" ", "").replace(" ", "")


def table_numbers():
    """Every number in every generated table, with and without its sign."""
    known = set()
    for path in glob.glob(os.path.join(config.RESULTS_DIR, "tables", "*.csv")):
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                for cell in row:
                    for n in NUMBER.findall(cell):
                        known.add(normal(n))
                        known.add(normal(n).lstrip("-"))
    return known


def paragraphs(docx_path):
    """(section heading, text) for every paragraph of the document body, table cells included, in order."""
    root = ET.fromstring(zipfile.ZipFile(docx_path).read("word/document.xml"))
    section = ""
    for p in root.iter(W + "p"):
        text = "".join(t.text or "" for t in p.iter(W + "t"))
        style = p.find(f"{W}pPr/{W}pStyle")
        if style is not None and style.get(W + "val", "").lower().startswith("heading") and text.strip():
            section = text.strip()
        yield section, text


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Check the report's numbers against results/tables/.")
    parser.add_argument("docx")
    parser.add_argument("--from", dest="start", default="", help="start at the first heading that begins so")
    args = parser.parse_args()

    known = table_numbers()
    started = not args.start
    total, misses = 0, []
    for section, text in paragraphs(args.docx):
        if not started:
            started = section.startswith(args.start)
            if not started:
                continue
        if text.lstrip().startswith("["):         # a working note, not report text
            continue
        for m in NUMBER.finditer(text):
            total += 1
            if normal(m.group()) not in known and normal(m.group()).lstrip("-") not in known:
                context = text[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")
                misses.append((section, m.group(), context))

    print(f"Skaičių: {total}, rasta lentelėse: {total - len(misses)}, nerasta: {len(misses)}")
    last = None
    for section, number, context in misses:
        if section != last:
            print(f"\n== {section}")
            last = section
        print(f"  {number:>14}   …{context}…")
