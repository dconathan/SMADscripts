"""
Grabs all articles from files in DATA_DIR and does a daily count.
Saves as a csv as OUT_FILE
column names will be names of newspapers
each row will be a date. first column is date, remaining columns will be counts for whatever newspaper is the column header
note that if the daily count is 0, that row/date will be skipped/not appear in out file.
"""
import pandas as pd
import csv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # this is just so we can import the extract_news_dir script
from extract_news.extract_news import extract_news_dir

DATE_FORMAT = "%Y-%m-%d"

DATA_DIR = 'examples'
OUT_FILE = 'counts.csv'

counts = {}

# data is a list of dicts, one dict per article containing all the metadata
data = extract_news_dir(DATA_DIR)

for article in data:
    counts[article['newspaper']] = counts.get(article['newspaper'], {})
    counts[article['newspaper']][article['date']] = counts[article['newspaper']].get(article['date'], 0) + 1

d = pd.DataFrame.from_dict(counts, dtype=int).fillna(0).astype(int)

d.to_csv(OUT_FILE, quoting=csv.QUOTE_ALL, index=True)
