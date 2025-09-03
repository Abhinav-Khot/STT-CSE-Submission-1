import pydriller
import pandas as pd
import re
from tqdm import tqdm

url = 'manim'

repo = pydriller.Repository(url)

pattern = re.compile('.*((solv(ed|es|e|ing))|(fix(s|es|ing|ed)?)|((error|bug|issue)(s)?)).*', re.IGNORECASE)
need = []
also_need = []

for commit in tqdm(repo.traverse_commits(), desc="Processing commits"):
    if pattern.search(commit.msg):
        need.append([commit.hash, commit.msg, commit.parents, commit.merge, [file.filename for file in commit.modified_files]])
        for file in commit.modified_files:
            also_need.append([commit.hash, commit.msg, file.filename, file.source_code_before, file.source_code, file.diff])

db = pd.DataFrame(need, columns=['Hash', 'Message', 'Hashes of Parents', 'Is a merge commit?', 'List of modified files'])
also_db = pd.DataFrame(also_need, columns=['Hash', 'Message', 'Filename', 'Source code(before)', 'Source code(after)', 'Diff'])

db.to_csv('commit_summary.csv', index=False, escapechar = '\\')
also_db.to_csv('commit_files_diff.csv', index=False, escapechar = '\\')
print("Done", len(db))