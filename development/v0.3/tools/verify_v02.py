"""Verify the untouched parent copy against the pre-development manifest."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1];original=root.parents[1]
files=json.loads((root/'V02_BASELINE.json').read_text())
wrong=[row['path'] for row in files if hashlib.sha256((original/row['path']).read_bytes()).hexdigest()!=row['sha256']]
print(f'{len(files)} original files checked; changed: {wrong}')
if wrong:raise SystemExit(1)
