import re
import sys
from pathlib import Path

app_path = Path("trustcv_final.py")
if not app_path.exists():
    app_path = Path(__file__).resolve().parent.parent / "trustcv_final.py"

text = app_path.read_text(encoding="utf-8")
lines = text.splitlines()

# Find all widget keys defined with key="..." or key='...'
widget_pattern = re.compile(r'st\.[a-zA-Z0-9_]+\([^)]*key=["\']([a-zA-Z0-9_]+)["\']', re.DOTALL)
# Find individual lines with key="..."
widget_key_lines = {}
for i, line in enumerate(lines, 1):
    m = re.search(r'key=["\']([a-zA-Z0-9_]+)["\']', line)
    if m:
        widget_key_lines.setdefault(m.group(1), []).append(i)

print(f"Total widget keys found: {len(widget_key_lines)}")

# Check for session_state assignment to any widget key after its instantiation
conflicts = []
for k, w_lines in widget_key_lines.items():
    assign_lines = []
    for i, line in enumerate(lines, 1):
        if re.search(rf'st\.session_state\[["\']{k}["\']\]\s*=', line) or re.search(rf'st\.session_state\.{k}\s*=', line):
            assign_lines.append(i)
    
    if assign_lines:
        for a in assign_lines:
            # Check if assignment occurs on or after widget line
            for w in w_lines:
                if a >= w:
                    conflicts.append((k, w, a, lines[a - 1].strip()))

print(f"Post-instantiation assignment conflicts: {len(conflicts)}")
for k, w, a, code in conflicts:
    print(f"  [CONFLICT] Key '{k}': widget at line {w}, assigned at line {a}: {code}")

if not conflicts:
    print("[PASS] Zero post-instantiation session_state widget conflicts found!")
else:
    sys.exit(1)
