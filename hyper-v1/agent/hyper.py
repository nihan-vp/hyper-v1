import subprocess
import sys

prompt = " ".join(sys.argv[1:])

if not prompt:
    print("Usage: python agent/hyper.py ask your question")
    exit()

cmd = f'python model/generate.py'
process = subprocess.Popen(
    cmd,
    shell=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

out, err = process.communicate(prompt)

print(out)