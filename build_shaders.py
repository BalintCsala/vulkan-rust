import os
import sys
import subprocess

try:
    os.mkdir("spv/")
except:
    # Folder already exists
    pass

SLANGC_COMMAND = "slangc" if len(sys.argv) == 1 else sys.argv[1]

for name in os.listdir("shaders"):
    path = f"shaders/{name}"
    if os.path.isdir(path):
        continue
    subprocess.run([SLANGC_COMMAND, path, "-fvk-use-c-layout", "-fvk-use-entrypoint-name", "-o", f"spv/{name.replace(".slang", ".spv")}"])

