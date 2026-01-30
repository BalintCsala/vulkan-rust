import os
import sys
import subprocess

try:
    os.mkdir("spv/")
except:
    # Folder already exists
    pass

SLANGC_COMMAND = "slangc" if len(sys.argv) == 1 else sys.argv[1]

for file in os.listdir("shaders"):
    subprocess.run([SLANGC_COMMAND, f"shaders/{file}", "-fvk-use-c-layout", "-fvk-use-entrypoint-name", "-o", f"spv/{file.replace(".slang", ".spv")}"])

