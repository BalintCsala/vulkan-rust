import os
from os.path import isdir
import sys
import subprocess
import shutil


SLANGC_COMMAND = "slangc" if len(sys.argv) == 1 else sys.argv[1]

if os.path.isdir("spv"):
    shutil.rmtree("spv")

os.mkdir("spv/")

for name in os.listdir("shaders"):
    path = f"shaders/{name}"
    if os.path.isdir(path):
        continue
    subprocess.run([SLANGC_COMMAND, path, "-fvk-use-c-layout", "-fvk-use-entrypoint-name", "-o", f"spv/{name.replace(".slang", ".spv")}"])

