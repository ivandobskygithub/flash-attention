import os, glob, subprocess, sys

root = r"D:\development\blackwell_flash\hopper"
pyroot = r"C:\Users\NICKF\AppData\Local\Programs\Python\Python313"
venv   = r"D:\development\flashattn\venv"
cuda   = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
msvc   = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808"
kits_ucrt = r"C:\Program Files (x86)\Windows Kits\10\lib\10.0.22621.0\ucrt\x64"
kits_um   = r"C:\Program Files (x86)\Windows Kits\10\lib\10.0.22621.0\um\x64"

build_temp = os.path.join(root, r"build\temp.win-amd64-cpython-313\Release")
build_lib  = os.path.join(root, r"build\lib.win-amd64-cpython-313\flash_attn_3")
os.makedirs(build_lib, exist_ok=True)

out_pyd = os.path.join(build_lib, "_C.pyd")
implib  = os.path.join(build_temp, "_C.lib")

libs = [
    os.path.join(venv, r"Lib\site-packages\torch\lib"),
    os.path.join(cuda, r"lib\x64"),
    os.path.join(venv, "libs"),
    os.path.join(pyroot, "libs"),
    pyroot,
    os.path.join(msvc, r"lib\x64"),
    kits_ucrt,
    kits_um,
]

# Collect every .obj that was already compiled
objs = glob.glob(os.path.join(build_temp, "**", "*.obj"), recursive=True)
if not objs:
    print("No .obj files found. Make sure you compiled first.", file=sys.stderr)
    sys.exit(1)

rsp_path = os.path.join(root, "build", "link.rsp")
with open(rsp_path, "w", encoding="utf-8") as f:
    f.write("/nologo\n/INCREMENTAL:NO\n/LTCG\n/DLL\n/MANIFEST:EMBED,ID=2\n/MANIFESTUAC:NO\n")
    for lp in libs:
        f.write(f'/LIBPATH:"{lp}"\n')
    # CUDA + Torch libs; include cudadevrt/cuda to satisfy device-link symbols.
    f.write("c10.lib\ntorch.lib\ntorch_cpu.lib\ncudart.lib\ncudadevrt.lib\ncuda.lib\nc10_cuda.lib\ntorch_cuda.lib\n")
    f.write("/EXPORT:PyInit__C\n")
    for o in objs:
        f.write(f'"{o}"\n')
    f.write(f'/OUT:"{out_pyd}"\n/IMPLIB:"{implib}"\n')

link = os.path.join(msvc, r"bin\HostX64\x64\link.exe")
print("Linking via response file...")
subprocess.check_call([link, "@" + rsp_path])
print("Linked:", out_pyd)
