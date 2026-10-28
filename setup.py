from setuptools import setup, find_packages, Extension
import pybind11
import os
import glob


# ১. rc.exe খোঁজার লজিক
bin_paths = glob.glob(r"C:\Program Files (x86)\Windows Kits\10\bin\*\x64")
if bin_paths:
    os.environ["PATH"] += os.pathsep + bin_paths[-1]

# ২. হেডার ফাইল (io.h) খোঁজার লজিক
includes = [pybind11.get_include()]
ucrt_paths = glob.glob(r"C:\Program Files (x86)\Windows Kits\10\Include\*\ucrt")
um_paths = glob.glob(r"C:\Program Files (x86)\Windows Kits\10\Include\*\um")
shared_paths = glob.glob(r"C:\Program Files (x86)\Windows Kits\10\Include\*\shared")

if ucrt_paths: includes.append(ucrt_paths[-1])
if um_paths: includes.append(um_paths[-1])
if shared_paths: includes.append(shared_paths[-1])

if "INCLUDE" in os.environ:
    includes += [p for p in os.environ["INCLUDE"].split(";") if p]

# ৩. লাইব্রেরি ফাইল (kernel32.lib) খোঁজার লজিক 
library_dirs = []
lib_ucrt_paths = glob.glob(r"C:\Program Files (x86)\Windows Kits\10\Lib\*\ucrt\x64")
lib_um_paths = glob.glob(r"C:\Program Files (x86)\Windows Kits\10\Lib\*\um\x64")

if lib_ucrt_paths: library_dirs.append(lib_ucrt_paths[-1])
if lib_um_paths: library_dirs.append(lib_um_paths[-1])

if "LIB" in os.environ:
    library_dirs += [p for p in os.environ["LIB"].split(";") if p]

ext_modules = [
    Extension(
        "kecia.backend.photonic_backend",
        ["src/kecia/backend/photonic_core.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3"], # O3 হলো C++ এর হাই-পারফরম্যান্স অপটিমাইজেশন
    ),
]

setup(
    name="bantai-kecia",
    version="2.0.0", # 🚀 ফিউচারিস্টিক রিলিজ
    author="S Sardar",
    description="BANTAI Kecia - Deep Learning framework simulating Diamond-Glass Hybrid Photonic Chips.",
    packages=["kecia", "kecia.nn", "kecia.backend"],
    package_dir={"": "src"},
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.10.0"],
    python_requires=">=3.8",
)