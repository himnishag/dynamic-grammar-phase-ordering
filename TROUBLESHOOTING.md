# TROUBLESHOOTING GUIDE

## 1. LLVM Tooling & Windows Environment

### Issue: `opt` or `clang` command not found
* **Explanation**: Native Windows LLVM binaries are either not installed or not added to your system PATH environment variable.
* **Resolution A (Recommended for Windows)**: Use the built-in `DEMO_MODE`. The system automatically falls back to surrogate neural inference and cached benchmark data.
* **Resolution B (Real Compilation on Windows)**: Install LLVM for Windows (e.g. LLVM 18) and set `LLVM_BIN=C:\Program Files\LLVM\bin` in environment variables.
* **Resolution C (WSL2 Ubuntu)**: Run inside WSL2 Ubuntu (`sudo apt install llvm clang`).

## 2. Python Dependencies

### Issue: `ModuleNotFoundError: No module named 'torch'`
* **Resolution**: Run `python -m pip install torch`.

### Issue: `ModuleNotFoundError: No module named 'fastapi'`
* **Resolution**: Run `python -m pip install fastapi uvicorn python-multipart`.

## 3. Frontend & Dashboard

### Issue: Dashboard page fails to connect to backend
* **Resolution**: Ensure the FastAPI server is running on port 8000 (`python -m uvicorn backend.main:app --port 8000`).
