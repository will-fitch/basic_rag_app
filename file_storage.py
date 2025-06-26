import os
from typing import List

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(file_name: str, file_content: bytes) -> str:
    """Save a file to disk and return its path."""
    file_path = os.path.join(UPLOAD_DIR, file_name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path

def load_file_content(filename: str) -> str:
    """Load the content of a file from disk, handling .txt and .pdf appropriately."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    ext = get_file_extension(filename)
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            return extract_text_from_pdf(f.read())
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def list_uploaded_files() -> List[str]:
    """List all uploaded files in the upload directory."""
    return [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]

def load_all_uploaded_files() -> List[str]:
    """Return a list of all file contents for all uploaded files."""
    files = list_uploaded_files()
    return [load_file_content(f) for f in files]

def delete_uploaded_file(filename: str) -> None:
    """Delete a file from the upload directory."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

def filename_exists(filename: str) -> bool:
    """Check if a file exists in the upload directory."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    return os.path.isfile(file_path)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file given as bytes."""
    import pdfplumber
    from io import BytesIO
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def get_file_extension(filename: str) -> str:
    """Get the extension of a file."""
    return os.path.splitext(filename)[1].lower()