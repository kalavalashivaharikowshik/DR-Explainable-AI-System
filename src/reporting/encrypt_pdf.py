import os
import secrets
import string
import warnings

# pypdf imports the legacy ARC4 cipher for reading old PDFs; we only use AES-256.
warnings.filterwarnings("ignore", message=".*ARC4 has been moved.*")

from pypdf import PdfReader, PdfWriter  # noqa: E402

# no look-alike characters (0/O, 1/l/I) so the password is easy to read out
_ALPHABET = "".join(c for c in string.ascii_letters + string.digits if c not in "0O1lI")


def generate_password(length: int = 10) -> str:
    """Cryptographically secure password (~57 bits of entropy for length 10)."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))


def encrypt_pdf(pdf_path: str, password: str = None, remove_original: bool = True):
    """Encrypt with AES-256. Returns (encrypted_path, password)."""
    password = password or generate_password()

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata(reader.metadata or {})
    writer.encrypt(user_password=password, algorithm="AES-256")

    root, ext = os.path.splitext(pdf_path)
    encrypted_path = f"{root}_secure{ext}"
    with open(encrypted_path, "wb") as f:
        writer.write(f)

    if remove_original:
        # never leave an unencrypted copy of the report on disk
        os.remove(pdf_path)

    return encrypted_path, password
