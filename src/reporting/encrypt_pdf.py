from PyPDF2 import PdfReader, PdfWriter
import random


def encrypt_pdf(pdf_path):

    password = str(random.randint(100000,999999))

    reader = PdfReader(pdf_path)

    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    writer.encrypt(password)

    encrypted_path = pdf_path.replace(".pdf","_secure.pdf")

    with open(encrypted_path,"wb") as f:
        writer.write(f)

    return encrypted_path, password