"""Send the encrypted report by email.

Credentials come only from environment variables (SMTP_USER, SMTP_PASSWORD,
optional SMTP_HOST / SMTP_PORT / SMTP_SENDER). The PDF password is deliberately
NOT included in the email - share it through a separate channel.
"""

import os
import re
import smtplib
from email.message import EmailMessage

from src.config import config

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


class EmailNotConfigured(RuntimeError):
    pass


def is_valid_email(address: str) -> bool:
    return bool(address) and len(address) <= 254 and bool(EMAIL_RE.match(address))


def send_email(receiver_email: str, pdf_path: str, report_id: str = "") -> None:
    user = os.environ.get("SMTP_USER", config.SMTP_USER)
    password = os.environ.get("SMTP_PASSWORD", config.SMTP_PASSWORD)
    host = os.environ.get("SMTP_HOST", config.SMTP_HOST)
    port = int(os.environ.get("SMTP_PORT", config.SMTP_PORT))
    sender = os.environ.get("SMTP_SENDER", config.SMTP_SENDER) or user

    if not (user and password):
        raise EmailNotConfigured("Email is not configured on this server.")
    if not is_valid_email(receiver_email):
        raise ValueError("Invalid email address.")

    msg = EmailMessage()
    msg["Subject"] = "Diabetic Retinopathy AI Report"
    msg["From"] = sender
    msg["To"] = receiver_email
    msg.set_content(
        "Hello,\n\n"
        "A diabetic retinopathy AI analysis report has been shared with you and is attached.\n"
        f"Report ID: {report_id[:12]}\n\n"
        "The PDF is password protected (AES-256). For security the password is not included in this "
        "email - please obtain it from the person who sent you the report.\n\n"
        "This is an AI decision-support report for research use and does not replace a clinical "
        "diagnosis by an ophthalmologist.\n"
    )

    with open(pdf_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename="DR_AI_Report.pdf")

    with smtplib.SMTP_SSL(host, port, timeout=20) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)
