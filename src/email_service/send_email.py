import smtplib
from email.message import EmailMessage


def send_email(receiver_email, pdf_path, password):

    sender_email = "kalavalakowshik183@gmail.com"
    sender_password = "alln zbwg bdsf gcqk"

    msg = EmailMessage()

    msg["Subject"] = "DR AI Report"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    msg.set_content(
        f"""
Your retinal analysis report is attached.

PDF Password: {password}

This is an AI decision support report.
"""
    )

    with open(pdf_path,"rb") as f:

        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename="DR_AI_Report.pdf"
        )

    with smtplib.SMTP_SSL("smtp.gmail.com",465) as smtp:

        smtp.login(sender_email,sender_password)

        smtp.send_message(msg)