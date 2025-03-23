# test_email.py
import os
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage

load_dotenv()

sender_email = os.getenv("EMAIL_ADDRESS")
app_password = os.getenv("EMAIL_APP_PASSWORD")

msg = EmailMessage()
msg['Subject'] = "Test Email"
msg['From'] = sender_email
msg['To'] = sender_email
msg.set_content("Test email from Python script.")

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(sender_email, app_password)
    smtp.send_message(msg)
    print("✅ Email sent!")
