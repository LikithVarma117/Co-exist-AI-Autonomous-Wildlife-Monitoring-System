import smtplib
from email.message import EmailMessage
import streamlit as st

# Load credentials from secrets.toml
try:
    user = st.secrets["SENDER_EMAIL"]
    password = st.secrets["SENDER_PASSWORD"]
    recipient = st.secrets["RECIPIENT_EMAIL_ADDRESS"]
except Exception as e:
    print("❌ ERROR: Could not find keys in secrets.toml")
    print("Make sure you added SENDER_EMAIL and SENDER_PASSWORD to .streamlit/secrets.toml")
    exit()

print(f"📧 Attempting to send from: {user}")
print(f"📧 Sending to: {recipient}")

try:
    msg = EmailMessage()
    msg.set_content("This is a test email from your Wildlife Detector.")
    msg['Subject'] = "Test Alert"
    msg['From'] = user
    msg['To'] = recipient

    # Connect to Gmail
    print("⏳ Connecting to Gmail Server...")
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    
    print("🔑 Logging in...")
    server.login(user, password)
    
    print("🚀 Sending message...")
    server.send_message(msg)
    server.quit()
    
    print("✅ SUCCESS! Email sent. Check your inbox.")

except Exception as e:
    print("\n❌ FAILED TO SEND EMAIL.")
    print("================ERROR MESSAGE=================")
    print(e)
    print("==============================================")
    print("\nTroubleshooting Tips:")
    print("1. If the error says 'WinError 10060' or 'Timeout', AVAST IS BLOCKING IT.")
    print("2. If the error says 'Username and Password not accepted', your App Password is wrong.")