import streamlit as st
import requests
import smtplib
from email.message import EmailMessage

print("\n--- 🕵️ DEBUGGING SECRETS & CONNECTIONS ---")

# 1. CHECK IF SECRETS EXIST
print("1. Checking secrets.toml...")
try:
    # Check Telegram
    tg_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = st.secrets.get("TELEGRAM_CHAT_ID", "")
    
    # Check Email
    # Try to grab the keys. If they don't exist in secrets, we define them as None to test later.
    email_user = st.secrets.get("SENDER_EMAIL", None)
    email_pass = st.secrets.get("SENDER_PASSWORD", None)
    
    # Mask them for printing so we don't show passwords on screen
    print(f"   - Telegram Token Found? {'YES' if len(tg_token) > 5 else 'NO (Value is empty)'}")
    print(f"   - Telegram Chat ID Found? {'YES' if len(tg_chat) > 1 else 'NO (Value is empty)'}")
    print(f"   - Email User Found? {email_user if email_user else 'NO (Not in secrets)'}")
    print(f"   - Email Pass Found? {'YES' if email_pass else 'NO (Not in secrets)'}")

except FileNotFoundError:
    print("❌ CRITICAL ERROR: .streamlit/secrets.toml file NOT found.")
    exit()

# 2. TEST TELEGRAM
print("\n2. Testing Telegram Connection...")
if len(tg_token) > 5 and len(tg_chat) > 1:
    url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
    try:
        data = {'chat_id': tg_chat, 'text': "✅ Test from VS Code Wildlife Detector"}
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print("   ✅ Telegram Success! Message sent.")
        else:
            print(f"   ❌ Telegram Failed. Status Code: {response.status_code}")
            print(f"   Reason: {response.text}")
    except Exception as e:
        print(f"   ❌ Telegram Connection Error: {e}")
else:
    print("   ⚠️ Skipping Telegram Test (Keys missing in secrets.toml)")

# 3. TEST EMAIL (Hardcoded Fallback Test)
print("\n3. Testing Email Connection...")

# --- UPDATE THESE TWO LINES MANUALLY JUST FOR THIS TEST ---
TEST_USER = "coexist.ai.wildlife@gmail.com" 
TEST_PASS = "txoq mbeu jtau xgcp" # Put your App Password here
RECIPIENT = "put_your_personal_email_here@gmail.com"
# --------------------------------------------------------

if TEST_PASS == "txoq mbeu jtau xgcp":
    print("   ⚠️ WARNING: You are using the default password from the example.")
    print("   If this app password was revoked or belongs to someone else, it will fail.")

try:
    msg = EmailMessage()
    msg.set_content("Debug test.")
    msg['Subject'] = "Debug Test"
    msg['From'] = TEST_USER
    msg['To'] = RECIPIENT
    
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(TEST_USER, TEST_PASS)
    server.send_message(msg)
    server.quit()
    print("   ✅ Email Success! Logged in and sent.")
except Exception as e:
    print(f"   ❌ Email Failed: {e}")

print("\n-------------------------------------------")