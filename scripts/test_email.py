"""Quick test script to verify email notifications are working."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rachatrades.notifications import EmailNotifier


def main():
    notifier = EmailNotifier()

    print("Email Configuration Check")
    print("=" * 40)
    print(f"  SMTP_USER:     {'SET' if notifier.smtp_user else 'MISSING'}")
    print(f"  SMTP_PASSWORD: {'SET' if notifier.smtp_password else 'MISSING'}")
    print(f"  NOTIFY_EMAILS: {notifier.recipients if notifier.recipients else 'MISSING'}")
    print(f"  Configured:    {notifier.is_configured}")
    print("=" * 40)

    if not notifier.is_configured:
        print("\nEmail is NOT configured. Set these env vars:")
        if not notifier.smtp_user:
            print("  export SMTP_USER='your@gmail.com'")
        if not notifier.smtp_password:
            print("  export SMTP_PASSWORD='your-gmail-app-password'")
        if not notifier.recipients:
            print("  export NOTIFY_EMAILS='recipient@gmail.com'")
        sys.exit(1)

    print("\nSending test email...")
    success = notifier.send_trade_alert(
        signal_type="BUY",
        ticker="TEST",
        price=123.45,
        reason="This is a test email from scripts/test_email.py",
        zone="LONG",
    )

    if success:
        print("Test email sent! Check your inbox.")
    else:
        print("Failed to send email. Check the error above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
