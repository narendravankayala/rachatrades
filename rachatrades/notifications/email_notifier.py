"""Email notification for trade signals via Gmail SMTP."""

import logging
import os
import smtplib
from datetime import datetime
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send trade alert emails via Gmail SMTP."""

    def __init__(
        self,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        recipients: Optional[List[str]] = None,
    ):
        """
        Initialize email notifier.

        Args:
            smtp_user: Gmail address (or set SMTP_USER env var)
            smtp_password: Gmail app password (or set SMTP_PASSWORD env var)
            recipients: List of email addresses to notify (or set NOTIFY_EMAILS env var, comma-separated)
        """
        self.smtp_user = smtp_user or os.environ.get("SMTP_USER", "")
        self.smtp_password = smtp_password or os.environ.get("SMTP_PASSWORD", "")
        self.recipients = recipients or [
            e.strip() for e in os.environ.get("NOTIFY_EMAILS", "").split(",") if e.strip()
        ]
        self.smtp_host = "smtp.gmail.com"
        self.smtp_port = 587

    @property
    def is_configured(self) -> bool:
        """Check if email notifications are properly configured."""
        return bool(self.smtp_user and self.smtp_password and self.recipients)

    def send_trade_alert(
        self,
        signal_type: str,
        ticker: str,
        price: float,
        reason: str,
        zone: str = "",
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Send a trade alert email.

        Args:
            signal_type: BUY, SELL, SHORT, or COVER
            ticker: Stock ticker
            price: Trade price
            reason: Strategy reason
            zone: Current zone (LONG/SHORT/FLAT)
            timestamp: When the signal was generated

        Returns:
            True if email sent successfully
        """
        if not self.is_configured:
            logger.warning("Email notifications not configured. Set SMTP_USER, SMTP_PASSWORD, NOTIFY_EMAILS.")
            return False

        time_str = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S ET")

        # Choose emoji and color based on signal type
        signal_config = {
            "BUY": {"emoji": "🟢", "color": "#28a745", "action": "OPENED LONG"},
            "SELL": {"emoji": "🔴", "color": "#dc3545", "action": "CLOSED LONG"},
            "SHORT": {"emoji": "🔻", "color": "#fd7e14", "action": "OPENED SHORT"},
            "COVER": {"emoji": "🔷", "color": "#17a2b8", "action": "CLOSED SHORT"},
        }
        config = signal_config.get(signal_type, {"emoji": "⚪", "color": "#6c757d", "action": signal_type})

        subject = f"{config['emoji']} RachaTrades: {signal_type} {ticker} @ ${price:.2f}"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto;">
            <div style="background: {config['color']}; color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0;">
                <h1 style="margin: 0; font-size: 28px;">{config['emoji']} {signal_type}</h1>
                <h2 style="margin: 5px 0 0; font-size: 36px;">{ticker}</h2>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border: 1px solid #dee2e6;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px 0; color: #666;">Action</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold;">{config['action']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #666;">Price</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: bold; font-size: 20px;">${price:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #666;">Zone</td>
                        <td style="padding: 8px 0; text-align: right;">{zone}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #666;">Time</td>
                        <td style="padding: 8px 0; text-align: right;">{time_str}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #666;">Reason</td>
                        <td style="padding: 8px 0; text-align: right;">{reason}</td>
                    </tr>
                </table>
            </div>
            <div style="padding: 10px; text-align: center; color: #999; font-size: 12px; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 10px 10px;">
                <p>Rashemator Cloud Flip Strategy | <a href="https://rachatrades.com">rachatrades.com</a></p>
            </div>
        </body>
        </html>
        """

        plain_body = f"""
RachaTrades Alert: {signal_type} {ticker}
{'='*40}
Action: {config['action']}
Price:  ${price:.2f}
Zone:   {zone}
Time:   {time_str}
Reason: {reason}
{'='*40}
Rashemator Cloud Flip Strategy
rachatrades.com
"""

        return self._send_email(subject, html_body, plain_body)

    def send_scan_summary(
        self,
        buys: list,
        sells: list,
        shorts: list,
        covers: list,
        stats: dict,
        zone_counts: dict,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Send a summary email after each scan (only if there are new signals).

        Args:
            buys/sells/shorts/covers: Lists of StrategyResult objects
            stats: Portfolio stats dict
            zone_counts: Zone distribution dict
            timestamp: Scan time

        Returns:
            True if email sent successfully
        """
        if not self.is_configured:
            return False

        # Only send if there are actionable signals
        total_signals = len(buys) + len(sells) + len(shorts) + len(covers)
        if total_signals == 0:
            logger.info("No new signals - skipping summary email")
            return False

        time_str = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S ET")

        subject = f"📊 RachaTrades Scan: {len(buys)}B {len(sells)}S {len(shorts)}SH {len(covers)}C | {time_str}"

        # Build trade rows
        trade_rows = ""
        for r in buys:
            trade_rows += f'<tr><td style="padding:6px;color:#28a745;font-weight:bold;">🟢 BUY</td><td style="padding:6px;">{r.ticker}</td><td style="padding:6px;">${r.price:.2f}</td><td style="padding:6px;font-size:12px;">{r.reason}</td></tr>\n'
        for r in shorts:
            trade_rows += f'<tr><td style="padding:6px;color:#fd7e14;font-weight:bold;">🔻 SHORT</td><td style="padding:6px;">{r.ticker}</td><td style="padding:6px;">${r.price:.2f}</td><td style="padding:6px;font-size:12px;">{r.reason}</td></tr>\n'
        for r in sells:
            trade_rows += f'<tr><td style="padding:6px;color:#dc3545;font-weight:bold;">🔴 SELL</td><td style="padding:6px;">{r.ticker}</td><td style="padding:6px;">${r.price:.2f}</td><td style="padding:6px;font-size:12px;">{r.reason}</td></tr>\n'
        for r in covers:
            trade_rows += f'<tr><td style="padding:6px;color:#17a2b8;font-weight:bold;">🔷 COVER</td><td style="padding:6px;">{r.ticker}</td><td style="padding:6px;">${r.price:.2f}</td><td style="padding:6px;font-size:12px;">{r.reason}</td></tr>\n'

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #1a1a2e; color: white; padding: 15px; text-align: center; border-radius: 10px 10px 0 0;">
                <h2 style="margin: 0;">📊 RachaTrades Scan Results</h2>
                <p style="margin: 5px 0 0; color: #aaa;">{time_str}</p>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border: 1px solid #dee2e6;">
                <p style="color: #666; margin: 0 0 5px;">Market Zones: 
                    <span style="color:#28a745;">LONG {zone_counts.get('LONG',0)}</span> | 
                    <span style="color:#6c757d;">FLAT {zone_counts.get('FLAT',0)}</span> | 
                    <span style="color:#dc3545;">SHORT {zone_counts.get('SHORT',0)}</span>
                </p>
                <table style="width:100%; border-collapse:collapse; margin-top:10px;">
                    <tr style="background:#e9ecef;">
                        <th style="padding:8px;text-align:left;">Signal</th>
                        <th style="padding:8px;text-align:left;">Ticker</th>
                        <th style="padding:8px;text-align:left;">Price</th>
                        <th style="padding:8px;text-align:left;">Reason</th>
                    </tr>
                    {trade_rows}
                </table>
            </div>
            <div style="background: #e9ecef; padding: 10px 15px; border: 1px solid #dee2e6; border-top: none;">
                <p style="margin:3px 0; font-size:13px;">Open positions: {stats.get('open_positions', 0)} | Total trades: {stats.get('total_trades', 0)} | Win rate: {stats.get('win_rate', 0):.0f}% | P&L: ${stats.get('total_pnl', 0):.2f}</p>
            </div>
            <div style="padding: 8px; text-align: center; color: #999; font-size: 11px; border-radius: 0 0 10px 10px;">
                <a href="https://rachatrades.com">rachatrades.com</a>
            </div>
        </body>
        </html>
        """

        plain_body = f"""
RachaTrades Scan Results - {time_str}
{'='*50}
Zones: LONG={zone_counts.get('LONG',0)} FLAT={zone_counts.get('FLAT',0)} SHORT={zone_counts.get('SHORT',0)}

"""
        for r in buys:
            plain_body += f"  BUY   {r.ticker:>6} @ ${r.price:.2f} - {r.reason}\n"
        for r in shorts:
            plain_body += f"  SHORT {r.ticker:>6} @ ${r.price:.2f} - {r.reason}\n"
        for r in sells:
            plain_body += f"  SELL  {r.ticker:>6} @ ${r.price:.2f} - {r.reason}\n"
        for r in covers:
            plain_body += f"  COVER {r.ticker:>6} @ ${r.price:.2f} - {r.reason}\n"

        plain_body += f"""
{'='*50}
Open: {stats.get('open_positions', 0)} | Trades: {stats.get('total_trades', 0)} | Win: {stats.get('win_rate', 0):.0f}% | P&L: ${stats.get('total_pnl', 0):.2f}
"""

        return self._send_email(subject, html_body, plain_body)

    def _send_email(self, subject: str, html_body: str, plain_body: str) -> bool:
        """Send an email via Gmail SMTP."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = Header(subject, "utf-8")
            msg["From"] = self.smtp_user
            msg["To"] = ", ".join(self.recipients)

            msg.attach(MIMEText(plain_body, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
