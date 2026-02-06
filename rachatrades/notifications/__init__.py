"""Notification module for trade alerts (email, future: WhatsApp, push, SMS)."""

from .email_notifier import EmailNotifier

__all__ = ["EmailNotifier"]
