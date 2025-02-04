#!/usr/bin/env python3
import subprocess
import time
import os
import yagmail
import logging
from datetime import datetime

class GPUMonitor:
    def __init__(self, email_sender, app_password, email_recipient):
        """
        Initialize GPU monitor with email credentials
        
        Args:
            email_sender (str): Gmail address to send from
            app_password (str): Gmail application-specific password
            email_recipient (str): Email address to send notifications to
        """
        self.email_sender = email_sender
        self.app_password = app_password
        self.email_recipient = email_recipient
        self.yag = yagmail.SMTP(email_sender, app_password)
        
        # Setup logging
        logging.basicConfig(
            filename='gpu_monitor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def check_nvidia_smi(self):
        """Check if nvidia-smi command works and return error if any"""
        try:
            result = subprocess.run(
                ['nvidia-smi'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return None
        except subprocess.CalledProcessError as e:
            return e.stderr
        except FileNotFoundError:
            return "nvidia-smi command not found. NVIDIA drivers may not be installed."

    def send_notification(self, error_msg):
        """Send email notification about GPU access error"""
        subject = "⚠️⚠️⚠️⚠️⚠️ GPU Access Error Detected"
        content = [
            f"GPU access error detected on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nError message:\n{error_msg}",
            "\nPossible causes:",
            "1. NVIDIA driver issues",
            "2. GPU overload or crash",
            "3. Hardware malfunction",
            "\nRecommended actions:",
            "1. Check dmesg logs: sudo dmesg | grep nvidia",
            "2. Verify driver status: nvidia-smi",
            "3. Restart NVIDIA services: sudo systemctl restart nvidia-*",
            "4. If persists, consider rebooting the system"
        ]
        
        try:
            self.yag.send(self.email_recipient, subject, content)
            logging.info("Email notification sent successfully")
        except Exception as e:
            logging.error(f"Failed to send email notification: {str(e)}")

    def monitor(self, check_interval=300):
        """
        Start monitoring GPU access
        
        Args:
            check_interval (int): Time between checks in seconds (default: 5 minutes)
        """
        logging.info("Starting GPU monitoring service")
        last_notification_time = None
        notification_cooldown = 3600  # 1 hour cooldown between notifications
        
        while True:
            error = self.check_nvidia_smi()
            
            if error:
                current_time = time.time()
                # Check if we should send a notification (respect cooldown)
                if (last_notification_time is None or 
                    current_time - last_notification_time >= notification_cooldown):
                    
                    logging.error(f"GPU access error detected: {error}")
                    self.send_notification(error)
                    last_notification_time = current_time
            
            time.sleep(check_interval)

if __name__ == "__main__":
    # Configuration
    SENDER_EMAIL = 'ai.deploys@gmail.com'
    APP_PASSWORD = os.getenv('GM_APP_PASSWORD')
    RECIPIENT_EMAIL = "iaf@gs.utm.mx"
    CHECK_INTERVAL = 1800  # 30 minutes
    logging.info("GPU monitor parameters: \nRecipient: {}\nSender: {}\nCheck interval: {}".format(RECIPIENT_EMAIL, SENDER_EMAIL, CHECK_INTERVAL))
    if APP_PASSWORD in [None, '']:
        logging.error(f"No Gmail App Password stablished:")
        exit()
    
    monitor = GPUMonitor(SENDER_EMAIL, APP_PASSWORD, RECIPIENT_EMAIL)
    
    try:
        monitor.monitor(check_interval=CHECK_INTERVAL)
    except KeyboardInterrupt:
        logging.info("GPU monitoring service stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
