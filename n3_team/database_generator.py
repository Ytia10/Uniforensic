"""
database_generator.py
Generates realistic SQLite forensic databases with logged ground truth.
Simulates SMS, Call Log, and Browser History databases across deletion pragmas:
- secure_delete (OFF, FAST, ON)
- auto_vacuum (NONE, FULL)
- journal_mode (DELETE, WAL)
Logs exact ground truth for active vs deleted rows.
"""

import sqlite3
import os
import random
import time
import json
from typing import Dict, List, Any, Tuple

SENDERS = ["+1-555-0199", "+1-555-0142", "+1-555-0188", "+1-555-0123", "BankNotify", "2FA-Service", "SupportDesk"]
RECIPIENTS = ["+1-555-0101", "+1-555-0102", "+1-555-0103", "+1-555-0104"]
SMS_TEMPLATES = [
    "Your verification code is {code}. Do not share.",
    "Hey, are we still meeting at {time} tomorrow?",
    "Payment of ${amt} received for invoice #{code}.",
    "Wire transfer initiated for account ending in {code}.",
    "Please call me back urgently regarding the investigation.",
    "Can you send the secret project documents to my private email?",
    "Meeting confirmed for {time}. Location: Downtown office.",
    "All records have been cleared as discussed."
]

DOMAINS = ["search.engine.com", "mail.internal.org", "crypto-exchange.io", "banking-portal.net", "social-feed.com", "darkweb-forum.onion", "cloud-storage.com"]
TITLES = [
    "Secure Account Login - Dashboard",
    "Private File Repository - Confidential",
    "Cryptocurrency Transaction Receipt",
    "Offshore Wire Transfer Services",
    "Encrypted Messaging Web Client",
    "News - Tech Forensics Update",
    "User Profile and Settings"
]

CALL_TYPES = ["INCOMING", "OUTGOING", "MISSED", "REJECTED"]


class ForensicDatabaseGenerator:
    def __init__(self, output_dir: str = "/tmp/forensic_dbs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_database(
        self,
        db_name: str,
        schema_type: str = "sms",
        num_rows: int = 60,
        num_delete: int = 20,
        secure_delete: int = 0,     # 0=OFF, 1=ON, 2=FAST
        auto_vacuum: int = 0,       # 0=NONE, 1=FULL
        journal_mode: str = "DELETE",
        seed: int = 42
    ) -> Tuple[str, str]:
        random.seed(seed)
        db_path = os.path.join(self.output_dir, f"{db_name}.db")
        gt_path = os.path.join(self.output_dir, f"{db_name}_ground_truth.json")

        for f in [db_path, f"{db_path}-wal", f"{db_path}-shm"]:
            if os.path.exists(f): os.remove(f)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Apply PRAGMAs
        cur.execute("PRAGMA page_size = 4096;")
        cur.execute(f"PRAGMA secure_delete = {secure_delete};")
        cur.execute(f"PRAGMA auto_vacuum = {auto_vacuum};")
        cur.execute(f"PRAGMA journal_mode = {journal_mode};")

        active_rows = []
        deleted_rows = []
        all_inserted = []
        table_name = ""
        columns = []

        base_time = 1700000000

        if schema_type == "sms":
            table_name = "messages"
            columns = ["id", "sender", "recipient", "body", "timestamp", "status"]
            schema_sql = """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                recipient TEXT NOT NULL,
                body TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                status INTEGER NOT NULL
            );
            """
            cur.execute(schema_sql)
            conn.commit()

            for i in range(1, num_rows + 1):
                sender = random.choice(SENDERS)
                recipient = random.choice(RECIPIENTS)
                tmpl = random.choice(SMS_TEMPLATES)
                body = tmpl.format(
                    code=random.randint(10000, 99999),
                    time=f"{random.randint(1,12)}:{random.choice(['00','15','30','45'])} PM",
                    amt=random.randint(50, 50000)
                )
                ts = base_time + i * 3600 + random.randint(10, 300)
                status = random.choice([0, 1, 2])
                cur.execute(
                    "INSERT INTO messages (sender, recipient, body, timestamp, status) VALUES (?, ?, ?, ?, ?);",
                    (sender, recipient, body, ts, status)
                )
                row_dict = {
                    "id": i,
                    "sender": sender,
                    "recipient": recipient,
                    "body": body,
                    "timestamp": ts,
                    "status": status
                }
                all_inserted.append(row_dict)

        elif schema_type == "calls":
            table_name = "calls"
            columns = ["id", "phone_number", "duration_sec", "timestamp", "call_type"]
            schema_sql = """
            CREATE TABLE calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT NOT NULL,
                duration_sec INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                call_type TEXT NOT NULL
            );
            """
            cur.execute(schema_sql)
            conn.commit()

            for i in range(1, num_rows + 1):
                phone = random.choice(SENDERS)
                dur = random.randint(5, 3600)
                ts = base_time + i * 1800 + random.randint(10, 300)
                ctype = random.choice(CALL_TYPES)
                cur.execute(
                    "INSERT INTO calls (phone_number, duration_sec, timestamp, call_type) VALUES (?, ?, ?, ?);",
                    (phone, dur, ts, ctype)
                )
                all_inserted.append({
                    "id": i,
                    "phone_number": phone,
                    "duration_sec": dur,
                    "timestamp": ts,
                    "call_type": ctype
                })

        elif schema_type == "browser":
            table_name = "browser_history"
            columns = ["id", "url", "title", "visit_time", "visit_count"]
            schema_sql = """
            CREATE TABLE browser_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                visit_time INTEGER NOT NULL,
                visit_count INTEGER NOT NULL
            );
            """
            cur.execute(schema_sql)
            conn.commit()

            for i in range(1, num_rows + 1):
                domain = random.choice(DOMAINS)
                url = f"https://{domain}/view?id={random.randint(100, 999)}"
                title = random.choice(TITLES)
                ts = base_time + i * 600
                vcount = random.randint(1, 45)
                cur.execute(
                    "INSERT INTO browser_history (url, title, visit_time, visit_count) VALUES (?, ?, ?, ?);",
                    (url, title, ts, vcount)
                )
                all_inserted.append({
                    "id": i,
                    "url": url,
                    "title": title,
                    "visit_time": ts,
                    "visit_count": vcount
                })

        conn.commit()

        # Randomly delete a subset of rows
        delete_indices = sorted(random.sample(range(len(all_inserted)), min(num_delete, len(all_inserted))))
        delete_ids = [all_inserted[idx]["id"] for idx in delete_indices]

        for did in delete_ids:
            cur.execute(f"DELETE FROM {table_name} WHERE id = ?;", (did,))

        conn.commit()
        conn.close()

        deleted_set = set(delete_ids)
        for row in all_inserted:
            if row["id"] in deleted_set:
                deleted_rows.append(row)
            else:
                active_rows.append(row)

        ground_truth = {
            "db_name": db_name,
            "schema_type": schema_type,
            "table_name": table_name,
            "columns": columns,
            "pragmas": {
                "secure_delete": secure_delete,
                "auto_vacuum": auto_vacuum,
                "journal_mode": journal_mode
            },
            "num_inserted": num_rows,
            "num_deleted": len(deleted_rows),
            "num_active": len(active_rows),
            "active_rows": active_rows,
            "deleted_rows": deleted_rows
        }

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)

        return db_path, gt_path
