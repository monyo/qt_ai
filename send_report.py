"""寄送盤前報告（含質化分析）

使用方式:
    python send_report.py 2026-07-06
    python send_report.py 20260706

會讀取:
    data/actions_YYYYMMDD.json      （必要）
    data/qualitative_YYYYMMDD.json  （可選，存在時自動合併到報告）
"""
import argparse
import json
import os
import sys

from src.notifier import GmailNotifier


def main():
    parser = argparse.ArgumentParser(description="寄送盤前報告（含質化分析）")
    parser.add_argument("date", help="日期，格式 YYYY-MM-DD 或 YYYYMMDD")
    args = parser.parse_args()

    date_str = args.date.replace("-", "")
    actions_path = f"data/actions_{date_str}.json"
    qualitative_path = f"data/qualitative_{date_str}.json"

    if not os.path.exists(actions_path):
        print(f"找不到 {actions_path}")
        sys.exit(1)

    with open(actions_path, encoding="utf-8") as f:
        actions_data = json.load(f)

    qualitative_data = None
    if os.path.exists(qualitative_path):
        with open(qualitative_path, encoding="utf-8") as f:
            qualitative_data = json.load(f)
        print(f"載入質化分析：{len(qualitative_data.get('analyses', []))} 筆")
    else:
        print("（無質化分析檔，直接寄送原始報告）")

    notifier = GmailNotifier()
    if not notifier.is_configured():
        print("Email 未設定（檢查 .env GMAIL_* 欄位）")
        sys.exit(1)

    print("正在發送 Email 報告...")
    if notifier.send_premarket_report(actions_data, qualitative_data=qualitative_data):
        print(f"Email 已發送至 {notifier.recipient}")
    else:
        print("Email 發送失敗，請檢查 .env 設定")
        sys.exit(1)


if __name__ == "__main__":
    main()
