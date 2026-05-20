#!/bin/bash
# 盤前系統自動執行腳本（rtcwake 喚醒後執行）
cd /home/mark/workspace/ML/qt
mkdir -p logs

echo "=== $(date) 自動喚醒執行盤前 ===" >> logs/premarket_cron.log

/home/mark/miniconda3/bin/conda run -n qt_env python premarket_main.py \
    >> logs/premarket_cron.log 2>&1

echo "=== 執行完畢，回到 suspend ===" >> logs/premarket_cron.log

# 執行完後自動回 suspend
sleep 5
systemctl suspend
