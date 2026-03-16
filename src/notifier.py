"""Gmail SMTP 通知模組"""
import smtplib
import os
import datetime
import pathlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv

from src.risk import TRANCHE_PARAMS

load_dotenv()


class GmailNotifier:
    """Gmail SMTP 郵件發送器"""

    def __init__(self):
        self.sender = os.getenv("GMAIL_SENDER", "")
        self.password = os.getenv("GMAIL_APP_PASSWORD", "")
        self.recipient = os.getenv("GMAIL_RECIPIENT", "")
        self.enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"

    def is_configured(self):
        """檢查是否已設定必要參數"""
        return all([self.sender, self.password, self.recipient, self.enabled])

    def send_premarket_report(self, actions_data):
        """發送盤前報告

        Args:
            actions_data: actions JSON 資料（與儲存到檔案的格式相同）

        Returns:
            bool: 是否發送成功
        """
        if not self.is_configured():
            return False

        portfolio = actions_data.get("portfolio_snapshot", {})
        total_value = portfolio.get("total_value", 0)

        regime = actions_data.get("regime_status", {})
        regime_tag = " 🔴BEAR" if not regime.get("is_bull", True) else ""
        subject = f"盤前報告 {actions_data['date']} | 投組 ${total_value:,.0f}{regime_tag}"
        text_body = self._format_text_report(actions_data)
        summary_html = self._format_summary_html(actions_data)
        full_html = self._format_html_report(actions_data)

        data_dir = pathlib.Path("data")
        year = datetime.date.today().year
        candidates = [
            data_dir / "portfolio.json",
            data_dir / f"snapshot_{year}.json",
            data_dir / "watchlist.json",
        ]
        attachments = [(p.name, p.read_bytes()) for p in candidates if p.exists()]

        # PDF 附件（完整報告）
        pdf_bytes = self._generate_pdf(full_html)
        if pdf_bytes:
            report_date = actions_data.get("date", str(datetime.date.today())).replace("-", "")
            attachments.insert(0, (f"premarket_{report_date}.pdf", pdf_bytes))

        return self._send_email(subject, text_body, summary_html, attachments=attachments)

    def _generate_pdf(self, html_content):
        """將 HTML 報告轉為 PDF bytes（A4 橫向）"""
        try:
            from weasyprint import HTML, CSS
            page_css = CSS(string='''
                @page { size: A4 landscape; margin: 1.2cm 1cm; }
                body {
                    font-family: Arial, sans-serif;
                    font-size: 11px;
                    max-width: none !important;
                    margin: 0 !important;
                    color: #333;
                }
                h2 { font-size: 15px; margin: 0 0 6px; }
                h3 {
                    font-size: 12px;
                    margin: 14px 0 4px;
                    page-break-after: avoid;
                    break-after: avoid;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    page-break-inside: auto;
                    margin-bottom: 4px;
                }
                thead { display: table-header-group; }
                tr { page-break-inside: avoid; break-inside: avoid; }
                th { padding: 5px 7px; font-size: 11px; }
                td { padding: 5px 7px; font-size: 11px; }
                .section-block {
                    page-break-inside: avoid;
                    break-inside: avoid;
                }
                p { margin: 4px 0; font-size: 11px; }
            ''')
            return HTML(string=html_content).write_pdf(stylesheets=[page_css])
        except Exception as e:
            print(f"PDF 生成失敗: {e}")
            return None

    def _format_summary_html(self, data):
        """產生簡短摘要 HTML（email 本文）"""
        portfolio = data.get("portfolio_snapshot", {})
        actions = data.get("actions", [])
        regime = data.get("regime_status", {})
        sector = data.get("sector_status", {})

        # 年度 P&L
        yearly = portfolio.get("yearly_pnl")
        yearly_str = ""
        if yearly:
            sign = "+" if yearly["pnl_amount"] >= 0 else ""
            color = "#28a745" if yearly["pnl_amount"] >= 0 else "#dc3545"
            yearly_str = f' &nbsp;<span style="color:{color};">{sign}${yearly["pnl_amount"]:,.0f} ({sign}{yearly["pnl_pct"]:.1f}% YTD)</span>'

        # 市場體制
        if regime.get("is_bull", True):
            pct = f"+{regime.get('pct_vs_ma200', 0):.1f}%"
            regime_str = f'<span style="color:#28a745;">🟢 BULL SPY {pct} vs MA200</span>'
        else:
            pct = f"{regime.get('pct_vs_ma200', 0):.1f}%"
            regime_str = f'<span style="color:#dc3545;">🔴 BEAR SPY {pct} vs MA200</span>'

        # 板塊概況（一行）
        sector_html = ""
        sectors_data = (sector.get("relative") or {}).get("sectors") or {}
        if sectors_data:
            parts = []
            for name, s in sectors_data.items():
                pct_val = s.get("vs_spy_5d")
                if pct_val is not None:
                    c = "#28a745" if pct_val >= 0 else "#dc3545"
                    parts.append(f'<span style="color:{c};">{name} {pct_val:+.1f}%</span>')
            if parts:
                sector_html = f'<p style="margin:6px 0;font-size:12px;">板塊 vs SPY (5d): {" &nbsp;|&nbsp; ".join(parts)}</p>'

        # Actions 分類
        exits = [a for a in actions if a["action"] == "EXIT"]
        new_adds = [a for a in actions if a["action"] == "ADD" and not a.get("is_backup") and not a.get("is_pyramid")]
        pyramid_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_pyramid")]
        rotates = [a for a in actions if a["action"] == "ROTATE"]
        holds = [a for a in actions if a["action"] == "HOLD"]

        # EXIT 表
        exit_html = ""
        if exits:
            rows = ""
            for a in exits:
                pnl = a.get("pnl_pct", 0)
                pnl_color = "#28a745" if pnl and pnl >= 0 else "#dc3545"
                tranche_str = f" 第{a['tranche_n']}批" if a.get("tranche_n") else ""
                rows += f'<tr style="background:#fdf2f2;"><td style="padding:5px 8px;font-weight:bold;">{a["symbol"]}{tranche_str}</td><td style="padding:5px 8px;">{a.get("shares", 0)} 股</td><td style="padding:5px 8px;color:{pnl_color};">{pnl:+.1f}%</td><td style="padding:5px 8px;font-size:11px;color:#666;">{a.get("reason", "")[:60]}</td></tr>'
            exit_html = f'<h3 style="color:#dc3545;margin:14px 0 5px;">⛔ EXIT ({len(exits)} 筆)</h3><table style="border-collapse:collapse;width:100%;font-size:12px;"><tr style="background:#f0f0f0;"><th style="padding:5px 8px;text-align:left;">標的</th><th>股數</th><th>P&amp;L</th><th style="text-align:left;">原因</th></tr>{rows}</table>'

        # ROTATE 表
        rotate_html = ""
        if rotates:
            rows = ""
            for a in rotates:
                rows += f'<tr><td style="padding:5px 8px;color:#dc3545;font-weight:bold;">{a["sell_symbol"]}</td><td style="padding:5px 8px;">→</td><td style="padding:5px 8px;color:#28a745;font-weight:bold;">{a["buy_symbol"]}</td><td style="padding:5px 8px;font-size:11px;color:#666;">{a.get("reason", "")[:70]}</td></tr>'
            rotate_html = f'<h3 style="color:#fd7e14;margin:14px 0 5px;">🔄 ROTATE ({len(rotates)} 組)</h3><table style="border-collapse:collapse;width:100%;font-size:12px;border-top:1px solid #eee;">{rows}</table>'

        # ADD 表
        add_html = ""
        if new_adds or pyramid_adds:
            rows = ""
            for a in new_adds:
                rows += f'<tr><td style="padding:5px 8px;font-weight:bold;">{a["symbol"]}</td><td style="padding:5px 8px;font-size:11px;color:#555;">新倉 {a.get("suggested_shares", 0)} 股 @ ${a.get("current_price", 0):.2f}</td><td style="padding:5px 8px;font-size:11px;color:#666;">{a.get("reason", "")[:60]}</td></tr>'
            for a in pyramid_adds:
                direction = "↑" if a.get("direction") == "up" else "↓"
                rows += f'<tr style="background:#f0f7ff;"><td style="padding:5px 8px;font-weight:bold;">{a["symbol"]}</td><td style="padding:5px 8px;font-size:11px;color:#0066cc;">金字塔{direction} 第{a.get("tranche_n", 2)}批 {a.get("suggested_shares", 0)} 股</td><td style="padding:5px 8px;font-size:11px;color:#666;">{a.get("reason", "")[:60]}</td></tr>'
            add_html = f'<h3 style="color:#28a745;margin:14px 0 5px;">➕ ADD ({len(new_adds)} 新倉 + {len(pyramid_adds)} 金字塔)</h3><table style="border-collapse:collapse;width:100%;font-size:12px;border-top:1px solid #eee;">{rows}</table>'

        hold_html = f'<p style="margin:10px 0;font-size:12px;color:#6c757d;">✅ HOLD: {len(holds)} 檔（詳見附件 PDF）</p>'

        return f'''<html>
<body style="font-family:Arial,sans-serif;max-width:620px;margin:0 auto;padding:20px;color:#333;">
  <h2 style="margin-bottom:4px;">盤前報告 {data["date"]}</h2>
  <p style="color:#6c757d;margin:0 0 12px;font-size:12px;">版本 {data.get("version", "N/A")} &nbsp;|&nbsp; {regime_str}</p>

  <div style="background:#f8f9fa;padding:12px 16px;border-radius:6px;margin-bottom:12px;font-size:13px;">
    <strong>投組總值: ${portfolio.get("total_value", 0):,.0f}</strong>
    &nbsp;&nbsp; 現金: ${portfolio.get("cash", 0):,.0f}
    &nbsp;&nbsp; 個股: {portfolio.get("individual_count", 0)}/30{yearly_str}
  </div>

  {sector_html}
  {exit_html}
  {rotate_html}
  {add_html}
  {hold_html}

  <p style="color:#aaa;font-size:11px;margin-top:20px;">📎 完整持倉表、建議詳情請見附件 PDF</p>
  <hr style="border:none;border-top:1px solid #eee;margin:12px 0;">
  <p style="color:#ccc;font-size:10px;">此郵件由盤前建議系統自動發送</p>
</body>
</html>'''

    def _format_text_report(self, data):
        """產生純文字報告"""
        portfolio = data.get("portfolio_snapshot", {})
        sector = data.get("sector_status", {})
        actions = data.get("actions", [])

        regime = data.get("regime_status", {})
        lines = [
            f"盤前報告 {data['date']}",
            f"版本 {data.get('version', 'N/A')}",
            "=" * 40,
            "",
            f"投組總值: ${portfolio.get('total_value', 0):,.2f}",
            f"現金:     ${portfolio.get('cash', 0):,.2f}",
            f"個股:     {portfolio.get('individual_count', 0)}/30 檔",
        ]

        # 市場體制
        if regime:
            if regime.get("is_bull", True):
                pct = f"+{regime['pct_vs_ma200']:.1f}%" if regime.get("pct_vs_ma200") is not None else ""
                lines.append(f"市場體制: 🟢 BULL  SPY ${regime.get('spy_price')} > MA200 ${regime.get('ma200')} ({pct})")
            else:
                pct = f"{regime['pct_vs_ma200']:.1f}%" if regime.get("pct_vs_ma200") is not None else ""
                lines.append(f"市場體制: 🔴 BEAR  SPY ${regime.get('spy_price')} < MA200 ${regime.get('ma200')} ({pct})")
                lines.append("  ⚠️  ADD / ROTATE 已暫停，等 SPY 站回 MA200")
        lines.append("")

        # 年度 P&L
        yearly = portfolio.get("yearly_pnl")
        if yearly:
            sign = "+" if yearly["pnl_amount"] >= 0 else ""
            lines.append(f"年度 P&L: {sign}${yearly['pnl_amount']:,.2f} ({sign}{yearly['pnl_pct']:.1f}%)")

        lines.append("")

        # 板塊警告
        if sector.get("alerts"):
            lines.append("板塊警告:")
            for alert in sector["alerts"]:
                lines.append(f"  - {alert}")
            lines.append("")

        # 分類 actions
        exits = [a for a in actions if a["action"] == "EXIT"]
        holds = [a for a in actions if a["action"] == "HOLD"]
        new_adds = [a for a in actions if a["action"] == "ADD" and not a.get("is_backup") and not a.get("is_pyramid")]
        pyramid_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_pyramid")]
        backup_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_backup")]
        adds = new_adds  # 向下相容

        if exits:
            lines.append(f"EXIT 建議 ({len(exits)} 筆):")
            for a in exits:
                pnl = f"{a.get('pnl_pct', 0):+.1f}%" if a.get("pnl_pct") is not None else "N/A"
                tranche_str = f" 第{a['tranche_n']}批" if a.get("tranche_n") else ""
                lines.append(f"  {a['symbol']:<6}{tranche_str} {a.get('shares', 0):>4} 股  {pnl:<8} {a.get('reason', '')}")
            lines.append("")

        if holds:
            # 標註有趨勢警告的持倉
            hold_parts = []
            for a in holds:
                ts = a.get("trend_state")
                if ts and ts["state"] == "轉弱":
                    hold_parts.append(f"{a['symbol']}(↘️轉弱)")
                elif ts and ts["state"] == "轉強":
                    hold_parts.append(f"{a['symbol']}(↗️轉強)")
                else:
                    hold_parts.append(a["symbol"])
            lines.append(f"HOLD ({len(holds)} 檔): {', '.join(hold_parts)}")
            lines.append("")

        if new_adds or pyramid_adds or backup_adds:
            lines.append(f"ADD 建議 ({len(new_adds)} 新倉 + {len(pyramid_adds)} 金字塔 + {len(backup_adds)} 備選):")
            for a in new_adds:
                momentum = f"+{a.get('momentum', 0):.1f}%" if a.get("momentum") else ""
                rank = a.get("momentum_rank", "?")
                shares = a.get("suggested_shares", 0)

                rsi_str = ""
                rsi = a.get("rsi")
                if rsi is not None and rsi > 80:
                    rsi_str = f"  🔴 RSI {rsi:.0f}"
                elif rsi is not None and rsi > 75:
                    rsi_str = f"  🟡 RSI {rsi:.0f}"

                alpha_str = ""
                alpha_1y = a.get("alpha_1y")
                if alpha_1y is not None:
                    alpha_emoji = "🟢" if alpha_1y > 0 else ("🟡" if alpha_1y > -20 else "🔴")
                    alpha_str = f"  1Y: {alpha_1y:+.0f}% {alpha_emoji}"
                alpha_3y = a.get("alpha_3y")
                if alpha_3y is not None:
                    alpha_3y_emoji = "🟢" if alpha_3y > 0 else ("🟡" if alpha_3y > -20 else "🔴")
                    alpha_str += f"  3Y: {alpha_3y:+.0f}% {alpha_3y_emoji}"

                shares_str = str(shares)
                post_rotate = a.get("suggested_shares_post_rotate")
                if post_rotate is not None and post_rotate != shares:
                    shares_str += f" (ROTATE後 {post_rotate} 股)"
                sector_tag = f"[{a['sector']}]" if a.get('sector') else ""
                lines.append(f"  #{rank} {a['symbol']}{sector_tag}  建議 {shares_str} @ ${a.get('current_price', 0):.2f}  {momentum}{rsi_str}{alpha_str}")
            for a in pyramid_adds:
                momentum = f"+{a.get('momentum', 0):.1f}%"
                direction_arrow = "↑" if a.get("direction") == "up" else "↓"
                alpha_str = f"  1Y: {a['alpha_1y']:+.0f}%" if a.get("alpha_1y") is not None else ""
                lines.append(f"  [{direction_arrow}第{a['tranche_n']}批] {a['symbol']}  +{a.get('suggested_shares', 0)} 股 @ ${a.get('current_price', 0):.2f}  {momentum}{alpha_str}")
            if backup_adds:
                lines.append("  [備選 — 可替換 1Y/3Y alpha 差的主要候選]")
                for a in backup_adds:
                    momentum = f"+{a.get('momentum', 0):.1f}%" if a.get("momentum") else ""
                    alpha_1y = a.get("alpha_1y")
                    alpha_3y = a.get("alpha_3y")
                    alpha_str = ""
                    if alpha_1y is not None:
                        alpha_emoji = "🟢" if alpha_1y > 0 else ("🟡" if alpha_1y > -20 else "🔴")
                        alpha_str = f"  1Y: {alpha_1y:+.0f}% {alpha_emoji}"
                    if alpha_3y is not None:
                        alpha_3y_emoji = "🟢" if alpha_3y > 0 else ("🟡" if alpha_3y > -20 else "🔴")
                        alpha_str += f"  3Y: {alpha_3y:+.0f}% {alpha_3y_emoji}"
                    sector_tag = f"[{a['sector']}]" if a.get('sector') else ""
                    lines.append(f"  [備#{a.get('momentum_rank', '?')}] {a['symbol']}{sector_tag}  @ ${a.get('current_price', 0):.2f}  {momentum}{alpha_str}")
            lines.append("")

        # ROTATE 建議（汰弱留強）
        rotates = [a for a in actions if a["action"] == "ROTATE"]
        if rotates:
            lines.append(f"ROTATE 建議（汰弱留強）({len(rotates)} 組):")
            for a in rotates:
                sell_pnl = f"{a.get('sell_pnl_pct', 0):+.1f}%" if a.get("sell_pnl_pct") is not None else "N/A"
                buy_alpha_1y = a.get("buy_alpha_1y")
                buy_alpha_3y = a.get("buy_alpha_3y")
                alpha_str = ""
                if buy_alpha_1y is not None:
                    alpha_emoji = "🟢" if buy_alpha_1y > 0 else ("🟡" if buy_alpha_1y > -20 else "🔴")
                    alpha_str = f"1Y: {buy_alpha_1y:+.0f}% {alpha_emoji}"
                if buy_alpha_3y is not None:
                    alpha_3y_emoji = "🟢" if buy_alpha_3y > 0 else ("🟡" if buy_alpha_3y > -20 else "🔴")
                    alpha_str += f"  3Y: {buy_alpha_3y:+.0f}% {alpha_3y_emoji}"
                sell_sector_tag = f"[{a['sell_sector']}]" if a.get('sell_sector') else ""
                buy_sector_tag = f"[{a['buy_sector']}]" if a.get('buy_sector') else ""
                lines.append(f"  賣 {a['sell_symbol']}{sell_sector_tag}  {a['sell_shares']} 股 (動能: {a['sell_momentum']:+.1f}%, P&L: {sell_pnl})")
                lines.append(f"  → 買 {a['buy_symbol']}{buy_sector_tag}  {a['buy_shares']} 股 (動能: +{a['buy_momentum']:.1f}%, {alpha_str})")
                lines.append(f"     {a.get('reason', '')}")
                lines.append("")

        # 需注意
        watch_lines = []
        # 1. 動能轉弱持倉
        weak = [a for a in holds if a.get("momentum") is not None and a.get("momentum") < 0]
        for a in sorted(weak, key=lambda x: x["momentum"]):
            ts = a.get("trend_state", {})
            trend = ts.get("state", "") if ts else ""
            watch_lines.append(f"  ⚠️  {a['symbol']:<6} 動能{a['momentum']:+.1f}% {trend}  P&L: {a.get('pnl_pct', 0):+.1f}%")
        # 2. P&L 偏低持倉（< -3%）
        losing = [a for a in holds if a.get("pnl_pct") is not None and a["pnl_pct"] < -3 and (a.get("momentum") or 0) >= 0]
        for a in sorted(losing, key=lambda x: x["pnl_pct"]):
            stop_price = round(a["avg_price"] * 0.85, 2)
            watch_lines.append(f"  🔴 {a['symbol']:<6} P&L {a['pnl_pct']:+.1f}%  停損線 ${stop_price:.2f}")
        # 3. ROTATE 目標 1Y Alpha 差
        bad_rotates = [a for a in rotates if a.get("buy_alpha_1y") is not None and a["buy_alpha_1y"] < -20]
        for a in bad_rotates:
            watch_lines.append(f"  ⚠️  ROTATE {a['sell_symbol']}→{a['buy_symbol']} 換股目標 1Y落後大盤 {a['buy_alpha_1y']:+.0f}%，建議謹慎")
        if watch_lines:
            lines.append("需注意:")
            lines.extend(watch_lines)
            lines.append("")

        # 台股觀察
        tw_stocks = data.get("tw_stocks", {})
        if tw_stocks:
            scan_count = tw_stocks.get("scan_count", 0)
            lines.append(f"🇹🇼 台股觀察（{scan_count} 檔高流動性股）:")
            leaders = tw_stocks.get("leaders", [])
            if leaders:
                lines.append("  動能領先:")
                for t in leaders:
                    alpha = t.get("alpha_1y")
                    alpha_str = ""
                    if alpha is not None:
                        alpha_emoji = "🟢" if alpha > 0 else ("🟡" if alpha > -10 else "🔴")
                        alpha_str = f"  1Y: {alpha:+.0f}% {alpha_emoji}"
                    lines.append(f"    #{t['rank']} {t['symbol']} {t.get('name', '')} +{t['momentum']:.1f}%{alpha_str}")

            laggards = tw_stocks.get("laggards", [])
            if laggards:
                lines.append("  動能落後:")
                for t in laggards:
                    lines.append(f"    #{t['rank']} {t['symbol']} {t.get('name', '')} {t['momentum']:.1f}%")
            lines.append("")

        return "\n".join(lines)

    def _format_html_report(self, data):
        """產生 HTML 報告"""
        portfolio = data.get("portfolio_snapshot", {})
        sector = data.get("sector_status", {})
        actions = data.get("actions", [])

        # 年度 P&L
        yearly_html = ""
        yearly = portfolio.get("yearly_pnl")
        if yearly:
            sign = "+" if yearly["pnl_amount"] >= 0 else ""
            color = "#28a745" if yearly["pnl_amount"] >= 0 else "#dc3545"
            yearly_html = f'<tr><td>年度 P&L</td><td style="color:{color}">{sign}${yearly["pnl_amount"]:,.2f} ({sign}{yearly["pnl_pct"]:.1f}%)</td></tr>'

        # 市場體制橫幅
        regime = data.get("regime_status", {})
        regime_html = ""
        if regime:
            if regime.get("is_bull", True):
                pct = f"+{regime['pct_vs_ma200']:.1f}%" if regime.get("pct_vs_ma200") is not None else ""
                regime_html = f'<div style="background:#d4edda;padding:10px;border-radius:5px;margin:10px 0;border-left:4px solid #28a745;"><strong>🟢 市場體制: BULL</strong> &nbsp; SPY ${regime.get("spy_price")} &gt; MA200 ${regime.get("ma200")} ({pct})</div>'
            else:
                pct = f"{regime['pct_vs_ma200']:.1f}%" if regime.get("pct_vs_ma200") is not None else ""
                regime_html = f'<div style="background:#f8d7da;padding:10px;border-radius:5px;margin:10px 0;border-left:4px solid #dc3545;"><strong>🔴 市場體制: BEAR</strong> &nbsp; SPY ${regime.get("spy_price")} &lt; MA200 ${regime.get("ma200")} ({pct})<br><span style="color:#721c24;">⚠️ ADD / ROTATE 已暫停，等 SPY 站回 MA200 再開放新倉</span></div>'

        # 板塊警告
        alerts_html = ""
        if sector.get("alerts"):
            alert_items = "".join(f"<li>{a}</li>" for a in sector["alerts"])
            alerts_html = f'<div style="background:#fff3cd;padding:10px;border-radius:5px;margin:10px 0;"><strong>板塊警告</strong><ul style="margin:5px 0;">{alert_items}</ul></div>'

        # Actions 表格
        exits = [a for a in actions if a["action"] == "EXIT"]
        holds = [a for a in actions if a["action"] == "HOLD"]
        new_adds = [a for a in actions if a["action"] == "ADD" and not a.get("is_backup") and not a.get("is_pyramid")]
        pyramid_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_pyramid")]
        backup_adds = [a for a in actions if a["action"] == "ADD" and a.get("is_backup")]
        adds = new_adds
        rotates_sell = {a["sell_symbol"] for a in actions if a["action"] == "ROTATE"}

        # === 持倉總覽表 ===
        # 排序：EXIT 優先，再依動能升序（弱的在前），core 放最後
        def portfolio_sort_key(a):
            if a["action"] == "EXIT":
                return (0, a.get("pnl_pct") or 0)
            if a.get("source") == "core_hold":
                return (2, 0)
            return (1, a.get("momentum") or 0)

        portfolio_rows_data = sorted(exits + holds, key=portfolio_sort_key)

        _stop_type_labels = {"standard": "標準", "tight_2": "緊②", "tight_3": "緊③"}

        def _protect_html(t):
            """計算批次保護期 HTML"""
            t_stop_type = t.get("stop_type", "standard")
            params = TRANCHE_PARAMS.get(t_stop_type, TRANCHE_PARAMS["standard"])
            try:
                entry_date = datetime.date.fromisoformat(t.get("entry_date", ""))
                days_since = (datetime.date.today() - entry_date).days
                days_left = params["protect"] - days_since
                if days_left > 0:
                    return f'<span style="color:#fd7e14;">保護{days_left}d</span>'
                return '<span style="color:#28a745;">可出場</span>'
            except Exception:
                return "—"

        portfolio_rows = ""
        for a in portfolio_rows_data:
            sym = a["symbol"]
            tranches = a.get("tranches") or []
            n_rows = len(tranches) if tranches else 1

            pnl = a.get("pnl_pct")
            momentum = a.get("momentum")
            rank = a.get("momentum_rank")
            ts = a.get("trend_state") or {}
            trend_state = ts.get("state", "")
            price = a.get("current_price", 0)
            avg_price = a.get("avg_price", 0)
            stop_price = round(avg_price * 0.85, 2) if avg_price else 0

            # 決定 Action 標籤
            if a["action"] == "EXIT":
                tranche_str = f" 第{a.get('tranche_n')}批" if a.get("tranche_n") else ""
                action_label = f'<span style="color:#dc3545;font-weight:bold;">⛔ EXIT{tranche_str}</span>'
            elif sym in rotates_sell:
                action_label = '<span style="color:#fd7e14;font-weight:bold;">🔄 ROTATE</span>'
            elif a.get("source") == "core_hold":
                action_label = '<span style="color:#6c757d;">🔒 CORE</span>'
            else:
                action_label = '<span style="color:#28a745;">✅ HOLD</span>'

            # 趨勢標籤
            if trend_state == "轉強":
                trend_label = '<span style="color:#28a745;">↗️轉強</span>'
            elif trend_state == "轉弱":
                trend_label = '<span style="color:#dc3545;">↘️轉弱</span>'
            else:
                trend_label = '<span style="color:#6c757d;">→盤整</span>' if trend_state == "盤整" else ""

            # 動能欄
            if momentum is not None:
                m_color = "#28a745" if momentum > 0 else "#dc3545"
                rank_str = f"#{rank} " if rank else ""
                momentum_str = f'<span style="color:{m_color};">{rank_str}{momentum:+.1f}%</span>'
            else:
                momentum_str = "—"

            # 距高%
            high_price = a.get("high_since_entry")
            if high_price and price and high_price > 0:
                from_high = (price - high_price) / high_price * 100
                if from_high <= -20:
                    fh_color, fh_icon = "#dc3545", "🔴"
                elif from_high <= -10:
                    fh_color, fh_icon = "#fd7e14", "🟡"
                else:
                    fh_color, fh_icon = "#6c757d", ""
                from_high_str = f'<span style="color:{fh_color};">{fh_icon}{from_high:.0f}%</span>'
            else:
                from_high_str = "—"

            # 列底色邏輯
            if a["action"] == "EXIT":
                row_bg = "background:#f8d7da;"
            elif sym in rotates_sell:
                row_bg = "background:#ffe8cc;"
            elif momentum is not None and momentum < 0:
                row_bg = "background:#fff3cd;"
            elif pnl is not None and pnl < -3 and price < stop_price * 1.05:
                row_bg = "background:#fff3cd;"
            elif momentum is not None and momentum > 15 and trend_state == "轉強":
                row_bg = "background:#d4edda;"
            else:
                row_bg = ""

            sector_str = a.get("sector") or "—"

            if not tranches:
                # EXIT 或無 tranches：沿用單行格式，保護期欄顯示 —
                if pnl is not None:
                    pnl_color = "#28a745" if pnl >= 0 else "#dc3545"
                    pnl_str = f'<span style="color:{pnl_color};">{pnl:+.1f}%</span>'
                else:
                    pnl_str = "—"
                portfolio_rows += f'''<tr style="{row_bg}border-bottom:1px solid #eee;">
                    <td style="padding:6px 8px;font-weight:bold;">{sym}</td>
                    <td style="padding:6px 8px;font-size:11px;color:#6c757d;">{sector_str}</td>
                    <td style="padding:6px 8px;text-align:center;font-size:11px;">—</td>
                    <td style="padding:6px 8px;text-align:center;">{a.get("shares", 0)}</td>
                    <td style="padding:6px 8px;text-align:right;">${price:.2f}</td>
                    <td style="padding:6px 8px;text-align:right;">${avg_price:.2f}</td>
                    <td style="padding:6px 8px;text-align:right;">{pnl_str}</td>
                    <td style="padding:6px 8px;text-align:center;">—</td>
                    <td style="padding:6px 8px;text-align:center;">{from_high_str}</td>
                    <td style="padding:6px 8px;text-align:center;">{momentum_str}</td>
                    <td style="padding:6px 8px;text-align:center;">{trend_label}</td>
                    <td style="padding:6px 8px;text-align:center;">{action_label}</td>
                </tr>'''
            else:
                # 有 tranches：每批次一行，標的/板塊/現價/距高/動能/趨勢/建議 用 rowspan
                for i, t in enumerate(tranches):
                    t_entry = t.get("entry_price", avg_price)
                    t_shares = t.get("shares", 0)
                    t_stop_type = t.get("stop_type", "standard")
                    t_n = t.get("n", i + 1)

                    # 批次 P&L（vs 該批進場成本）
                    if price and t_entry:
                        t_pnl = (price - t_entry) / t_entry * 100
                        t_pnl_color = "#28a745" if t_pnl >= 0 else "#dc3545"
                        t_pnl_str = f'<span style="color:{t_pnl_color};">{t_pnl:+.1f}%</span>'
                    else:
                        t_pnl_str = "—"

                    protect_html = _protect_html(t)

                    stop_label = _stop_type_labels.get(t_stop_type, t_stop_type)
                    batch_html = f'#{t_n} <span style="font-size:10px;color:#6c757d;">{stop_label}</span>'

                    is_last = (i == n_rows - 1)
                    row_border = "border-bottom:2px solid #dee2e6;" if is_last else "border-bottom:1px solid #f0f0f0;"

                    if i == 0:
                        portfolio_rows += f'''<tr style="{row_bg}{row_border}">
                            <td style="padding:6px 8px;font-weight:bold;" rowspan="{n_rows}">{sym}</td>
                            <td style="padding:6px 8px;font-size:11px;color:#6c757d;" rowspan="{n_rows}">{sector_str}</td>
                            <td style="padding:6px 8px;text-align:center;font-size:11px;">{batch_html}</td>
                            <td style="padding:6px 8px;text-align:center;">{t_shares}</td>
                            <td style="padding:6px 8px;text-align:right;" rowspan="{n_rows}">${price:.2f}</td>
                            <td style="padding:6px 8px;text-align:right;">${t_entry:.2f}</td>
                            <td style="padding:6px 8px;text-align:right;">{t_pnl_str}</td>
                            <td style="padding:6px 8px;text-align:center;">{protect_html}</td>
                            <td style="padding:6px 8px;text-align:center;" rowspan="{n_rows}">{from_high_str}</td>
                            <td style="padding:6px 8px;text-align:center;" rowspan="{n_rows}">{momentum_str}</td>
                            <td style="padding:6px 8px;text-align:center;" rowspan="{n_rows}">{trend_label}</td>
                            <td style="padding:6px 8px;text-align:center;" rowspan="{n_rows}">{action_label}</td>
                        </tr>'''
                    else:
                        portfolio_rows += f'''<tr style="{row_bg}{row_border}">
                            <td style="padding:6px 8px;text-align:center;font-size:11px;">{batch_html}</td>
                            <td style="padding:6px 8px;text-align:center;">{t_shares}</td>
                            <td style="padding:6px 8px;text-align:right;">${t_entry:.2f}</td>
                            <td style="padding:6px 8px;text-align:right;">{t_pnl_str}</td>
                            <td style="padding:6px 8px;text-align:center;">{protect_html}</td>
                        </tr>'''

        portfolio_html = f'''
        <h3 style="margin-top:20px;">📋 持倉總覽</h3>
        <table style="border-collapse:collapse;width:100%;font-size:13px;">
            <tr style="background:#343a40;color:#fff;">
                <th style="padding:7px 8px;text-align:left;">標的</th>
                <th style="padding:7px 8px;text-align:left;">板塊</th>
                <th style="padding:7px 8px;">批次</th>
                <th style="padding:7px 8px;">股數</th>
                <th style="padding:7px 8px;">現價</th>
                <th style="padding:7px 8px;">進場成本</th>
                <th style="padding:7px 8px;">P&amp;L</th>
                <th style="padding:7px 8px;">保護期</th>
                <th style="padding:7px 8px;">距高</th>
                <th style="padding:7px 8px;">動能</th>
                <th style="padding:7px 8px;">趨勢</th>
                <th style="padding:7px 8px;">建議</th>
            </tr>
            {portfolio_rows}
        </table>
        <p style="font-size:11px;color:#6c757d;margin:4px 0 0 0;">
            🔴 EXIT &nbsp;|&nbsp; 🟠 ROTATE/動能負 &nbsp;|&nbsp; 🟡 接近停損 &nbsp;|&nbsp; 🟢 動能強+轉強 &nbsp;|&nbsp; 保護期=不觸發逐批停損
        </p>'''

        exits_html = ""
        if exits:
            rows = ""
            for a in exits:
                pnl = a.get("pnl_pct", 0)
                pnl_color = "#28a745" if pnl and pnl >= 0 else "#dc3545"
                pnl_str = f"{pnl:+.1f}%" if pnl is not None else "N/A"
                tranche_str = f" 第{a['tranche_n']}批" if a.get("tranche_n") else ""
                rows += f'<tr><td>{a["symbol"]}{tranche_str}</td><td>{a.get("shares", 0)} 股</td><td style="color:{pnl_color}">{pnl_str}</td><td>{a.get("reason", "")}</td></tr>'
            exits_html = f'''
            <div class="section-block">
            <h3 style="color:#dc3545;">EXIT 建議 ({len(exits)} 筆)</h3>
            <table style="border-collapse:collapse;width:100%;">
                <tr style="background:#f8f9fa;"><th style="text-align:left;padding:8px;">標的</th><th>股數</th><th>P&L</th><th>原因</th></tr>
                {rows}
            </table>
            </div>'''

        holds_html = ""
        if holds:
            hold_parts = []
            for a in holds:
                ts = a.get("trend_state")
                if ts and ts["state"] == "轉弱":
                    hold_parts.append(f'<span style="color:#dc3545;">{a["symbol"]}↘️</span>')
                elif ts and ts["state"] == "轉強":
                    hold_parts.append(f'<span style="color:#28a745;">{a["symbol"]}↗️</span>')
                else:
                    hold_parts.append(a["symbol"])
            symbols = ", ".join(hold_parts)
            holds_html = f'<h3 style="color:#6c757d;">HOLD ({len(holds)} 檔)</h3><p>{symbols}</p>'

        adds_html = ""
        if new_adds or pyramid_adds or backup_adds:

            def _add_rsi_html(a):
                rsi = a.get("rsi")
                if rsi is not None and rsi > 80:
                    return f'<td style="color:#dc3545;">🔴 {rsi:.0f}</td>'
                if rsi is not None and rsi > 75:
                    return f'<td style="color:#fd7e14;">🟡 {rsi:.0f}</td>'
                if rsi is not None:
                    return f'<td style="color:#28a745;">{rsi:.0f}</td>'
                return "<td></td>"

            def _add_alpha_html(a):
                alpha_1y = a.get("alpha_1y")
                alpha_3y = a.get("alpha_3y")
                if alpha_1y is None:
                    return "<td></td><td></td>"
                e1 = "🟢" if alpha_1y > 0 else ("🟡" if alpha_1y > -20 else "🔴")
                if alpha_3y is not None:
                    e3 = "🟢" if alpha_3y > 0 else ("🟡" if alpha_3y > -20 else "🔴")
                    td3 = f"<td>{e3} {alpha_3y:+.0f}%</td>"
                else:
                    td3 = "<td></td>"
                return f"<td>{e1} {alpha_1y:+.0f}%</td>{td3}"

            rows = ""
            # 新倉 + 金字塔：依排名升序排序
            primary = sorted(new_adds + pyramid_adds, key=lambda x: x.get("momentum_rank") or 9999)
            for a in primary:
                price = a.get("current_price", 0)
                momentum = f"+{a.get('momentum', 0):.1f}%"
                rank = a.get("momentum_rank", "?")
                rsi_html = _add_rsi_html(a)
                alpha_html = _add_alpha_html(a)
                sector_td = f'<td style="font-size:11px;color:#6c757d;">{a.get("sector") or "—"}</td>'

                if a.get("is_pyramid"):
                    direction_arrow = "↑" if a.get("direction") == "up" else "↓"
                    tranche_label = f'<span style="color:#0d6efd;">{direction_arrow}第{a["tranche_n"]}批</span>'
                    shares_str = str(a.get("suggested_shares", 0))
                    post_rotate = a.get("suggested_shares_post_rotate")
                    if post_rotate is not None and post_rotate != a.get("suggested_shares", 0):
                        shares_str += f'<br><span style="color:#fd7e14;font-size:11px;">ROTATE後 {post_rotate} 股</span>'
                    rows += f'<tr style="background:#e8f4ff;"><td>#{rank}</td><td><strong>{a["symbol"]}</strong> {tranche_label}</td>{sector_td}<td>{shares_str}</td><td>${price:.2f}</td><td>{momentum}</td>{rsi_html}{alpha_html}</tr>'
                else:
                    shares_str = str(a.get("suggested_shares", 0))
                    post_rotate = a.get("suggested_shares_post_rotate")
                    if post_rotate is not None and post_rotate != a.get("suggested_shares", 0):
                        shares_str += f'<br><span style="color:#fd7e14;font-size:11px;">ROTATE後 {post_rotate} 股</span>'
                    rows += f'<tr><td>#{rank}</td><td>{a["symbol"]}</td>{sector_td}<td>{shares_str}</td><td>${price:.2f}</td><td>{momentum}</td>{rsi_html}{alpha_html}</tr>'

            for a in backup_adds:
                price = a.get("current_price", 0)
                momentum = f"+{a.get('momentum', 0):.1f}%"
                rsi_html = _add_rsi_html(a)
                alpha_html = _add_alpha_html(a)
                sector_td = f'<td style="font-size:11px;color:#6c757d;">{a.get("sector") or "—"}</td>'
                rows += f'<tr style="background:#fff9e6;"><td style="color:#856404;">備#{a.get("momentum_rank", "?")}</td><td>{a["symbol"]}</td>{sector_td}<td style="color:#6c757d;font-size:11px;">備選參考</td><td>${price:.2f}</td><td>{momentum}</td>{rsi_html}{alpha_html}</tr>'

            adds_html = f'''
            <div class="section-block">
            <h3 style="color:#28a745;">ADD 建議 ({len(new_adds)} 新倉 + {len(pyramid_adds)} 金字塔 + {len(backup_adds)} 備選)</h3>
            <table style="border-collapse:collapse;width:100%;">
                <tr style="background:#f8f9fa;"><th style="padding:8px;text-align:left;">排名</th><th style="text-align:left;">標的</th><th style="text-align:left;">板塊</th><th>建議股數</th><th>目前價格</th><th>動能</th><th>RSI</th><th>1Y vs SPY</th><th>3Y vs SPY</th></tr>
                {rows}
            </table>
            </div>'''

        # ROTATE 建議（汰弱留強）
        rotates = [a for a in actions if a["action"] == "ROTATE"]
        rotates_html = ""
        if rotates:
            rows = ""
            for a in rotates:
                sell_pnl = a.get("sell_pnl_pct", 0)
                sell_pnl_color = "#28a745" if sell_pnl and sell_pnl >= 0 else "#dc3545"
                sell_pnl_str = f"{sell_pnl:+.1f}%" if sell_pnl is not None else "N/A"

                buy_alpha_1y = a.get("buy_alpha_1y")
                buy_alpha_3y = a.get("buy_alpha_3y")
                alpha_str = ""
                alpha_3y_str = ""
                if buy_alpha_1y is not None:
                    alpha_emoji = "🟢" if buy_alpha_1y > 0 else ("🟡" if buy_alpha_1y > -20 else "🔴")
                    alpha_str = f"{alpha_emoji} {buy_alpha_1y:+.0f}%"
                if buy_alpha_3y is not None:
                    alpha_3y_emoji = "🟢" if buy_alpha_3y > 0 else ("🟡" if buy_alpha_3y > -20 else "🔴")
                    alpha_3y_str = f"{alpha_3y_emoji} {buy_alpha_3y:+.0f}%"

                sell_sector = a.get("sell_sector") or "—"
                buy_sector = a.get("buy_sector") or "—"
                rows += f'''<tr style="border-bottom:1px solid #ddd;">
                    <td style="padding:8px;color:#dc3545;">賣 {a["sell_symbol"]}</td>
                    <td style="font-size:11px;color:#6c757d;">{sell_sector}</td>
                    <td>{a["sell_shares"]} 股</td>
                    <td>{a["sell_momentum"]:+.1f}%</td>
                    <td style="color:{sell_pnl_color}">{sell_pnl_str}</td>
                    <td style="color:#28a745;">→ 買 {a["buy_symbol"]}</td>
                    <td style="font-size:11px;color:#6c757d;">{buy_sector}</td>
                    <td>{a["buy_shares"]} 股</td>
                    <td>+{a["buy_momentum"]:.1f}%</td>
                    <td>{alpha_str}</td>
                    <td>{alpha_3y_str}</td>
                </tr>'''
            rotates_html = f'''
            <div class="section-block">
            <h3 style="color:#fd7e14;">ROTATE 建議（汰弱留強）({len(rotates)} 組)</h3>
            <table style="border-collapse:collapse;width:100%;">
                <tr style="background:#f8f9fa;"><th style="padding:8px;">賣出</th><th>板塊</th><th>股數</th><th>動能</th><th>P&L</th><th>買入</th><th>板塊</th><th>股數</th><th>動能</th><th>1Y</th><th>3Y</th></tr>
                {rows}
            </table>
            </div>'''

        topups_html = ""

        # 需注意
        watch_items = []
        weak = [a for a in actions if a["action"] == "HOLD" and (a.get("momentum") or 0) < 0]
        for a in sorted(weak, key=lambda x: x.get("momentum", 0)):
            ts = a.get("trend_state", {}) or {}
            trend = ts.get("state", "")
            watch_items.append(f'<li>⚠️ <strong>{a["symbol"]}</strong> 動能{a.get("momentum", 0):+.1f}% {trend}，P&L: {a.get("pnl_pct", 0):+.1f}%</li>')
        losing = [a for a in actions if a["action"] == "HOLD" and (a.get("pnl_pct") or 0) < -3 and (a.get("momentum") or 0) >= 0]
        for a in sorted(losing, key=lambda x: x.get("pnl_pct", 0)):
            stop_price = round(a["avg_price"] * 0.85, 2)
            watch_items.append(f'<li>🔴 <strong>{a["symbol"]}</strong> P&L {a.get("pnl_pct", 0):+.1f}%，停損線 ${stop_price:.2f}</li>')
        rotates_list = [a for a in actions if a["action"] == "ROTATE"]
        for a in rotates_list:
            if (a.get("buy_alpha_1y") or 0) < -20:
                watch_items.append(f'<li>⚠️ ROTATE <strong>{a["sell_symbol"]}→{a["buy_symbol"]}</strong> 換股目標 1Y落後大盤 {a.get("buy_alpha_1y", 0):+.0f}%，建議謹慎</li>')
        watch_html = ""
        if watch_items:
            items_str = "".join(watch_items)
            watch_html = f'<div style="background:#f8d7da;padding:12px;border-radius:5px;margin:10px 0;"><strong>需注意</strong><ul style="margin:5px 0;">{items_str}</ul></div>'

        # 台股觀察
        tw_stocks = data.get("tw_stocks", {})
        tw_stocks_html = ""
        if tw_stocks:
            leaders = tw_stocks.get("leaders", [])
            laggards = tw_stocks.get("laggards", [])

            leader_rows = ""
            for t in leaders:
                alpha = t.get("alpha_1y")
                alpha_str = ""
                if alpha is not None:
                    alpha_emoji = "🟢" if alpha > 0 else ("🟡" if alpha > -10 else "🔴")
                    alpha_str = f"{alpha_emoji} {alpha:+.0f}%"
                leader_rows += f'<tr><td style="padding:4px;">#{t["rank"]}</td><td>{t["symbol"]}</td><td>{t.get("name", "")}</td><td style="color:#28a745;">+{t["momentum"]:.1f}%</td><td>{alpha_str}</td></tr>'

            laggard_rows = ""
            for t in laggards:
                laggard_rows += f'<tr><td style="padding:4px;">#{t["rank"]}</td><td>{t["symbol"]}</td><td>{t.get("name", "")}</td><td style="color:#dc3545;">{t["momentum"]:.1f}%</td><td></td></tr>'

            tw_stocks_html = f'''
            <h3>🇹🇼 台股觀察（{tw_stocks.get("scan_count", 0)} 檔高流動性股）</h3>
            <table style="border-collapse:collapse;width:100%;">
                <tr style="background:#f8f9fa;"><th style="padding:8px;">排名</th><th>代碼</th><th>名稱</th><th>動能</th><th>1Y vs 0050</th></tr>
                <tr><td colspan="5" style="background:#d4edda;padding:4px;"><strong>動能領先</strong></td></tr>
                {leader_rows}
                <tr><td colspan="5" style="background:#f8d7da;padding:4px;"><strong>動能落後</strong></td></tr>
                {laggard_rows}
            </table>'''

        html = f'''
        <html>
        <body style="font-family:Arial,sans-serif;padding:16px;color:#333;">
            <h2>盤前報告 {data["date"]}</h2>
            <p style="color:#6c757d;">版本 {data.get("version", "N/A")}</p>

            <table style="border-collapse:collapse;width:100%;margin:20px 0;">
                <tr><td style="padding:8px;border-bottom:1px solid #ddd;">投組總值</td><td style="padding:8px;border-bottom:1px solid #ddd;"><strong>${portfolio.get("total_value", 0):,.2f}</strong></td></tr>
                <tr><td style="padding:8px;border-bottom:1px solid #ddd;">現金</td><td style="padding:8px;border-bottom:1px solid #ddd;">${portfolio.get("cash", 0):,.2f}</td></tr>
                <tr><td style="padding:8px;border-bottom:1px solid #ddd;">個股</td><td style="padding:8px;border-bottom:1px solid #ddd;">{portfolio.get("individual_count", 0)}/30 檔</td></tr>
                {yearly_html}
            </table>

            {regime_html}
            {alerts_html}
            {watch_html}
            {portfolio_html}
            {exits_html}
            {holds_html}
            {adds_html}
            {rotates_html}
            {topups_html}
            {tw_stocks_html}

            <hr style="margin:30px 0;border:none;border-top:1px solid #ddd;">
            <p style="color:#6c757d;font-size:12px;">此郵件由盤前建議系統自動發送</p>
        </body>
        </html>
        '''
        return html

    def _send_email(self, subject, text_body, html_body=None, attachments=None):
        """發送郵件

        Args:
            subject: 郵件主旨
            text_body: 純文字內容
            html_body: HTML 內容（可選）
            attachments: list of (filename, bytes)，附件（可選）

        Returns:
            bool: 是否成功
        """
        try:
            if attachments:
                msg = MIMEMultipart("mixed")
                msg["Subject"] = subject
                msg["From"] = self.sender
                msg["To"] = self.recipient

                body_part = MIMEMultipart("alternative")
                body_part.attach(MIMEText(text_body, "plain", "utf-8"))
                if html_body:
                    body_part.attach(MIMEText(html_body, "html", "utf-8"))
                msg.attach(body_part)

                for filename, data in attachments:
                    part = MIMEApplication(data, Name=filename)
                    part["Content-Disposition"] = f'attachment; filename="{filename}"'
                    msg.attach(part)
            else:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = self.sender
                msg["To"] = self.recipient

                msg.attach(MIMEText(text_body, "plain", "utf-8"))
                if html_body:
                    msg.attach(MIMEText(html_body, "html", "utf-8"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.sender, self.password)
                server.sendmail(self.sender, [self.recipient], msg.as_string())

            return True

        except smtplib.SMTPAuthenticationError:
            print("郵件發送失敗: Gmail 認證錯誤，請檢查 App Password")
            return False
        except Exception as e:
            print(f"郵件發送失敗: {e}")
            return False
