"""Gmail SMTP 通知模組"""
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

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

        subject = f"盤前報告 {actions_data['date']} | 投組 ${total_value:,.0f}"
        text_body = self._format_text_report(actions_data)
        html_body = self._format_html_report(actions_data)

        return self._send_email(subject, text_body, html_body)

    def _format_text_report(self, data):
        """產生純文字報告"""
        portfolio = data.get("portfolio_snapshot", {})
        sector = data.get("sector_status", {})
        actions = data.get("actions", [])

        lines = [
            f"盤前報告 {data['date']}",
            f"版本 {data.get('version', 'N/A')}",
            "=" * 40,
            "",
            f"投組總值: ${portfolio.get('total_value', 0):,.2f}",
            f"現金:     ${portfolio.get('cash', 0):,.2f}",
            f"個股:     {portfolio.get('individual_count', 0)}/30 檔",
        ]

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
        adds = [a for a in actions if a["action"] == "ADD"]

        if exits:
            lines.append(f"EXIT 建議 ({len(exits)} 檔):")
            for a in exits:
                pnl = f"{a.get('pnl_pct', 0):+.1f}%" if a.get("pnl_pct") is not None else "N/A"
                lines.append(f"  {a['symbol']:<6} {a.get('shares', 0):>4} 股  {pnl:<8} {a.get('reason', '')}")
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

        if adds:
            lines.append(f"ADD 建議 ({len(adds)} 檔):")
            for a in adds:
                momentum = f"+{a.get('momentum', 0):.1f}%" if a.get("momentum") else ""
                rank = a.get("momentum_rank", "?")
                shares = a.get("suggested_shares", 0)

                # Format RSI warning
                rsi_str = ""
                rsi = a.get("rsi")
                if rsi is not None and rsi > 80:
                    rsi_str = f"  🔴 RSI {rsi:.0f}"
                elif rsi is not None and rsi > 75:
                    rsi_str = f"  🟡 RSI {rsi:.0f}"

                # Format alpha_1y
                alpha_str = ""
                alpha_1y = a.get("alpha_1y")
                if alpha_1y is not None:
                    alpha_emoji = "🟢" if alpha_1y > 0 else ("🟡" if alpha_1y > -20 else "🔴")
                    alpha_str = f"  1Y: {alpha_1y:+.0f}% {alpha_emoji}"

                lines.append(f"  #{rank} {a['symbol']:<6} 建議 {shares} 股 @ ${a.get('current_price', 0):.2f}  {momentum}{rsi_str}{alpha_str}")
            lines.append("")

        # ROTATE 建議（汰弱留強）
        rotates = [a for a in actions if a["action"] == "ROTATE"]
        if rotates:
            lines.append(f"ROTATE 建議（汰弱留強）({len(rotates)} 組):")
            for a in rotates:
                sell_pnl = f"{a.get('sell_pnl_pct', 0):+.1f}%" if a.get("sell_pnl_pct") is not None else "N/A"
                buy_alpha = a.get("buy_alpha_1y")
                alpha_str = ""
                if buy_alpha is not None:
                    alpha_emoji = "🟢" if buy_alpha > 0 else ("🟡" if buy_alpha > -20 else "🔴")
                    alpha_str = f"1Y: {buy_alpha:+.0f}% {alpha_emoji}"
                lines.append(f"  賣 {a['sell_symbol']:<6} {a['sell_shares']} 股 (動能: {a['sell_momentum']:+.1f}%, P&L: {sell_pnl})")
                lines.append(f"  → 買 {a['buy_symbol']:<6} {a['buy_shares']} 股 (動能: +{a['buy_momentum']:.1f}%, {alpha_str})")
                lines.append(f"     {a.get('reason', '')}")
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

        # 板塊警告
        alerts_html = ""
        if sector.get("alerts"):
            alert_items = "".join(f"<li>{a}</li>" for a in sector["alerts"])
            alerts_html = f'<div style="background:#fff3cd;padding:10px;border-radius:5px;margin:10px 0;"><strong>板塊警告</strong><ul style="margin:5px 0;">{alert_items}</ul></div>'

        # Actions 表格
        exits = [a for a in actions if a["action"] == "EXIT"]
        holds = [a for a in actions if a["action"] == "HOLD"]
        adds = [a for a in actions if a["action"] == "ADD"]

        exits_html = ""
        if exits:
            rows = ""
            for a in exits:
                pnl = a.get("pnl_pct", 0)
                pnl_color = "#28a745" if pnl and pnl >= 0 else "#dc3545"
                pnl_str = f"{pnl:+.1f}%" if pnl is not None else "N/A"
                rows += f'<tr><td>{a["symbol"]}</td><td>{a.get("shares", 0)} 股</td><td style="color:{pnl_color}">{pnl_str}</td><td>{a.get("reason", "")}</td></tr>'
            exits_html = f'''
            <h3 style="color:#dc3545;">EXIT 建議 ({len(exits)} 檔)</h3>
            <table style="border-collapse:collapse;width:100%;">
                <tr style="background:#f8f9fa;"><th style="text-align:left;padding:8px;">標的</th><th>股數</th><th>P&L</th><th>原因</th></tr>
                {rows}
            </table>'''

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
        if adds:
            rows = ""
            for a in adds:
                momentum = f"+{a.get('momentum', 0):.1f}%" if a.get("momentum") else ""
                shares = a.get("suggested_shares", 0)
                price = a.get("current_price", 0)

                # Format RSI warning for HTML
                rsi_html = "<td></td>"
                rsi = a.get("rsi")
                if rsi is not None and rsi > 80:
                    rsi_html = f'<td style="color:#dc3545;">🔴 {rsi:.0f}</td>'
                elif rsi is not None and rsi > 75:
                    rsi_html = f'<td style="color:#fd7e14;">🟡 {rsi:.0f}</td>'
                elif rsi is not None:
                    rsi_html = f'<td style="color:#28a745;">{rsi:.0f}</td>'

                # Format alpha_1y for HTML
                alpha_html = "<td></td>"
                alpha_1y = a.get("alpha_1y")
                if alpha_1y is not None:
                    alpha_emoji = "🟢" if alpha_1y > 0 else ("🟡" if alpha_1y > -20 else "🔴")
                    alpha_html = f"<td>{alpha_emoji} {alpha_1y:+.0f}%</td>"

                rows += f'<tr><td>#{a.get("momentum_rank", "?")}</td><td>{a["symbol"]}</td><td>{shares} 股</td><td>${price:.2f}</td><td>{momentum}</td>{rsi_html}{alpha_html}</tr>'
            adds_html = f'''
            <h3 style="color:#28a745;">ADD 建議 ({len(adds)} 檔)</h3>
            <table style="border-collapse:collapse;width:100%;">
                <tr style="background:#f8f9fa;"><th style="padding:8px;">排名</th><th>標的</th><th>建議股數</th><th>目前價格</th><th>動能</th><th>RSI</th><th>1Y vs SPY</th></tr>
                {rows}
            </table>'''

        # ROTATE 建議（汰弱留強）
        rotates = [a for a in actions if a["action"] == "ROTATE"]
        rotates_html = ""
        if rotates:
            rows = ""
            for a in rotates:
                sell_pnl = a.get("sell_pnl_pct", 0)
                sell_pnl_color = "#28a745" if sell_pnl and sell_pnl >= 0 else "#dc3545"
                sell_pnl_str = f"{sell_pnl:+.1f}%" if sell_pnl is not None else "N/A"

                buy_alpha = a.get("buy_alpha_1y")
                alpha_str = ""
                if buy_alpha is not None:
                    alpha_emoji = "🟢" if buy_alpha > 0 else ("🟡" if buy_alpha > -20 else "🔴")
                    alpha_str = f"{alpha_emoji} {buy_alpha:+.0f}%"

                rows += f'''<tr style="border-bottom:1px solid #ddd;">
                    <td style="padding:8px;color:#dc3545;">賣 {a["sell_symbol"]}</td>
                    <td>{a["sell_shares"]} 股</td>
                    <td>{a["sell_momentum"]:+.1f}%</td>
                    <td style="color:{sell_pnl_color}">{sell_pnl_str}</td>
                    <td style="color:#28a745;">→ 買 {a["buy_symbol"]}</td>
                    <td>{a["buy_shares"]} 股</td>
                    <td>+{a["buy_momentum"]:.1f}%</td>
                    <td>{alpha_str}</td>
                </tr>'''
            rotates_html = f'''
            <h3 style="color:#fd7e14;">ROTATE 建議（汰弱留強）({len(rotates)} 組)</h3>
            <table style="border-collapse:collapse;width:100%;">
                <tr style="background:#f8f9fa;"><th style="padding:8px;">賣出</th><th>股數</th><th>動能</th><th>P&L</th><th>買入</th><th>股數</th><th>動能</th><th>1Y</th></tr>
                {rows}
            </table>'''

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
        <body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;">
            <h2>盤前報告 {data["date"]}</h2>
            <p style="color:#6c757d;">版本 {data.get("version", "N/A")}</p>

            <table style="border-collapse:collapse;width:100%;margin:20px 0;">
                <tr><td style="padding:8px;border-bottom:1px solid #ddd;">投組總值</td><td style="padding:8px;border-bottom:1px solid #ddd;"><strong>${portfolio.get("total_value", 0):,.2f}</strong></td></tr>
                <tr><td style="padding:8px;border-bottom:1px solid #ddd;">現金</td><td style="padding:8px;border-bottom:1px solid #ddd;">${portfolio.get("cash", 0):,.2f}</td></tr>
                <tr><td style="padding:8px;border-bottom:1px solid #ddd;">個股</td><td style="padding:8px;border-bottom:1px solid #ddd;">{portfolio.get("individual_count", 0)}/30 檔</td></tr>
                {yearly_html}
            </table>

            {alerts_html}
            {exits_html}
            {holds_html}
            {adds_html}
            {rotates_html}
            {tw_stocks_html}

            <hr style="margin:30px 0;border:none;border-top:1px solid #ddd;">
            <p style="color:#6c757d;font-size:12px;">此郵件由盤前建議系統自動發送</p>
        </body>
        </html>
        '''
        return html

    def _send_email(self, subject, text_body, html_body=None):
        """發送郵件

        Args:
            subject: 郵件主旨
            text_body: 純文字內容
            html_body: HTML 內容（可選）

        Returns:
            bool: 是否成功
        """
        try:
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
