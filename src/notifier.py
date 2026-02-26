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

        safe_topups = data.get("safe_topups", [])
        if adds or safe_topups:
            lines.append(f"ADD / TOPUP 建議 ({len(adds)} 新倉 + {len(safe_topups)} 增持):")
            for a in adds:
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

                shares_str = str(shares)
                post_rotate = a.get("suggested_shares_post_rotate")
                if post_rotate is not None and post_rotate != shares:
                    shares_str += f" (ROTATE後 {post_rotate} 股)"
                lines.append(f"  #{rank} {a['symbol']:<6} 建議 {shares_str} @ ${a.get('current_price', 0):.2f}  {momentum}{rsi_str}{alpha_str}")
            for s in safe_topups:
                momentum = f"+{s.get('momentum', 0):.1f}%(#{s.get('momentum_rank', '?')})"
                alpha_str = f"  1Y: {s['alpha_1y']:+.0f}%" if s.get("alpha_1y") is not None else ""
                lines.append(f"  [增持] {s['symbol']:<6} +{s['topup_shares']} 股 @ ${s['current_price']:.2f}  {momentum}  {s['current_weight_pct']:.1f}%→等權重  🟢 安全{alpha_str}")
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

        # TOPUP 增持參考
        topups = data.get("topup_suggestions", [])
        if topups:
            lines.append(f"TOPUP 增持參考（倉位<2%、動能強、趨勢轉強）({len(topups)} 檔):")
            for s in topups:
                lines.append(f"  {s['symbol']:<6} 倉位{s['current_weight_pct']:.1f}%  動能+{s['momentum']:.1f}%(#{s['momentum_rank']})  追高{s['run_up_pct']:+.1f}%  {s['safety']}")
                lines.append(f"         現價${s['current_price']:.2f}  成本${s['avg_price']:.2f}  新停損${s['new_stop']:.2f}  {s['safety_note']}")
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

        safe_topups = data.get("safe_topups", [])
        adds_html = ""
        if adds or safe_topups:
            rows = ""
            for a in adds:
                momentum = f"+{a.get('momentum', 0):.1f}%" if a.get("momentum") else ""
                shares = a.get("suggested_shares", 0)
                price = a.get("current_price", 0)

                rsi_html = "<td></td>"
                rsi = a.get("rsi")
                if rsi is not None and rsi > 80:
                    rsi_html = f'<td style="color:#dc3545;">🔴 {rsi:.0f}</td>'
                elif rsi is not None and rsi > 75:
                    rsi_html = f'<td style="color:#fd7e14;">🟡 {rsi:.0f}</td>'
                elif rsi is not None:
                    rsi_html = f'<td style="color:#28a745;">{rsi:.0f}</td>'

                alpha_html = "<td></td>"
                alpha_1y = a.get("alpha_1y")
                if alpha_1y is not None:
                    alpha_emoji = "🟢" if alpha_1y > 0 else ("🟡" if alpha_1y > -20 else "🔴")
                    alpha_html = f"<td>{alpha_emoji} {alpha_1y:+.0f}%</td>"

                shares_str = str(shares)
                post_rotate = a.get("suggested_shares_post_rotate")
                if post_rotate is not None and post_rotate != shares:
                    shares_str += f'<br><span style="color:#fd7e14;font-size:11px;">ROTATE後 {post_rotate} 股</span>'
                rows += f'<tr><td>#{a.get("momentum_rank", "?")}</td><td>{a["symbol"]}</td><td>{shares_str}</td><td>${price:.2f}</td><td>{momentum}</td>{rsi_html}{alpha_html}</tr>'

            for s in safe_topups:
                momentum = f"+{s.get('momentum', 0):.1f}%(#{s.get('momentum_rank', '?')})"
                alpha_html = "<td></td>"
                if s.get("alpha_1y") is not None:
                    alpha_emoji = "🟢" if s["alpha_1y"] > 0 else ("🟡" if s["alpha_1y"] > -20 else "🔴")
                    alpha_html = f"<td>{alpha_emoji} {s['alpha_1y']:+.0f}%</td>"
                rows += f'<tr style="background:#f0fff0;"><td style="color:#28a745;">增持</td><td><strong>{s["symbol"]}</strong></td><td>+{s["topup_shares"]} 股<br><span style="font-size:11px;color:#28a745;">{s["current_weight_pct"]:.1f}%→等權重 🟢</span></td><td>${s["current_price"]:.2f}</td><td>{momentum}</td><td></td>{alpha_html}</tr>'

            adds_html = f'''
            <h3 style="color:#28a745;">ADD / TOPUP 建議 ({len(adds)} 新倉 + {len(safe_topups)} 增持)</h3>
            <table style="border-collapse:collapse;width:100%;">
                <tr style="background:#f8f9fa;"><th style="padding:8px;">類型</th><th>標的</th><th>建議股數</th><th>目前價格</th><th>動能</th><th>RSI</th><th>1Y vs SPY</th></tr>
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

        # TOPUP 增持參考
        topups = data.get("topup_suggestions", [])
        topups_html = ""
        if topups:
            rows = ""
            for s in topups:
                safety_color = "#28a745" if "安全" in s["safety"] else ("#fd7e14" if "謹慎" in s["safety"] else "#dc3545")
                rows += f'''<tr style="border-bottom:1px solid #eee;">
                    <td style="padding:6px;"><strong>{s["symbol"]}</strong></td>
                    <td>{s["current_weight_pct"]:.1f}%</td>
                    <td>+{s["momentum"]:.1f}% (#{s["momentum_rank"]})</td>
                    <td>{s["run_up_pct"]:+.1f}%</td>
                    <td style="color:{safety_color};">{s["safety"]}</td>
                    <td>${s["current_price"]:.2f}</td>
                    <td>${s["avg_price"]:.2f}</td>
                    <td>${s["new_stop"]:.2f}</td>
                </tr>'''
            topups_html = f'''
            <h3 style="color:#6f42c1;">TOPUP 增持參考（倉位&lt;2%、動能強、趨勢轉強）</h3>
            <table style="border-collapse:collapse;width:100%;font-size:13px;">
                <tr style="background:#f8f9fa;"><th style="padding:6px;text-align:left;">標的</th><th>倉位</th><th>動能</th><th>追高</th><th>安全度</th><th>現價</th><th>成本</th><th>新停損</th></tr>
                {rows}
            </table>'''

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
            {watch_html}
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
