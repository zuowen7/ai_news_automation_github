"""
邮件模板模块
提供美观的HTML邮件模板
"""

from typing import List, Dict, Any
from datetime import datetime

from ..fetchers.base import NewsItem
from ..config.constants import COLOR_THEME


class EmailTemplate:
    """邮件模板基类"""

    def __init__(self, color_theme: Dict[str, str] = None):
        self.colors = color_theme or COLOR_THEME

    def render(self, **kwargs) -> str:
        """渲染模板"""
        raise NotImplementedError


class EditorialNewsTemplate(EmailTemplate):
    """
    现代编辑风格邮件模板
    灵感来自高质量新闻简报，简洁专业，注重阅读体验
    """

    def render(
        self,
        news_list: List[NewsItem],
        ai_summary: str = "",
        ai_trends: str = "",
        date_str: str = None
    ) -> str:
        """渲染现代编辑风格HTML邮件"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y年%m月%d日')

        # 分类新闻 - GitHub/HF单独显示，其他作为常规新闻
        github_news = [n for n in news_list if n.news_type == "github"]
        hf_news = [n for n in news_list if n.news_type == "huggingface"]
        # 常规新闻包括 type: "news" 和 type: "rss"
        domestic_news = [n for n in news_list if n.region == "domestic" and n.news_type in ["news", "rss"]]
        global_news = [n for n in news_list if n.region == "global" and n.news_type in ["news", "rss"]]

        stats = {
            "total": len(news_list),
            "domestic": len(domestic_news),
            "global": len(global_news),
            "github": len(github_news),
            "huggingface": len(hf_news)
        }

        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI新闻日报 - {date_str}</title>
    <!--[if mso]>
    <style type="text/css">
        body, table, td {{font-family: Arial, sans-serif !important;}}
    </style>
    <![endif]-->
</head>
<body style="margin:0;padding:0;background:#fafafa;font-family:'Georgia','Times New Roman',serif;line-height:1.6;">
    <!-- 外层容器 -->
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#fafafa;padding:40px 20px;">
        <tr>
            <td align="center">
                <!-- 主容器 -->
                <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;max-width:600px;border:1px solid #e8e8e8;">

                    <!-- 头部 -->
                    <tr>
                        <td style="padding:50px 40px 30px 40px;border-bottom:1px solid #e8e8e8;">
                            <table width="100%" cellpadding="0" cellspacing="0">
                                <tr>
                                    <td>
                                        <h1 style="margin:0;font-size:28px;font-weight:400;letter-spacing:2px;color:#1a1a1a;font-family:'Helvetica Neue',Arial,sans-serif;">AI新闻日报</h1>
                                        <p style="margin:8px 0 0 0;font-size:14px;color:#888;font-family:'Helvetica Neue',Arial,sans-serif;">{date_str}</p>
                                    </td>
                                    <td style="text-align:right;">
                                        <span style="display:inline-block;padding:6px 12px;background:#1a1a1a;color:#fff;font-size:11px;letter-spacing:1px;font-family:'Helvetica Neue',Arial,sans-serif;">DAILY</span>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- AI洞察 -->
                    {self._render_insights(ai_summary, ai_trends) if ai_summary or ai_trends else ''}

                    <!-- 统计 -->
                    <tr>
                        <td style="padding:30px 40px;background:#f8f8f8;">
                            <table width="100%" cellpadding="0" cellspacing="0">
                                <tr>
                                    <td style="width:33%;text-align:center;padding:10px;border-right:1px solid #e8e8e8;">
                                        <div style="font-size:36px;font-weight:300;color:#1a1a1a;font-family:'Helvetica Neue',Arial,sans-serif;">{stats['total']}</div>
                                        <div style="font-size:11px;color:#888;margin-top:5px;letter-spacing:1px;font-family:'Helvetica Neue',Arial,sans-serif;">总新闻</div>
                                    </td>
                                    <td style="width:33%;text-align:center;padding:10px;border-right:1px solid #e8e8e8;">
                                        <div style="font-size:36px;font-weight:300;color:#1a1a1a;font-family:'Helvetica Neue',Arial,sans-serif;">{stats['domestic']}</div>
                                        <div style="font-size:11px;color:#888;margin-top:5px;letter-spacing:1px;font-family:'Helvetica Neue',Arial,sans-serif;">国内</div>
                                    </td>
                                    <td style="width:34%;text-align:center;padding:10px;">
                                        <div style="font-size:36px;font-weight:300;color:#1a1a1a;font-family:'Helvetica Neue',Arial,sans-serif;">{stats['global']}</div>
                                        <div style="font-size:11px;color:#888;margin-top:5px;letter-spacing:1px;font-family:'Helvetica Neue',Arial,sans-serif;">国际</div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- GitHub热门项目 -->
                    {self._render_github_section(github_news) if github_news else ''}

                    <!-- Hugging Face热门模型 -->
                    {self._render_huggingface_section(hf_news) if hf_news else ''}

                    <!-- 国内新闻 -->
                    <tr>
                        <td style="padding:40px 40px 20px 40px;">
                            <h2 style="margin:0 0 20px 0;font-size:14px;letter-spacing:2px;color:#888;border-bottom:1px solid #1a1a1a;padding-bottom:10px;font-family:'Helvetica Neue',Arial,sans-serif;">国内AI新闻</h2>
                            {self._render_news_list(domestic_news, 1)}
                        </td>
                    </tr>

                    <!-- 国际新闻 -->
                    <tr>
                        <td style="padding:20px 40px 40px 40px;">
                            <h2 style="margin:0 0 20px 0;font-size:14px;letter-spacing:2px;color:#888;border-bottom:1px solid #1a1a1a;padding-bottom:10px;font-family:'Helvetica Neue',Arial,sans-serif;">国际AI新闻</h2>
                            {self._render_news_list(global_news, len(domestic_news) + 1)}
                        </td>
                    </tr>

                    <!-- 页脚 -->
                    <tr>
                        <td style="padding:30px 40px;background:#1a1a1a;text-align:center;">
                            <p style="margin:0 0 10px 0;font-size:12px;color:#666;font-family:'Helvetica Neue',Arial,sans-serif;">由AI新闻自动化系统生成</p>
                            <p style="margin:0;font-size:11px;color:#444;font-family:'Helvetica Neue',Arial,sans-serif;">数据来源：量子位、TechCrunch、VentureBeat、ArsTechnica等</p>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>'''

    def _render_insights(self, summary: str, trends: str) -> str:
        """渲染AI洞察区域"""
        return f'''<tr>
        <td style="padding:40px 40px 30px 40px;background:#f8f8f8;">
            <h2 style="margin:0 0 25px 0;font-size:12px;letter-spacing:2px;color:#888;text-transform:uppercase;font-family:'Helvetica Neue',Arial,sans-serif;">AI洞察</h2>

            {self._insight_card("今日摘要", summary) if summary else ''}
            {self._insight_card("趋势分析", trends) if trends else ''}
        </td>
    </tr>'''

    def _insight_card(self, title: str, content: str) -> str:
        """渲染洞察卡片"""
        return f'''<div style="background:#fff;border:1px solid #e8e8e8;padding:25px;margin-bottom:20px;">
            <h3 style="margin:0 0 12px 0;font-size:15px;font-weight:500;color:#1a1a1a;font-family:'Helvetica Neue',Arial,sans-serif;">{title}</h3>
            <p style="margin:0;font-size:15px;line-height:1.8;color:#333;white-space:pre-line;">{content}</p>
        </div>'''

    def _render_github_section(self, github_news: List[NewsItem]) -> str:
        """渲染GitHub热门项目栏目"""
        return f'''<tr>
        <td style="padding:40px 40px 20px 40px;">
            <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                    <td>
                        <h2 style="margin:0 0 20px 0;font-size:14px;letter-spacing:2px;color:#888;border-bottom:1px solid #1a1a1a;padding-bottom:10px;font-family:'Helvetica Neue',Arial,sans-serif;">
                            <span style="display:inline-block;background:#1a1a1a;color:#fff;padding:4px 10px;font-size:12px;font-weight:600;font-family:'Helvetica Neue',Arial,sans-serif;">GH</span> GitHub热门项目
                        </h2>
                    </td>
                    <td style="text-align:right;">
                        <span style="display:inline-block;padding:4px 10px;background:#f0f0f0;color:#666;font-size:11px;font-family:'Helvetica Neue',Arial,sans-serif;">{len(github_news)}个项目</span>
                    </td>
                </tr>
            </table>
            {self._render_projects_list(github_news)}
        </td>
    </tr>'''

    def _render_huggingface_section(self, hf_news: List[NewsItem]) -> str:
        """渲染Hugging Face热门模型栏目"""
        return f'''<tr>
        <td style="padding:20px 40px 20px 40px;background:#fafafa;">
            <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                    <td>
                        <h2 style="margin:0 0 20px 0;font-size:14px;letter-spacing:2px;color:#888;border-bottom:1px solid #1a1a1a;padding-bottom:10px;font-family:'Helvetica Neue',Arial,sans-serif;">
                            <span style="display:inline-block;background:#5A2D82;color:#fff;padding:4px 10px;font-size:12px;font-weight:600;font-family:'Helvetica Neue',Arial,sans-serif;">HF</span> Hugging Face热门模型
                        </h2>
                    </td>
                    <td style="text-align:right;">
                        <span style="display:inline-block;padding:4px 10px;background:#e8e8e8;color:#666;font-size:11px;font-family:'Helvetica Neue',Arial,sans-serif;">{len(hf_news)}个模型</span>
                    </td>
                </tr>
            </table>
            {self._render_projects_list(hf_news)}
        </td>
    </tr>'''

    def _render_projects_list(self, projects: List[NewsItem]) -> str:
        """渲染项目列表"""
        if not projects:
            return '<p style="color:#888;font-style:italic;">暂无项目</p>'

        items = []
        for i, project in enumerate(projects, 1):
            # 获取项目描述/摘要
            desc = project.summary if project.summary else ""

            # 判断项目类型
            is_hf = project.news_type == "huggingface"
            is_gh = project.news_type == "github"

            # 提取数字徽章（下载量或star数）
            badge_value = ""
            badge_color = ""

            if is_hf and desc:
                # 解析下载量: "下载: X.XM" 或 "下载: XXX"
                import re
                download_match = re.search(r'下载:\s*([\d.]+[MK]?)', desc)
                if download_match:
                    badge_value = download_match.group(1)
                    badge_color = "#5A2D82"  # HF紫色
                    # 从描述中移除下载量部分，只保留任务等其他信息
                    desc_parts = desc.split(' | ')
                    desc_parts = [p for p in desc_parts if not p.startswith('下载:')]
                    desc = ' | '.join(desc_parts) if desc_parts else ''

            elif is_gh and desc:
                # 解析star数: "stars: XXXXX"
                import re
                star_match = re.search(r'stars:\s*(\d+)', desc)
                if star_match:
                    badge_value = star_match.group(1)
                    badge_color = "#1a1a1a"  # GH黑色
                    # 从描述中移除star数部分
                    desc_parts = desc.split(' | ')
                    desc_parts = [p for p in desc_parts if not p.startswith('stars:')]
                    desc = ' | '.join(desc_parts) if desc_parts else ''

            # 构建数字徽章
            number_badge = f'<span style="display:inline-block;background:{badge_color};color:#fff;padding:3px 8px;font-size:11px;font-weight:600;border-radius:3px;">{badge_value}</span>' if badge_value else ''

            # 根据项目类型选择标签
            if is_hf:
                source_badge = '<span style="display:inline-block;background:#5A2D82;color:#fff;padding:1px 5px;font-size:10px;font-weight:600;">HF</span>'
            else:
                source_badge = '<span style="display:inline-block;background:#1a1a1a;color:#fff;padding:1px 5px;font-size:10px;font-weight:600;">GH</span>'

            items.append(f'''
            <div style="margin-bottom:20px;padding-bottom:20px;border-bottom:1px solid #f0f0f0;position:relative;">
                <h3 style="margin:0 0 8px 0;font-size:16px;font-weight:500;line-height:1.4;padding-right:60px;">
                    <a href="{project.url}" style="color:#1a1a1a;text-decoration:none;" target="_blank">{i}. {project.title}</a>
                </h3>
                {f'<p style="margin:0 0 8px 0;font-size:13px;color:#666;line-height:1.6;">{desc[:150]}...</p>' if desc else ''}
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <p style="margin:0;font-size:12px;color:#999;font-family:'Helvetica Neue',Arial,sans-serif;">{source_badge} {project.source}</p>
                    {f'<p style="margin:0;">{number_badge}</p>' if number_badge else ''}
                </div>
            </div>''')

        return ''.join(items)

    def _render_news_list(self, news_list: List[NewsItem], start_num: int) -> str:
        """渲染新闻列表"""
        if not news_list:
            return '<p style="color:#888;font-style:italic;">暂无新闻</p>'

        items = []
        for i, news in enumerate(news_list, start_num):
            region_icon = "🇨🇳" if news.region == "domestic" else "🌍"
            link_text = "阅读全文 →"
            news_date = news.date if news.date else datetime.now().strftime('%Y年%m月%d日')

            items.append(f'''
            <div style="margin-bottom:25px;padding-bottom:25px;border-bottom:1px solid #f0f0f0;">
                <h3 style="margin:0 0 8px 0;font-size:17px;font-weight:500;line-height:1.4;">
                    <a href="{news.url}" style="color:#1a1a1a;text-decoration:none;" target="_blank">{i}. {news.title}</a>
                </h3>
                <p style="margin:0 0 10px 0;font-size:12px;color:#999;font-family:'Helvetica Neue',Arial,sans-serif;">{region_icon} {news.source} · {news_date}</p>
                <a href="{news.url}" style="color:#666;font-size:13px;text-decoration:none;border-bottom:1px solid #ddd;" target="_blank">{link_text}</a>
            </div>''')

        return ''.join(items)


class MinimalNewsTemplate(EmailTemplate):
    """
    极简风格邮件模板
    黑白灰配色，极致简洁，专注于内容
    """

    def render(
        self,
        news_list: List[NewsItem],
        ai_summary: str = "",
        ai_trends: str = "",
        date_str: str = None
    ) -> str:
        """渲染极简风格HTML邮件"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        # 分类新闻 - GitHub/HF单独显示，其他作为常规新闻
        github_news = [n for n in news_list if n.news_type == "github"]
        hf_news = [n for n in news_list if n.news_type == "huggingface"]
        # 常规新闻包括 type: "news" 和 type: "rss"
        domestic_news = [n for n in news_list if n.region == "domestic" and n.news_type in ["news", "rss"]]
        global_news = [n for n in news_list if n.region == "global" and n.news_type in ["news", "rss"]]

        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI新闻 - {date_str}</title>
</head>
<body style="margin:0;padding:0;background:#fff;color:#111;font-family:Menlo,Monaco,Consolas,'Courier New',monospace;line-height:1.6;font-size:14px;">
    <table width="100%" cellpadding="0" cellspacing="0" style="max-width:700px;margin:0 auto;padding:40px 20px;">

        <!-- 头部 -->
        <tr>
            <td style="padding-bottom:40px;border-bottom:2px solid #000;">
                <h1 style="margin:0;font-size:24px;font-weight:400;letter-spacing:-1px;">AI新闻日报</h1>
                <p style="margin:5px 0 0 0;color:#666;">{date_str}</p>
            </td>
        </tr>

        <!-- AI洞察 -->
        {(self._minimal_insight(ai_summary, ai_trends) if (ai_summary or ai_trends) else '')}

        <!-- GitHub热门项目 -->
        {(self._minimal_github_section(github_news) if github_news else '')}

        <!-- Hugging Face热门模型 -->
        {(self._minimal_hf_section(hf_news) if hf_news else '')}

        <!-- 新闻列表 -->
        <tr>
            <td style="padding-top:40px;">
                {self._minimal_section("国内新闻", domestic_news)}
            </td>
        </tr>
        <tr>
            <td style="padding-top:40px;">
                {self._minimal_section("国际新闻", global_news)}
            </td>
        </tr>

        <!-- 页脚 -->
        <tr>
            <td style="padding-top:60px;padding-bottom:20px;border-top:1px solid #eee;color:#999;font-size:12px;">
                <p style="margin:0;">Generated by AI News Automation · {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </td>
        </tr>

    </table>
</body>
</html>'''

    def _minimal_insight(self, summary: str, trends: str) -> str:
        """极简风格洞察"""
        parts = []
        if summary:
            parts.append(f'<tr><td style="padding:30px 0;border-bottom:1px solid #eee;"><strong style="background:#000;color:#fff;padding:2px 6px;">摘要</strong><p style="margin:15px 0 0 0;white-space:pre-line;">{summary}</p></td></tr>')
        if trends:
            parts.append(f'<tr><td style="padding:30px 0;border-bottom:1px solid #eee;"><strong style="background:#000;color:#fff;padding:2px 6px;">趋势</strong><p style="margin:15px 0 0 0;white-space:pre-line;">{trends}</p></td></tr>')
        return ''.join(parts)

    def _minimal_github_section(self, github_news: List[NewsItem]) -> str:
        """极简风格GitHub栏目"""
        return f'''<tr><td style="padding:30px 0;border-bottom:1px solid #eee;">
            <strong style="background:#000;color:#fff;padding:4px 8px;">GH</strong> GitHub热门项目
            {self._minimal_projects_list(github_news)}
        </td></tr>'''

    def _minimal_hf_section(self, hf_news: List[NewsItem]) -> str:
        """极简风格Hugging Face栏目"""
        return f'''<tr><td style="padding:30px 0;border-bottom:1px solid #eee;">
            <strong style="background:#5A2D82;color:#fff;padding:4px 8px;">HF</strong> Hugging Face热门模型
            {self._minimal_projects_list(hf_news)}
        </td></tr>'''

    def _minimal_projects_list(self, projects: List[NewsItem]) -> str:
        """极简风格项目列表"""
        items = []
        for n in projects:
            desc = n.summary if n.summary else ""

            # 判断项目类型
            is_hf = n.news_type == "huggingface"
            is_gh = n.news_type == "github"

            # 提取数字徽章
            number_badge = ""

            if is_hf and desc:
                import re
                download_match = re.search(r'下载:\s*([\d.]+[MK]?)', desc)
                if download_match:
                    dl_value = download_match.group(1)
                    number_badge = f' <span style="background:#5A2D82;color:#fff;padding:1px 4px;font-size:10px;">{dl_value}</span>'
                    # 从描述中移除下载量部分
                    desc_parts = desc.split(' | ')
                    desc_parts = [p for p in desc_parts if not p.startswith('下载:')]
                    desc = ' | '.join(desc_parts) if desc_parts else ''

            elif is_gh and desc:
                import re
                star_match = re.search(r'stars:\s*(\d+)', desc)
                if star_match:
                    star_value = star_match.group(1)
                    number_badge = f' <span style="background:#000;color:#fff;padding:1px 4px;font-size:10px;">{star_value}</span>'
                    # 从描述中移除star数部分
                    desc_parts = desc.split(' | ')
                    desc_parts = [p for p in desc_parts if not p.startswith('stars:')]
                    desc = ' | '.join(desc_parts) if desc_parts else ''

            items.append(f'''<li style="margin:10px 0;">
                <a href="{n.url}" style="color:#111;text-decoration:none;" target="_blank">
                    <strong>{n.title}</strong>
                </a>
                {f'<p style="margin:3px 0 0 0;color:#666;font-size:12px;">{desc[:100]}...</p>' if desc else ''}
                <p style="margin:3px 0 0 0;color:#999;font-size:11px;">{n.source}{number_badge}</p>
            </li>''')

        return f'<ul style="margin:15px 0 0 0;padding:0;list-style:none;">{"".join(items)}</ul>'

    def _minimal_section(self, title: str, news_list: List[NewsItem]) -> str:
        """极简风格章节"""
        items = []
        for n in news_list:
            items.append(f'''
                <div style="margin:0 0 25px 0;">
                    <a href="{n.url}" style="color:#111;text-decoration:none;" target="_blank">
                        <strong>{n.title}</strong>
                    </a>
                    <p style="margin:5px 0 0 0;color:#666;font-size:12px;">{n.source} · {n.date or ''}</p>
                </div>''')

        return f'''<strong style="font-size:12px;letter-spacing:2px;color:#999;text-transform:uppercase;display:block;margin-bottom:20px;">{title}</strong>
                {"".join(items)}'''


class ModernNewsTemplate(EmailTemplate):
    """保留原有的现代化模板（向后兼容）"""

    def render(
        self,
        news_list: List[NewsItem],
        ai_summary: str = "",
        ai_trends: str = "",
        date_str: str = None
    ) -> str:
        """渲染现代化HTML邮件"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        # 分类新闻 - GitHub/HF单独显示，其他作为常规新闻
        github_news = [n for n in news_list if n.news_type == "github"]
        hf_news = [n for n in news_list if n.news_type == "huggingface"]
        # 常规新闻包括 type: "news" 和 type: "rss"
        domestic_news = [n for n in news_list if n.region == "domestic" and n.news_type in ["news", "rss"]]
        global_news = [n for n in news_list if n.region == "global" and n.news_type in ["news", "rss"]]

        stats = {
            "total": len(news_list),
            "domestic": len(domestic_news),
            "global": len(global_news),
            "github": len(github_news),
            "huggingface": len(hf_news)
        }

        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>AI新闻日报 - {date_str}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif; line-height: 1.6; color: #333; background: linear-gradient(135deg, #f5f7fa 0%, #e8eaf6 100%); padding: 20px; -webkit-font-smoothing: antialiased; }}
        .container {{ max-width: 800px; margin: 0 auto; background: #ffffff; border-radius: 16px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 30px; text-align: center; position: relative; overflow: hidden; }}
        .header::before {{ content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%); animation: rotate 20s linear infinite; }}
        @keyframes rotate {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
        .header-content {{ position: relative; z-index: 1; }}
        .header h1 {{ font-size: 32px; font-weight: 700; margin-bottom: 10px; letter-spacing: 1px; }}
        .header p {{ font-size: 16px; opacity: 0.95; }}
        .header-badge {{ display: inline-block; background: rgba(255, 255, 255, 0.2); padding: 5px 15px; border-radius: 20px; font-size: 14px; margin-top: 15px; backdrop-filter: blur(10px); }}
        .ai-insights {{ background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); padding: 30px; margin: 0; }}
        .insight-card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); }}
        .insight-card h3 {{ color: {self.colors['accent']}; font-size: 18px; font-weight: 600; margin-bottom: 12px; }}
        .insight-card p {{ color: #333; font-size: 15px; line-height: 1.8; white-space: pre-line; }}
        .stats {{ display: flex; justify-content: space-around; padding: 25px 30px; background: #fafafa; }}
        .stat-item {{ text-align: center; }}
        .stat-number {{ font-size: 36px; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .news-section {{ padding: 30px; }}
        .section-header {{ display: flex; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #667eea; }}
        .news-item {{ padding: 20px 0; border-bottom: 1px solid #eee; }}
        .news-title {{ font-size: 17px; font-weight: 600; margin-bottom: 10px; }}
        .news-title a {{ color: #2c3e50; text-decoration: none; }}
        .news-meta {{ font-size: 13px; color: #666; margin-bottom: 10px; }}
        .footer {{ background: #2c3e50; color: white; padding: 25px 30px; text-align: center; }}
        .footer p {{ font-size: 13px; opacity: 0.8; margin-bottom: 8px; }}
        @media screen and (max-width: 600px) {{ .stats {{ flex-direction: column; gap: 15px; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>AI新闻日报</h1>
                <p>{date_str} | 全方位覆盖AI领域最新动态</p>
                <div class="header-badge">✨ 启用本地AI</div>
            </div>
        </div>

        <div class="ai-insights">
            {self._render_modern_insight_cards(ai_summary, ai_trends)}
        </div>

        <div class="stats">
            <div class="stat-item"><div class="stat-number">{stats['total']}</div><div style="color:#666;margin-top:5px;">总新闻</div></div>
            <div class="stat-item"><div class="stat-number">{stats['domestic']}</div><div style="color:#666;margin-top:5px;">国内</div></div>
            <div class="stat-item"><div class="stat-number">{stats['global']}</div><div style="color:#666;margin-top:5px;">国际</div></div>
        </div>

        {self._render_modern_github_section(github_news) if github_news else ''}

        {self._render_modern_hf_section(hf_news) if hf_news else ''}

        <div class="news-section">
            <div class="section-header"><h2 style="color:#667eea;">国内AI新闻</h2><span style="background:#667eea;color:#fff;padding:4px 12px;border-radius:12px;font-size:13px;margin-left:10px;">{len(domestic_news)}篇</span></div>
            {self._render_modern_news_list(domestic_news, 1)}
        </div>

        <div class="news-section">
            <div class="section-header"><h2 style="color:#667eea;">国际AI新闻</h2><span style="background:#667eea;color:#fff;padding:4px 12px;border-radius:12px;font-size:13px;margin-left:10px;">{len(global_news)}篇</span></div>
            {self._render_modern_news_list(global_news, len(domestic_news) + 1)}
        </div>

        <div class="footer">
            <p>本邮件由AI新闻自动化系统生成</p>
            <p>更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>'''

    def _render_modern_insight_cards(self, summary: str, trends: str) -> str:
        """渲染现代风格洞察卡片"""
        parts = []
        if summary:
            parts.append(f'<div class="insight-card"><h3>今日摘要</h3><p>{summary}</p></div>')
        if trends:
            parts.append(f'<div class="insight-card"><h3>趋势分析</h3><p>{trends}</p></div>')
        return ''.join(parts)

    def _render_modern_news_list(self, news_list: List[NewsItem], start_num: int) -> str:
        """渲染现代风格新闻列表"""
        if not news_list:
            return '<div style="text-align:center;padding:40px;color:#666;">暂无新闻</div>'

        items = []
        for i, news in enumerate(news_list, start_num):
            icon = "🇨🇳" if news.region == "domestic" else "🌍"
            items.append(f'''
            <div class="news-item">
                <div class="news-title"><a href="{news.url}" target="_blank">{i}. {news.title}</a></div>
                <div class="news-meta">{icon} 来源：{news.source} · 📅 {news.date or date_str}</div>
            </div>''')
        return ''.join(items)

    def _render_modern_github_section(self, github_news: List[NewsItem]) -> str:
        """渲染现代风格GitHub栏目"""
        items = []
        for i, news in enumerate(github_news, 1):
            desc = news.summary if news.summary else ""

            # 解析star数
            star_badge = ""
            clean_desc = desc
            if desc:
                import re
                star_match = re.search(r'stars:\s*(\d+)', desc)
                if star_match:
                    star_value = star_match.group(1)
                    star_badge = f'<span style="background:#333;color:#fff;padding:4px 10px;font-size:12px;font-weight:600;border-radius:8px;">{star_value}</span>'
                    # 从描述中移除star数部分
                    desc_parts = desc.split(' | ')
                    desc_parts = [p for p in desc_parts if not p.startswith('stars:')]
                    clean_desc = ' | '.join(desc_parts) if desc_parts else ''

            items.append(f'''
            <div class="news-item">
                <div class="news-title"><a href="{news.url}" target="_blank">{i}. {news.title}</a></div>
                {f'<div style="font-size:14px;color:#666;margin:8px 0;">{clean_desc[:120]}...</div>' if clean_desc else ''}
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;">
                    <div class="news-meta"><span style="background:#333;color:#fff;padding:2px 6px;font-size:11px;">GH</span> {news.source}</div>
                    {f'<div>{star_badge}</div>' if star_badge else ''}
                </div>
            </div>''')

        return f'''<div class="news-section" style="background:#f8f9fa;">
            <div class="section-header"><h2 style="color:#333;"><span style="background:#333;color:#fff;padding:4px 10px;font-size:14px;font-weight:600;margin-right:10px;">GH</span>GitHub热门项目</h2><span style="background:#333;color:#fff;padding:4px 12px;border-radius:12px;font-size:13px;margin-left:10px;">{len(github_news)}个项目</span></div>
            {''.join(items)}
        </div>'''

    def _render_modern_hf_section(self, hf_news: List[NewsItem]) -> str:
        """渲染现代风格Hugging Face栏目"""
        items = []
        for i, news in enumerate(hf_news, 1):
            desc = news.summary if news.summary else ""

            # 解析下载量
            download_badge = ""
            clean_desc = desc
            if desc:
                import re
                download_match = re.search(r'下载:\s*([\d.]+[MK]?)', desc)
                if download_match:
                    dl_value = download_match.group(1)
                    download_badge = f'<span style="background:#856404;color:#fff;padding:4px 10px;font-size:12px;font-weight:600;border-radius:8px;">{dl_value}</span>'
                    # 从描述中移除下载量部分
                    desc_parts = desc.split(' | ')
                    desc_parts = [p for p in desc_parts if not p.startswith('下载:')]
                    clean_desc = ' | '.join(desc_parts) if desc_parts else ''

            items.append(f'''
            <div class="news-item">
                <div class="news-title"><a href="{news.url}" target="_blank">{i}. {news.title}</a></div>
                {f'<div style="font-size:14px;color:#666;margin:8px 0;">{clean_desc[:120]}...</div>' if clean_desc else ''}
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;">
                    <div class="news-meta"><span style="background:#856404;color:#fff;padding:2px 6px;font-size:11px;">HF</span> {news.source}</div>
                    {f'<div>{download_badge}</div>' if download_badge else ''}
                </div>
            </div>''')

        return f'''<div class="news-section" style="background:#fff3cd;">
            <div class="section-header"><h2 style="color:#856404;"><span style="background:#856404;color:#fff;padding:4px 10px;font-size:14px;font-weight:600;margin-right:10px;">HF</span>Hugging Face热门模型</h2><span style="background:#856404;color:#fff;padding:4px 12px;border-radius:12px;font-size:13px;margin-left:10px;">{len(hf_news)}个模型</span></div>
            {''.join(items)}
        </div>'''


class TextEmailTemplate(EmailTemplate):
    """纯文本邮件模板"""

    def render(
        self,
        news_list: List[NewsItem],
        ai_summary: str = "",
        ai_trends: str = "",
        date_str: str = None
    ) -> str:
        """渲染纯文本邮件"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        # 分类新闻
        github_news = [n for n in news_list if n.news_type == "github"]
        hf_news = [n for n in news_list if n.news_type == "huggingface"]
        # 常规新闻包括 type: "news" 和 type: "rss"
        domestic_news = [n for n in news_list if n.region == "domestic" and n.news_type in ["news", "rss"]]
        global_news = [n for n in news_list if n.region == "global" and n.news_type in ["news", "rss"]]

        lines = [
            "=" * 50,
            "AI新闻日报",
            "=" * 50,
            f"{date_str}",
            ""
        ]

        if ai_summary:
            lines.extend(["【AI洞察】", ai_summary, ""])

        if ai_trends:
            lines.extend(["【趋势分析】", ai_trends, ""])

        # GitHub热门项目
        if github_news:
            lines.extend(["", "[GH] 【GitHub热门项目】", ""])
            for i, news in enumerate(github_news, 1):
                # 解析star数和描述
                star_str = ""
                desc_str = ""
                if news.summary:
                    import re
                    star_match = re.search(r'stars:\s*(\d+)', news.summary)
                    if star_match:
                        star_str = f" | {star_match.group(1)}★"
                    # 获取除star数外的其他描述
                    desc_parts = news.summary.split(' | ')
                    desc_parts = [p for p in desc_parts if not p.startswith('stars:')]
                    if desc_parts:
                        desc_str = f"\n   描述: {' | '.join(desc_parts)[:80]}..."

                lines.extend([f"{i}. {news.title}", f"   [GH] {news.source}{star_str}{desc_str}", f"   链接：{news.url}", ""])

        # Hugging Face热门模型
        if hf_news:
            lines.extend(["", "[HF] 【Hugging Face热门模型】", ""])
            for i, news in enumerate(hf_news, 1):
                # 解析下载量和描述
                download_str = ""
                desc_str = ""
                if news.summary:
                    import re
                    download_match = re.search(r'下载:\s*([\d.]+[MK]?)', news.summary)
                    if download_match:
                        download_str = f" | 下载: {download_match.group(1)}"
                    # 获取除下载量外的其他描述
                    desc_parts = news.summary.split(' | ')
                    desc_parts = [p for p in desc_parts if not p.startswith('下载:')]
                    if desc_parts:
                        desc_str = f"\n   描述: {' | '.join(desc_parts)[:80]}..."

                lines.extend([f"{i}. {news.title}", f"   [HF] {news.source}{download_str}{desc_str}", f"   链接：{news.url}", ""])

        # 国内新闻
        if domestic_news:
            lines.extend(["", "【国内AI新闻】", ""])
            for i, news in enumerate(domestic_news, 1):
                lines.extend([f"{i}. {news.title}", f"   来源：{news.source}", f"   链接：{news.url}", ""])

        # 国际新闻
        if global_news:
            lines.extend(["", "【国际AI新闻】", ""])
            for i, news in enumerate(global_news, 1):
                lines.extend([f"{i}. {news.title}", f"   来源：{news.source}", f"   链接：{news.url}", ""])

        lines.extend(["", "-" * 50, f"Generated by AI News Automation · {datetime.now().strftime('%Y-%m-%d')}", "=" * 50])

        return "\n".join(lines)
