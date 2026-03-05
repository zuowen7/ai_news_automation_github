# AI新闻自动化系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production--ready)
![Version](https://img.shields.io/badge/Version-2.3.0-blue)

每天定时从全球新闻源抓取AI新闻，使用本地AI模型进行分类、总结，并通过邮件发送。

[功能特点](#功能特点) • [快速开始](#快速开始) • [配置说明](#配置说明) • [更新日志](#更新日志)

</div>

## ✨ 功能特点

### 核心功能
- **🌍 多样化新闻源** - 50+个有效源，覆盖8+个地区，避免信息茧房
- **🤖 本地AI处理** - 使用Ollama，保护隐私，零API费用
- **🎯 智能筛选** - AI相关新闻精准识别，自动去重
- **📊 GitHub/HF集成** - 自动获取热门项目和模型
- **🌐 双模型处理** - 国内用Qwen，国际用Llama
- **📧 美观邮件** - 现代化HTML模板，响应式设计
- **⚡ 高性能** - 3分钟完成抓取和处理，支持动态并发

### 🚀 最新优化（v2.3.0）
- **🌍 多样化新闻源** - 新增40+个源，覆盖欧洲、亚太、学术、开源社区
- **⚡ 性能优化** - 运行时间优化到3分钟，成功率提升到70%+
- **🔧 智能重试** - 指数退避算法，自动处理请求失败
- **📈 动态并发** - 根据成功率自动调整线程数（8-10个）
- **💾 智能缓存** - 新闻历史缓存 + AI结果缓存
- **🏥 健康监控** - 新闻源可用性监控和质量评分

## 核心功能

### 新闻源覆盖（50+个有效源）
**🌏 国内源（3个）**：雷锋网AI、新浪科技AI、IT之家AI

**🇺🇸 北美源（20+个）**：
- 媒体：TechCrunch AI、VentureBeat、ArsTechnica、Wired、MIT Tech Review
- 公司博客：Google DeepMind、Microsoft AI、Microsoft Research、NVIDIA、AWS AI Blog
- 社区：Towards Data Science、Machine Learning Mastery、Papers with Code

**🇪🇺 欧洲源（5+个）**：
- AI Business（英国）、The Guardian Tech（英国）
- ScienceDaily（国际学术）、Tech.eu（欧洲）
- Silicon Republic（爱尔兰）

**🌏 亚太源（5+个）**：
- Tech in Asia（亚洲）、Nikkei Asia（日本）
- Analytics India Mag、YourStory AI、Inc42 AI（印度）

**🎓 学术/研究（10+个）**：
- 顶级机构：Stanford AI Blog、MIT CSAIL、Allen Institute for AI
- 会议/出版：AAAI、ACM AI、IEEE AI

**💼 产业/投资（5+个）**：
- VC视角：Sequoia AI、a16z AI
- 咨询：McKinsey AI、BCG AI、Gartner AI

**👥 开源社区（5+个）**：
- Dev.to AI、Medium AI Tag、DataReport AI、KDnuggets

**🔧 开发工具**：
- GitHub Trending：5个热门AI项目
- Hugging Face：5个热门模型

### 新闻源分类统计
- 媒体类：30%
- 公司博客：25%
- 学术机构：15%
- 开源社区：15%
- 产业投资：10%
- 社区论坛：5%

### AI处理流程
1. **智能筛选**: 关键词预筛选 + AI精准筛选
2. **分别处理**:
   - 国内新闻: qwen2.5:7b-instruct处理，保留前5名
   - 国际新闻: llama3.1:8b处理，保留前10名
3. **摘要生成**: 国内外分别生成摘要
4. **趋势分析**: 国内外分别分析趋势
5. **综合汇总**: qwen2.5:7b-instruct生成最终概要和趋势

### 输出内容
- GitHub热门项目（5个）
- Hugging Face热门模型（5个）
- 国内AI新闻（5篇）
- 国际AI新闻（10篇）
- AI综合摘要
- AI趋势分析

## 项目结构

```
ai_news_refactored/
├── src/
│   ├── __init__.py
│   ├── main.py                 # 主程序入口
│   ├── config/                 # 配置模块
│   │   ├── __init__.py
│   │   ├── constants.py        # 常量定义、新闻源配置
│   │   └── settings.py         # 配置管理
│   ├── fetchers/               # 新闻抓取模块
│   │   ├── __init__.py
│   │   ├── base.py             # 基础抓取器、NewsItem类
│   │   ├── html_fetcher.py     # HTML网页抓取器
│   │   ├── rss_fetcher.py      # RSS订阅抓取器
│   │   ├── github_fetcher.py   # GitHub Trending抓取器
│   │   ├── huggingface_fetcher.py # Hugging Face抓取器
│   │   └── manager.py          # 抓取管理器（并发、去重）
│   ├── ai/                     # AI处理模块
│   │   ├── __init__.py
│   │   ├── ollama_client.py    # Ollama客户端（支持多模型）
│   │   ├── processor.py        # AI处理器（摘要、趋势）
│   │   └── filter.py           # AI筛选器（关键词+AI筛选）
│   ├── email/                  # 邮件模块
│   │   ├── __init__.py
│   │   ├── templates.py        # 邮件模板（4种风格）
│   │   └── sender.py           # 邮件发送器
│   └── utils/                  # 工具模块
│       ├── __init__.py
│       ├── logger.py           # 日志系统
│       └── helpers.py          # 辅助函数
├── tests/                      # 测试模块
├── logs/                       # 日志目录
├── output/                     # 输出目录
├── config.json                 # 配置文件
├── requirements.txt            # 依赖列表
├── run.py                      # 快速启动脚本
└── README.md                   # 项目说明
```

## 环境要求

- Python 3.8+
- Ollama (本地AI服务)
- qwen2.5:7b-instruct 模型
- llama3.1:8b 模型（用于国际新闻处理）

## 快速开始

### 1. 安装Ollama

从 [ollama.com](https://ollama.com) 下载并安装Ollama。

### 2. 拉取AI模型

```bash
# 国内新闻处理模型（必需）
ollama pull qwen2.5:7b-instruct

# 国际新闻处理模型（必需）
ollama pull llama3.1:8b
```

### 3. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 4. 配置邮箱

编辑 `config.json`，填入邮箱信息：

```json
{
  "email": {
    "sender_email": "your_email@qq.com",
    "sender_password": "",  // 留空，通过环境变量设置
    "smtp_server": "smtp.qq.com",
    "smtp_port": 465,
    "recipient_email": "recipient@foxmail.com"
  }
}
```

### 5. 设置邮箱密码（推荐方式）

设置环境变量 `AI_NEWS_EMAIL_PASSWORD` 为QQ邮箱授权码：

```bash
# Windows (PowerShell)
$env:AI_NEWS_EMAIL_PASSWORD="your_qq_auth_code"
[Environment]::SetEnvironmentVariable("AI_NEWS_EMAIL_PASSWORD", $env:AI_NEWS_EMAIL_PASSWORD, "User")
```

### 6. 运行程序

#### 方式1：使用批处理文件（推荐）⭐

直接双击 `run_ai_news.bat` 文件即可运行！

**优点：**
- ✅ 自动切换到正确的项目目录
- ✅ 自动加载配置文件
- ✅ 显示运行状态和当前目录
- ✅ 运行完成后暂停，方便查看输出

**文件位置：**
```
ai_news_automation_github/
└── run_ai_news.bat          # 一键启动脚本
```

#### 方式2：使用命令行

```bash
python run.py
```

#### 方式3：指定配置文件

如果配置文件不在默认位置，可以使用：

```bash
python run.py --config "D:\path\to\config.json"
```

## 配置选项

### AI配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `ai.enabled` | 是否启用AI处理 | true |
| `ai.model_name` | 默认AI模型 | qwen2.5:7b-instruct |
| `ai.temperature` | AI温度参数 | 0.7 |
| `ai.timeout` | AI请求超时(秒) | 60 |
| `ai.enable_filter` | 是否启用AI筛选 | true |

### 抓取器配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `fetcher.concurrent_requests` | 并发请求数 | 3 |
| `fetcher.enable_github` | 是否抓取GitHub | true |
| `fetcher.enable_huggingface` | 是否抓取HuggingFace | true |

### 邮件配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `settings.qq_mail_format` | 使用QQ邮箱兼容格式 | true |
| `settings.html_email` | 发送HTML邮件 | true |

## AI模型使用说明

### 模型分工

| 任务 | 使用模型 | 说明 |
|------|---------|------|
| 国内新闻筛选 | qwen2.5:7b-instruct | 保留前5篇 |
| 国际新闻筛选 | qwen2.5:7b-instruct | 保留前10篇 |
| 国内新闻摘要 | qwen2.5:7b-instruct | 100字以内 |
| 国际新闻摘要 | llama3.1:8b | 100字以内 |
| 国内趋势分析 | qwen2.5:7b-instruct | 150字以内 |
| 国际趋势分析 | llama3.1:8b | 200字以内 |
| 综合汇总 | qwen2.5:7b-instruct | 合并国内外内容 |

### 自定义模型

如需使用其他模型，修改 `src/ai/processor.py` 中的模型名称：

```python
# 国内新闻使用其他模型
domestic_summary = self.generate_summary_for_region(news_list, "domestic", "your_model_name")
```

## 命令行参数

```bash
python run.py --help
```

| 参数 | 说明 |
|------|------|
| `--config` | 指定配置文件路径 |
| `--no-email` | 不发送邮件，仅保存文件 |
| `--test-ai` | 测试AI功能 |
| `--log-level` | 日志级别 (DEBUG/INFO/WARNING/ERROR) |

## 输出说明

### 文件输出

运行后会在 `output/` 目录生成：

- `ai_news_YYYYMMDD.json` - JSON格式新闻数据
- `ai_news_YYYYMMDD.html` - HTML格式邮件预览

### 邮件内容

1. **GitHub热门项目** (5个)
   - 项目名称和描述
   - Star数量（右下角黑色徽章）
   - 项目链接

2. **Hugging Face热门模型** (5个)
   - 模型名称
   - 下载量（右下角紫色徽章）
   - 任务类型和点赞数
   - 模型链接

3. **国内AI新闻** (5篇)
   - 按AI评分排序
   - 包含标题、来源、日期
   - 新闻链接

4. **国际AI新闻** (10篇)
   - 按AI评分排序
   - 包含标题、来源、日期
   - 新闻链接

5. **AI洞察**
   - 今日摘要（国内外综合）
   - 趋势分析（国内外综合）

## 定时任务设置

### Windows任务计划程序

1. 打开"任务计划程序"
2. 创建基本任务
3. 设置触发器（每天指定时间）
4. 操作：启动程序 `python run.py`
5. 起始于：项目根目录路径

### 使用批处理文件

创建 `daily_task.bat`：

```batch
@echo off
cd /d D:\ai_news_automation1\ai_news_refactored
python run.py
```

## 故障排除

### 问题1: 国内新闻不显示

**原因**: RSS新闻类型为"rss"，需要包含在过滤条件中

**解决**: 确保 `src/ai/processor.py` 中的过滤条件为：
```python
real_news = [n for n in news_list if n.news_type in ["news", "rss"]]
```

### 问题2: 摘要和趋势为空

**原因**: AI模型未正确响应或模型未加载

**解决**:
1. 检查Ollama是否运行: `curl http://localhost:11434/api/tags`
2. 确认模型已下载: `ollama list`
3. 查看日志中的AI处理错误

### 问题3: 邮件显示"暂无新闻"

**原因**: 新闻筛选条件过于严格

**解决**:
1. 检查日志中抓取到的新闻数量
2. 降低关键词筛选阈值（`filter.py`中的`min_score`）
3. 检查新闻源是否可访问

### 问题4: Hugging Face下载量显示0

**原因**: 字段访问错误

**解决**: 确保 `src/fetchers/huggingface_fetcher.py` 第65行为：
```python
downloads = model_data.get('downloads', 0)
```

## 自定义配置

### 添加新闻源

编辑 `src/config/constants.py` 中的 `DEFAULT_SOURCES`：

```python
DEFAULT_SOURCES = {
    "domestic": [
        {"name": "新闻源名称", "url": "https://...", "enabled": True}
    ],
    "global": [
        {"name": "News Source", "url": "https://...", "enabled": True}
    ]
}
```

### 调整新闻数量

编辑 `src/fetchers/manager.py`：

```python
# GitHub项目数量
GitHubTrendingFetcher(token=github_token, max_news=5)

# Hugging Face模型数量
HuggingFaceFetcher(max_news=5)
```

### 自定义邮件模板

修改 `src/email/templates.py` 中的模板类。

## 更新日志

### v2.3.0 (2026-03-04)
- ✨ **智能去重系统**：URL指纹识别 + 标题相似度检测，自动移除重复内容
- ✨ **质量评分系统**：6维度评分（标题质量、源质量、AI相关性等），智能排序
- 🐛 **修复中文编码**：改进编码检测逻辑，正确处理国内网站UTF-8/GBK编码
- 🎯 **优化AI筛选**：增加输入输出数量，改进提示词，保留更多高质量新闻
- ⚙️ **配置改进**：禁用增量抓取（改为智能去重），新增去重和评分配置项
- 📊 **效果提升**：新闻数量从20条提升到31条（+55%），国际新闻从10条提升到20条（+100%）

### v2.2.0 (2026-02-13)
- 🌏 新增国内AI官方渠道：智谱AI、MiniMax、百度文心、阿里通义、DeepSeek
- 📰 新增国内科技媒体：IT之家AI、中关村在线AI、至顶网、雷锋网AI
- 🤖 扩展AI关键词库，新增GLM-4/5、MiniMax-2.5、DeepSeek V3、Qwen 2.5等国内大模型
- 📉 降低新闻预筛选门槛（1.5→1.0），提高国内新闻覆盖率
- 📊 增加国内新闻AI评估数量（15→20条），优化筛选质量
- 🔧 新增环境变量设置脚本（setup_env.bat）
- 🐛 优化新闻源筛选逻辑

### v2.1.0 (2026-02-11)
- ✨ 新增智能缓存系统（新闻历史 + AI结果）
- ✨ 新增增量抓取模式（只抓取24小时内新内容）
- ✨ 新增智能重试机制（指数退避 + 随机抖动）
- ✨ 新增User-Agent轮换（10+种浏览器UA）
- ✨ 新增动态并发调整（根据成功率自动调整）
- ✨ 新增自动缓存清理（LRU策略 + 数量限制）
- 🐛 修复部分新闻源SSL证书问题
- 📝 添加Docker和Docker Compose支持

### v2.0.0 (2026-02-10)
- ♻️ 完全重构代码结构，模块化设计
- ✨ 新增GitHub Trending集成（5个热门项目）
- ✨ 新增Hugging Face集成（5个热门模型）
- ✨ 实现双模型处理：国内用Qwen，国际用Llama3.1:8b
- ✨ 国内外新闻分别排名（国内前5，国际前10）
- 🐛 修复RSS新闻类型过滤问题
- 🐛 修复Hugging Face下载量显示为0的问题
- 📝 优化AI摘要和趋势生成流程

### v1.0.0
- 🎉 初始版本

## 🐳 Docker部署

### 使用Docker Compose（推荐）

```bash
# 1. 复制配置文件
cp config.example.json config.json
# 编辑 config.json 填入配置

# 2. 设置环境变量
export AI_NEWS_EMAIL_PASSWORD=your_password

# 3. 启动服务（包含Ollama）
docker-compose up -d

# 查看日志
docker-compose logs -f ai-news
```

### 使用Docker单独运行

```bash
# 构建镜像
docker build -t ai-news .

# 运行容器
docker run -v $(pwd)/config.json:/app/config.json \
           -v $(pwd)/logs:/app/logs \
           -v $(pwd)/cache:/app/cache \
           -v $(pwd)/output:/app/output \
           -e AI_NEWS_EMAIL_PASSWORD=your_password \
           ai-news
```

## ⭐ GitHub Actions

在GitHub上启用后，可实现每日自动抓取：

1. Fork本仓库
2. 在Settings > Secrets中添加：
   - `SENDER_EMAIL`: 发件邮箱
   - `RECIPIENT_EMAIL`: 收件邮箱
   - `EMAIL_PASSWORD`: 邮箱授权码
3. 启用Actions

查看完整配置：[`.github/workflows/scheduled.yml`](.github/workflows/scheduled.yml)

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 许可证

MIT License
