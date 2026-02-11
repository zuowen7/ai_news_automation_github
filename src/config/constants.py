"""
常量定义模块 - 优化版
扩展新闻源、优化关键词和打分机制
"""

from typing import Dict, List


# 默认新闻源配置 - 权威专业媒体（已验证可用）
DEFAULT_SOURCES = {
    "domestic": [
        # ========== 国内有效AI媒体 ==========
        {"name": "量子位AI", "url": "https://www.qbitai.com/", "enabled": True},
        {"name": "雷锋网AI", "url": "https://www.leiphone.com/category/ai", "enabled": True},
        {"name": "机器之心", "url": "https://www.jiqizhixin.com/", "enabled": True},
        {"name": "新智元", "url": "https://www.ai-era.net/", "enabled": False},  # SSL无法修复
        {"name": "AI研习社", "url": "https://www.yanxishe.com/", "enabled": False},  # SSL无法修复
        {"name": "大数据文摘", "url": "http://www.36dsj.com/", "enabled": False},  # SSL无法修复

        # ========== 科技媒体AI版块 ==========
        {"name": "36氪AI", "url": "https://36kr.com/tags/ai", "enabled": True},
        {"name": "虎嗅AI", "url": "https://www.huxiu.com/searchResult.action?searchKeywords=AI", "enabled": True},
        {"name": "InfoQ中文", "url": "https://www.infoq.cn/topic/AI", "enabled": True},
        {"name": "极客公园AI", "url": "https://www.geekpark.net/tags/AI", "enabled": True},
        {"name": "钛媒体AI", "url": "https://www.tmtpost.com/tag/1654.html", "enabled": False},
        {"name": "新浪科技AI", "url": "https://tech.sina.com.cn/", "enabled": True},
        {"name": "搜狐科技AI", "url": "https://www.sohu.com/", "enabled": True},
        {"name": "网易智能AI", "url": "https://tech.163.com/special/", "enabled": False},  # 404
        {"name": "中科院自动化所", "url": "http://www.ia.cas.cn/", "enabled": False},
        {"name": "清华AI", "url": "https://www.tsinghua.edu.cn/ai/", "enabled": False},  # 404
        {"name": "算法与数学之美", "url": "https://www.cbdio.com/", "enabled": True},  # SSL修复
    ],
    "global": [
        # ========== 顶级科技媒体RSS（已验证）==========
        {"name": "TechCrunch AI RSS", "url": "https://techcrunch.com/category/artificial-intelligence/feed/", "enabled": True},
        {"name": "VentureBeat AI RSS", "url": "https://venturebeat.com/category/ai/feed/", "enabled": True},
        {"name": "VentureBeat RSS", "url": "https://venturebeat.com/feed/", "enabled": True},
        {"name": "The Verge AI RSS", "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "enabled": True},
        {"name": "The Verge RSS", "url": "https://www.theverge.com/rss/index.xml", "enabled": True},
        {"name": "ArsTechnica AI RSS", "url": "https://feeds.arstechnica.com/arstechnica/technology-lab", "enabled": True},
        {"name": "ArsTechnica RSS", "url": "https://feeds.arstechnica.com/arstechnica/index", "enabled": True},
        {"name": "Wired RSS", "url": "https://www.wired.com/feed/rss", "enabled": True},
        {"name": "MIT Tech Review RSS", "url": "https://www.technologyreview.com/feed/", "enabled": True},
        {"name": "Engadget RSS", "url": "https://www.engadget.com/rss.xml", "enabled": True},
        {"name": "Gizmodo RSS", "url": "https://gizmodo.com/rss", "enabled": True},

        # ========== AI专业媒体RSS ==========
        {"name": "AI News RSS", "url": "https://artificialintelligence-news.com/feed/", "enabled": True},
        {"name": "Synced AI", "url": "https://syncedreview.com/feed", "enabled": False},  # 429
        {"name": "Unite.AI RSS", "url": "https://www.unite.ai/feed", "enabled": False},  # 403
        {"name": "KDnuggets RSS", "url": "https://www.kdnuggets.com/feed", "enabled": False},  # 429
        {"name": "AI Trends RSS", "url": "https://www.aitrends.com/feed/", "enabled": True},
        {"name": "InfoQ AI RSS", "url": "https://www.infoq.com/feed", "enabled": True},
        {"name": "Dark Reading RSS", "url": "https://www.darkreading.com/rss.xml", "enabled": True},

        # ========== 大公司官方博客RSS ==========
        {"name": "OpenAI Blog RSS", "url": "https://openai.com/news/rss/", "enabled": False},  # 403
        {"name": "Google AI RSS", "url": "https://blog.google/technology/ai/rss/", "enabled": True},
        {"name": "Google DeepMind RSS", "url": "https://deepmind.google/discover/blog/feed/", "enabled": True},
        {"name": "Google RSS", "url": "https://blog.google/technology/rss/", "enabled": True},
        {"name": "Microsoft AI RSS", "url": "https://blogs.microsoft.com/ai/feed/", "enabled": True},
        {"name": "Microsoft Research RSS", "url": "https://www.microsoft.com/en-us/research/blog/feed/", "enabled": True},
        {"name": "Meta AI RSS", "url": "https://ai.meta.com/blog/rss/", "enabled": False},  # 400
        {"name": "NVIDIA Blog RSS", "url": "https://blogs.nvidia.com/ai/feed/", "enabled": False},  # 404
        {"name": "NVIDIA RSS", "url": "https://blogs.nvidia.com/feed/", "enabled": True},
        {"name": "AWS AI Blog RSS", "url": "https://aws.amazon.com/blogs/machine-learning/feed/", "enabled": True},
        {"name": "IBM AI Blog RSS", "url": "https://www.ibm.com/blogs/think/ai/feed/", "enabled": True},
        {"name": "IBM Research RSS", "url": "https://research.ibm.com/feed/", "enabled": False},  # 404
        {"name": "Anthropic RSS", "url": "https://www.anthropic.com/news/rss", "enabled": False},  # 404
        {"name": "Stability AI RSS", "url": "https://stability.ai/blog/rss", "enabled": False},  # 404
        {"name": "Adept AI RSS", "url": "https://www.adept.ai/blog/rss", "enabled": False},  # 403
        {"name": "Cohere RSS", "url": "https://cohere.com/blog/rss", "enabled": True},
        {"name": "Mistral AI RSS", "url": "https://mistral.ai/news/rss", "enabled": False},  # 404
        {"name": "Hugging Face Blog RSS", "url": "https://huggingface.co/blog/feed.xml", "enabled": True},

        # ========== 学术与研究RSS ==========
        {"name": "ArXiv AI RSS", "url": "http://export.arxiv.org/rss/cs.AI", "enabled": True},
        {"name": "ArXiv ML RSS", "url": "http://export.arxiv.org/rss/cs.LG", "enabled": True},
        {"name": "ArXiv CV RSS", "url": "http://export.arxiv.org/rss/cs.CV", "enabled": True},
        {"name": "ArXiv CL RSS", "url": "http://export.arxiv.org/rss/cs.CL", "enabled": True},
        {"name": "ArXiv NE RSS", "url": "http://export.arxiv.org/rss/cs.NE", "enabled": True},
        {"name": "ArXiv Robotics RSS", "url": "http://export.arxiv.org/rss/cs.RO", "enabled": True},
        {"name": "Nature AI RSS", "url": "https://www.nature.com/subjects/artificial-intelligence/rss", "enabled": False},  # 404
        {"name": "IEEE Spectrum RSS", "url": "https://spectrum.ieee.org/rss.xml", "enabled": False},  # 404
        {"name": "Science AI RSS", "url": "https://www.science.org/rss/news_current.xml", "enabled": True},

        # ========== 专业开发媒体RSS ==========
        {"name": "Towards AI RSS", "url": "https://towardsai.net/rss", "enabled": True},
        {"name": "DataCamp RSS", "url": "https://www.datacamp.com/blog/rss", "enabled": False},  # 403
        {"name": "MLOps Community RSS", "url": "https://mlops.community/feed/", "enabled": True},
        {"name": "Semantic Scholar Blog RSS", "url": "https://blog.semanticscholar.org/feed/", "enabled": True},  # SSL修复
        {"name": "Papers with Code RSS", "url": "https://paperswithcode.com/rss", "enabled": True},
        {"name": "The Gradient RSS", "url": "https://thegradient.pub/rss/", "enabled": True},
        {"name": "Machine Learning Mastery RSS", "url": "https://machinelearningmastery.com/feed/", "enabled": True},

        # ========== 产业媒体RSS ==========
        {"name": "ZDNet AI RSS", "url": "https://www.zdnet.com/topic/artificial-intelligence/rss/", "enabled": False},  # 404
        {"name": "InfoWorld AI RSS", "url": "https://www.infoworld.com/category/artificial-intelligence/index.rss", "enabled": True},
        {"name": "Computerworld AI RSS", "url": "https://www.computerworld.com/category/artificial-intelligence/index.rss", "enabled": True},
        {"name": "CIO AI RSS", "url": "https://www.cio.com/feed/", "enabled": True},
        {"name": "ReadWrite AI RSS", "url": "https://readwrite.com/feed/", "enabled": True},
        {"name": "The Register RSS", "url": "https://www.theregister.com/headlines.atom", "enabled": True},
        {"name": "TechTarget AI RSS", "url": "https://www.techtarget.com/rss/whatis.xml", "enabled": False},  # 404
        {"name": "Network World RSS", "url": "https://www.networkworld.com/newsrss/tech-news.jsp", "enabled": True},
    ]
}

# AI关键词 - 2025年最新全面优化
# 按类别和重要性分组，便于扩展和维护
AI_KEYWORDS = {
    # ========== 核心AI术语（高权重）==========
    "core": [
        'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning', 'dl',
        'neural network', 'transformer', 'attention mechanism', 'diffusion model',
        'foundation model', 'large language model', 'llm', 'generative ai', 'aigc',
        'multimodal', 'computer vision', 'cv', 'nlp', 'natural language processing',
        'reinforcement learning', 'rl', 'supervised learning', 'unsupervised learning',
    ],

    # ========== 大语言模型（高权重）==========
    "llm": [
        'gpt', 'chatgpt', 'gpt-4', 'gpt-4o', 'gpt-5', 'o1', 'o3',
        'claude', 'claude 4', 'claude opus', 'claude sonnet',
        'gemini', 'gemini 2.0', 'gemini ultra', 'gemini pro',
        'llama', 'llama 2', 'llama 3', 'llama 4', 'mistral', 'mixtral',
        'qwen', 'qwen 2', 'yi', 'baichuan', 'internlm', 'deepseek', 'grok',
        'mctr', 'dbrx', 'olmo', 'phi', 'vicuna', 'alpaca', 'wizardlm',
        'chatglm', 'chatglm 2', 'chatglm 3', 'chatglm 4',
        'prompt', 'prompt engineering', 'prompt tuning', 'in-context learning',
        'few-shot', 'zero-shot', 'one-shot', 'chain of thought', 'cot',
        'instruction tuning', 'rlhf', 'sft', 'ppo', 'dpo',
        'fine-tuning', 'finetuning', 'peft', 'lora', 'qlora', 'adapter',
    ],

    # ========== AI框架与工具 ==========
    "frameworks": [
        'pytorch', 'torch', 'tensorflow', 'tf', 'keras', 'jax', 'flax',
        'hugging face', 'transformers', 'diffusers', 'accelerate',
        'langchain', 'langgraph', 'llamaindex', 'haystack', 'semantic kernel',
        'openai api', 'anthropic api', 'cohere api', 'huggingface api',
        'vector database', 'pinecone', 'weaviate', 'milvus', 'chromadb', 'faiss',
        'redis', 'pgvector', 'qdrant',
        'gradio', 'streamlit', 'dash', 'panel',
    ],

    # ========== RAG与检索增强 ==========
    "rag": [
        'rag', 'retrieval augmented generation', 'retrieval',
        'embedding', 'vector embedding', 'text embedding', 'sentence embedding',
        'semantic search', 'hybrid search', 'dense retrieval', 'sparse retrieval',
        'chunking', 'document splitting', 'reranking', 'cross-encoder',
        'knowledge base', 'knowledge graph', 'rag pipeline',
    ],

    # ========== 多模态AI ==========
    "multimodal": [
        'text-to-image', 'image generation', 'stable diffusion', 'midjourney',
        'dall-e', 'dall-e 2', 'dall-e 3', 'sora', 'runway',
        'text-to-video', 'video generation', 'video diffusion',
        'image-to-video', 'video-to-video',
        'voice generation', 'tts', 'text-to-speech', 'speech synthesis',
        'asr', 'speech recognition', 'whisper',
        'object detection', 'segmentation', 'instance segmentation',
        'semantic segmentation', 'panoptic segmentation',
        'face recognition', 'face detection', 'pose estimation',
        'ocr', 'optical character recognition', 'document ai',
        'image classification', 'visual question answering', 'vqa',
        'image captioning', 'image understanding',
    ],

    # ========== AI应用领域 ==========
    "applications": [
        'ai agent', 'agent', 'autonomous agent', 'multi-agent', 'agentic',
        'autonomous system', 'autonomous driving', 'self-driving',
        'robotics', 'robot', 'robotic', 'manipulation', 'navigation',
        'recommendation system', 'recommender', 'personalization',
        'fraud detection', 'anomaly detection', 'outlier detection',
        'predictive analytics', 'predictive modeling',
        'medical ai', 'healthcare ai', 'clinical ai', 'drug discovery',
        'protein folding', 'alpha fold', 'biotech ai',
        'financial ai', 'trading ai', 'risk assessment', 'credit scoring',
        'legal ai', 'legaltech', 'contract analysis',
        'education ai', 'edtech', 'personalized learning',
        'code generation', 'code assistant', 'copilot', 'code interpreter',
        'ai assistant', 'ai chatbot', 'conversational ai',
    ],

    # ========== AI基础设施 ==========
    "infrastructure": [
        'gpu', 'tpu', 'npu', 'ai accelerator', 'h100', 'a100', 'h200', 'b200',
        'rtx', 'cuda', 'rocm', 'oneapi',
        'cloud ai', 'aws ai', 'azure ai', 'google cloud ai', 'gcp ai',
        'model deployment', 'model serving', 'inference engine', 'triton',
        'onnx', 'tensorrt', 'openvino', 'coreml', 'tflite',
        'quantization', 'pruning', 'distillation', 'model compression',
        'edge ai', 'on-device ai', 'mobile ai', 'embedded ai', 'iot ai',
        'mlops', 'mllOps', 'modelops', 'dataops', 'aiml',
        'feature store', 'model registry', 'experiment tracking',
        'mlflow', 'wandb', 'weights & biases', 'comet', 'neptune',
    ],

    # ========== AI公司/产品 ==========
    "companies": [
        'openai', 'anthropic', 'google deepmind', 'google gemini', 'gemini',
        'microsoft copilot', 'microsoft bing', 'microsoft azure',
        'meta ai', 'nvidia', 'amd ai', 'intel ai',
        'stability ai', 'midjourney', 'runway', 'pika', 'heygen', 'synthesia',
        'character.ai', 'perplexity', 'you.com', 'andrei karpathy',
        'hugging face', 'huggingface', 'langchain',
        'mistral ai', 'cohere', 'adept', 'inflection',
        'deepmind', 'waymo', 'tesla ai', 'spacex ai',
        'baidu ai', 'alibaba ai', 'tencent ai', 'bydance ai', 'huawei ai',
        'xiaomi ai', 'sensetime', 'megvii', 'cloudwalk', 'yitu',
    ],

    # ========== AI伦理与安全 ==========
    "ethics": [
        'ai safety', 'ai alignment', 'interpretability', 'explainability', 'xai',
        'bias', 'fairness', 'transparency', 'accountability',
        'ai governance', 'ai regulation', 'eu ai act', 'ai executive order',
        'privacy', 'data protection', 'gdpr', 'consent',
        'deepfake', 'deepfake detection', 'synthetic media',
        'adversarial attack', 'prompt injection', 'jailbreak', 'ai security',
        'hallucination', 'fact-checking', 'verification',
    ],

    # ========== 新兴AI趋势 ==========
    "trends": [
        'world model', 'embodied ai', 'robotics learning',
        'reasoning model', 'reasoning', 'q*', 'q-star',
        'agentic ai', 'agentic workflow', 'tool use', 'function calling',
        'ai research', 'ai breakthrough', 'ai milestone', 'sota', 'state of the art',
        'ai benchmark', 'leaderboard', 'mmlu', 'hellaswag', 'gsm8k', 'arc',
        'human eval', 'bbh', 'agieval', 'mt-bench',
        'agi', 'general intelligence', 'superintelligence', 'singularity',
        'ai consciousness', 'ai sentience',
        'ai automation', 'job displacement', 'future of work', 'ai economy',
        'open source ai', 'opensource model', 'llama', 'mistral',
        'closed source ai', 'proprietary model', 'api model',
    ],

    # ========== 中文AI术语 ==========
    "chinese": [
        '人工智能', '机器学习', '深度学习', '神经网络', '大模型', '语言模型',
        '生成式ai', 'aigc', '多模态', '自动驾驶', '智能驾驶', '无人驾驶',
        '强化学习', '监督学习', '无监督学习', '半监督学习', '自监督学习',
        '图像识别', '语音识别', '自然语言处理', '计算机视觉', '机器视觉',
        '智能体', '代理', 'agent', '知识图谱', '知识库',
        '预训练', '微调', '提示工程', 'prompt', '提示词',
        '推理', '训练', '算力', '芯片', 'gpu', 'tpu',
        '文心一言', '通义千问', '悟道', '混元', '星火', '日日新',
        'kimi', '月之暗面', 'deepseek', '智谱', 'chatglm', '百川',
        '阿里云ai', '腾讯ai', '百度ai', '字节ai', '华为ai', '科大讯飞',
        '商汤', '旷视', '云从', '依图', '中科院', '清华ai',
    ],

    # ========== 学术/会议术语 ==========
    "academic": [
        'neurips', 'icml', 'iclr', 'cvpr', 'iccv', 'eccv', 'aaai', 'ijcai',
        'acl', 'emnlp', 'naacl', 'interspeech',
        'ai conference', 'ai summit', 'ai forum', 'ai workshop',
        'arxiv', 'arxiv paper', 'publication', 'citation', 'h-index',
        'peer review', 'paper', 'preprint', 'journal', 'conference',
    ],
}

# 展平所有关键词用于搜索
ALL_AI_KEYWORDS = []
for category_keywords in AI_KEYWORDS.values():
    ALL_AI_KEYWORDS.extend(category_keywords)

# 排除关键词 - 更精确的过滤
EXCLUDE_KEYWORDS = [
    # 招聘类
    '招聘', '求职', 'hiring', 'job opening', 'job posting', 'career', 'resume', 'cv',
    'we are hiring', 'now hiring', 'join our team', 'seeking', 'wanted', 'wanted:',

    # 广告推广类
    '广告', '营销', '推广', 'sponsored', 'advertisement', 'ad:', 'promo',
    'affiliate', 'paid promotion', 'native ad', 'promoted content',

    # 无关内容
    '股票', '股价', 'stock price', 'earnings call', 'financial report',
    'weather', 'sports', 'entertainment', 'gossip', 'celebrity', 'kardashian',

    # 低质量内容
    'clickbait', '标题党', 'you won', 'shocking', 'mind-blowing', 'unbelievable',

    # 游戏娱乐（除非是AI游戏）
    'game review', 'movie review', 'tv show', 'anime', 'manga', 'comic',

    # 普通科技新闻（非AI）
    'smartphone', 'iphone', 'android phone', 'laptop', 'tablet', 'tv',
]

# 新闻标题长度限制
MIN_TITLE_LENGTH = 20
MAX_TITLE_LENGTH = 300

# 请求配置 - 增强错误处理
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,ja;q=0.7',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Referer': 'https://www.google.com',
}

# ========== 新增：缓存和请求优化配置 ==========

# 缓存配置
CACHE_CONFIG = {
    "cache_dir": "cache",
    "news_expire_hours": 24,  # 新闻缓存过期时间（小时）
    "ai_cache_hours": 6,       # AI结果缓存过期时间（小时）
    "max_news_cache": 1000,    # 最大新闻缓存数量
    "max_ai_cache": 100,       # 最大AI缓存数量
}

# 请求重试配置
RETRY_CONFIG = {
    "max_retries": 3,           # 最大重试次数
    "initial_backoff": 1.0,     # 初始退避时间（秒）
    "max_backoff": 60.0,        # 最大退避时间（秒）
    "exponential_base": 2.0,    # 指数退避基数
    "jitter": True,             # 是否添加随机抖动
    "jitter_factor": 0.1,       # 抖动因子（0-1之间）
}

# 动态并发配置
DYNAMIC_CONCURRENCY_CONFIG = {
    "min_workers": 1,           # 最小工作线程数
    "max_workers": 10,          # 最大工作线程数
    "initial_workers": 3,       # 初始工作线程数
    "adjust_interval": 10,      # 调整间隔（每N次请求评估一次）
}

# 增量抓取配置
INCREMENTAL_FETCH_CONFIG = {
    "enabled": True,            # 是否启用增量抓取
    "check_hours": 24,          # 检查最近N小时的新闻
    "skip_cached": True,        # 是否跳过已缓存的新闻
}

# HTML解析选择器
NEWS_SELECTORS = [
    # 标准文章标题
    ('h1 a', 'href'),
    ('h2 a', 'href'),
    ('h3 a', 'href'),
    ('h4 a', 'href'),

    # 文章容器内的标题
    ('article h2 a', 'href'),
    ('article h3 a', 'href'),
    ('article h1 a', 'href'),
    ('article .title a', 'href'),
    ('article .post-title a', 'href'),

    # 常见class选择器
    ('.post-title a', 'href'),
    ('.entry-title a', 'href'),
    ('.article-title a', 'href'),
    ('.title a', 'href'),
    ('.headline a', 'href'),
    ('.story-title a', 'href'),
    ('.post__title a', 'href'),
    ('.card-title a', 'href'),

    # rel属性选择器
    ('a[rel="bookmark"]', 'href'),
    ('a[rel="bookmark"]', ''),

    # 特定网站选择器
    ('.post-item__title a', 'href'),
    ('.item-title a', 'href'),
    ('.feed-item-title a', 'href'),

    # 没有链接的标题
    ('h1', ''),
    ('h2', ''),
    ('h3', ''),
]

# 邮件主题模板
EMAIL_SUBJECT_TEMPLATE = "AI新闻日报 - {date}"

# 默认配置模板
DEFAULT_CONFIG = {
    "email": {
        "sender_email": "",
        "sender_password": "",
        "smtp_server": "smtp.qq.com",
        "smtp_port": 465,
        "recipient_email": ""
    },
    "settings": {
        "email_subject": "AI新闻日报 - {date}",
        "html_email": True,
        "qq_mail_format": True
    },
    "ai": {
        "enabled": True,
        "ollama_url": "http://localhost:11434",
        "model_name": "qwen2.5:7b-instruct",
        "temperature": 0.7,
        "max_tokens": 2000,
        "timeout": 60,
        "enable_filter": True
    },
    "fetcher": {
        "max_news_per_source": 5,
        "concurrent_requests": 3,
        "retry_times": 2,
        "retry_delay": 1,
        "enable_github": True,
        "enable_huggingface": True
    },
    "output": {
        "save_json": True,
        "save_html": True,
        "output_dir": "output"
    }
}

# AI提示词模板 - 优化版
AI_PROMPTS = {
    "summary": """请总结以下AI新闻，生成一个简洁但完整的摘要：

新闻列表：
{news_titles}

要求：
1. 按条目形式输出，每条新闻用数字序号列出
2. 每条用1-2句话概括核心内容，突出关键信息
3. 最后用一句话总结整体趋势
4. 语言简洁明了，突出重点
5. 总字数控制在400字以内
6. 重点关注技术突破、产品发布、行业动态

摘要：""",

    "trends": """基于以下AI新闻，分析当前AI领域的发展趋势：

新闻列表：
{news_titles}

请按以下结构输出（分段清晰，每点独立成段）：

【技术热点】
（列出2-3个当前最热门的技术方向，如大模型、多模态、AI Agent等，每个用一句话描述）

【行业发展】
（列出2-3个主要行业发展方向，如商业化、应用落地、基础设施等，每个用一句话描述）

【影响与变革】
（列出2-3个主要影响，如就业、教育、科研等，每个用一句话描述）

【前沿动态】
（列出2-3个前沿研究或突破，每个用一句话描述）

【总结】
（一句话总结整体趋势）

要求：
1. 严格按照上述格式分段输出
2. 每个点独立成段，不要挤在一起
3. 使用【】标记各个部分
4. 总字数控制在500字以内

分析结果："""
}

# 颜色主题（用于邮件模板）
COLOR_THEME = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "accent": "#1976d2",
    "success": "#4caf50",
    "warning": "#ff9800",
    "danger": "#f44336",
    "light": "#f5f7fa",
    "dark": "#2c3e50",
    "text": "#333333",
    "text_light": "#666666",
    "border": "#eeeeee"
}
