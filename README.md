项目结构：

```bash
CheckupAI/
├── backend/                   # 后端核心
│   ├── ocr/                   # OCR模块
│   │   ├── paddle_table_parser.py
│   │   └── handwrite_enhancer.py
│   ├── ner/                   # 结构化解析
│   │   ├── medical_ner.py     # 基于BERT的NER
│   │   └── llm_validator.py   # LLM校验补全
│   ├── rag/                   # RAG系统
│   │   ├── retriever.py       # 双路检索（知识库+用户画像）
│   │   ├── user_profile.py    # 用户画像生成与更新
│   │   └── risk_guard.py      # 规则校验层
│   ├── model/                 # 模型相关
│   │   ├── lora_finetune/     # LoRA微调脚本+数据
│   │   └── inference.py       # 微调后Qwen3-8B推理
│   ├── api/                   # API接口（供前端调用）
│   │   └── main.py
│   └── config/                # 配置（Milvus连接、模型路径等）
├── frontend/                  # 前端（Chainlit应用）
│   ├── app.py                 # 主应用（含文件上传逻辑）
│   ├── components/            # 自定义UI组件
│   │   └── report_uploader.py
│   └── static/                # CSS/JS资源
├── data/                      # 数据目录
│   ├── sensitive/             # 敏感数据
│   │   │── raw_reports/       # 原始体检报告
│   │   │── processed/         # 结构化JSON
│   │   └── .gitignore         # 确保整个sensitive/被忽略！
│   ├── knowledge_base/        # 【独立目录：权威知识库】
│   │   ├── raw/               # 原始权威文档（PDF/Word）
│   │   ├── processed/         # 预处理后的文本块
│   │   └── scripts/           # 知识库构建脚本
│   └── finetune_data/         # LoRA微调数据
├── scripts/                   # 工具脚本
│   ├── build_knowledge_base.py # 构建权威知识库向量
│   └── profile_sync.py        # 用户画像同步到Milvus
├── requirements.txt
├── .gitignore
└── README.md
```