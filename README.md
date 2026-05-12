# CheckupAI

体检报告 AI 分析系统：OCR 识别 → 结构化解析 → 向量检索 → LLM 分析 → 健康建议。

## 项目结构

```
CheckupAI/
├── backend/
│   ├── main.py                  # FastAPI 入口（待重构）
│   ├── config/                  # 配置管理
│   │   ├── settings.py          # TOML 解析 + 路径校验
│   │   └── settings.toml        # OCR/LLM 环境路径配置
│   ├── ocr/                     # OCR 模块
│   │   ├── schema.py            # 数据模型定义（dataclass）
│   │   ├── runner.py            # PaddleOCR subprocess 调用
│   │   ├── paddle_runner.py     # PaddleOCR 子进程脚本
│   │   ├── parser.py            # 通用解析器（表格分类 + LLM 解析）
│   │   ├── checkup_parser.py    # 高层入口 parse_checkup()
│   │   └── utils.py             # HTML→Markdown 转换
│   ├── llm/                     # LLM 模块（vLLM + OpenAI API）
│   │   ├── base_llm.py          # OpenAI 兼容客户端基类
│   │   ├── table_parser.py      # 表格结构化解析
│   │   ├── table_parse_router.py# 表格类型分类
│   │   ├── text_analyzer.py     # 文本分析（个人信息/阳性发现提取）
│   │   ├── utils.py             # JSON 安全解析
│   │   ├── config.yaml          # vLLM 服务配置
│   │   └── prompt_templates/    # TOML 格式 Prompt 模板
│   ├── rag/                     # RAG 检索（Milvus Server 模式，legacy）
│   │   ├── retriever.py         # 双路检索（知识库 + 用户画像）
│   │   ├── reranker.py          # BGE-Reranker 重排序
│   │   ├── risk_guard.py        # 规则校验层
│   │   └── user_profile.py      # 用户画像管理
│   └── vector/                  # 向量入库模块（Milvus Lite + BGE-M3）
│       ├── config.py            # Collection Schema 定义
│       ├── embeddings.py        # BGE-M3 编码器封装
│       ├── collections.py       # 建库/初始化
│       ├── ingest_report.py     # 体检报告入库
│       ├── ingest_knowledge.py  # 知识库分块入库
│       └── ingest_qa.py         # 问答资料入库
├── frontend/
│   └── app.py                   # Chainlit 聊天界面
├── training/                    # 模型训练（独立模块）
│   ├── inference.py             # Qwen3-8B 直接推理（legacy）
│   └── lora_finetune/           # LoRA 微调脚本
├── scripts/
│   ├── build_knowledge_base.py  # 知识库向量构建（legacy）
│   └── profile_sync.py          # 用户画像同步（legacy）
├── tests/
│   └── test_ocr/                # OCR 测试（含测试图片）
├── data/
│   ├── sensitive/               # 用户数据（gitignore）
│   ├── knowledge_base/          # 权威知识库文档
│   │   ├── raw/                 # 原始 PDF
│   │   ├── processed/           # 预处理文本
│   │   └── scripts/             # 清洗/分块脚本
│   └── finetune_data/           # LoRA 训练数据
├── pyproject.toml               # Python 3.12+
├── requirements.txt
└── .env.example
```

## 技术栈

| 类别 | 技术 |
|------|------|
| OCR | PaddleOCR（独立 conda 环境，subprocess 调用） |
| 结构化解析 | Qwen3.5-4B（vLLM + OpenAI API） |
| Embedding | BGE-M3（Dense + Sparse 双向量） |
| 向量数据库 | Milvus Lite（本地文件，无需服务进程） |
| LLM 推理 | vLLM 服务（Qwen3.5-4B） |
| 前端 | Chainlit |
| 后端框架 | FastAPI |
| Python | 3.12+（使用内置 tomllib） |

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 复制并填写配置文件
cp .env.example .env
cp backend/config/settings.toml.example backend/config/settings.toml
```

`settings.toml` 中需配置 `ocr_python` 指向装有 PaddleOCR 的独立 conda 环境。

### 2. 启动服务

```bash
# 启动 vLLM 服务
bash backend/llm/start_server.sh

# 启动后端 API（端口 8000）
python backend/main.py

# 启动前端（端口 8080）
chainlit run frontend/app.py
```

### 3. 向量数据入库

```python
from backend.vector import (
    get_milvus_client,
    ingest_checkup_report,
    ingest_knowledge_chunks,
    ingest_qa_from_csv,
)

# 初始化 Milvus Lite（data/milvus_vectors.db）
client = get_milvus_client()

# 体检报告入库（自动解析并写入 report_pages + report_items）
from backend.ocr import parse_checkup
results = parse_checkup("path/to/report.jpg")
ingest_checkup_report(client, results)

# 知识库入库（从 data/knowledge_base/final_chunks/chunks.json）
ingest_knowledge_chunks(client)

# 问答资料入库（CSV: department, title, ask, answer）
ingest_qa_from_csv(client, "data/qa.csv")
```

### 4. 直接解析体检报告（无需 API）

```python
from backend.ocr import parse_checkup
results = parse_checkup("path/to/report.jpg")
# 返回 list[OCRResult]，包含完整的结构化数据
```

## 向量数据库 Collection

| Collection | 数据来源 | 用途 |
|---|---|---|
| `report_pages` | OCRResult → Page | 页面完整信息（summary 向量化） |
| `report_items` | TableItem | 检验项目精确/语义检索 |
| `knowledge_chunks` | PDF 分块 | 权威医学知识检索 |
| `medical_qa` | CSV 问答 | 医疗问答检索（含科室路由） |

## 测试

```bash
# OCR 模块测试（需要测试图片）
pytest tests/test_ocr/

# 单次管道测试
python tests/test_ocr/test_checkup_parse_pipeline.py
```

## 注意事项

- `data/sensitive/` 目录已在 `.gitignore` 中完全忽略
- 模型权重文件（`.pth`, `.bin`, `.safetensors`）不提交
- OCR 在独立 Python 环境中运行，`settings.toml` 中路径必须正确
- `backend/rag/` 为 legacy Milvus Server 模式，新开发使用 `backend/vector/`（Milvus Lite）
- `training/` 为独立模块，与主流水线解耦
