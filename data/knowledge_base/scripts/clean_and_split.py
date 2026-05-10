import os
import re
from pathlib import Path
from typing import List, Dict, Optional

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain.schema import Document
from transformers import AutoTokenizer


def get_qwen_tokenizer(model_path: str = None) -> AutoTokenizer:
    """
    获取 Qwen tokenizer
    
    Args:
        model_path: Qwen 模型路径，如果为 None 则使用默认路径
    
    Returns:
        Qwen tokenizer 实例
    """
    if model_path is None:
        model_path = os.getenv("QWEN_MODEL_PATH", "/root/autodl-tmp/models/Qwen/Qwen3.5-4B")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"成功加载 Qwen tokenizer from: {model_path}")
        return tokenizer
    except Exception as e:
        print(f"警告: 无法从 {model_path} 加载 tokenizer，尝试使用在线模型...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B",
                trust_remote_code=True
            )
            print("成功加载在线 Qwen tokenizer")
            return tokenizer
        except Exception as e2:
            raise RuntimeError(f"无法加载 Qwen tokenizer: {e}\n{e2}")


def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """
    使用 Qwen tokenizer 计算文本的 token 数量
    
    Args:
        text: 输入文本
        tokenizer: Qwen tokenizer 实例
    
    Returns:
        token 数量
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def read_markdown_files(input_dir: str) -> List[Document]:
    """读取指定目录下的所有 Markdown 文件"""
    documents = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    md_files = list(input_path.glob("*.md"))
    
    if not md_files:
        raise ValueError(f"在目录 {input_dir} 中未找到任何 .md 文件")
    
    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    "filename": md_file.name,
                }
            )
            documents.append(doc)
            print(f"已读取文件: {md_file.name}")
        except Exception as e:
            print(f"读取文件 {md_file.name} 时出错: {e}")
    
    return documents


def clean_text(raw_text: str) -> str:
    """
    清洗文本：去除页眉页脚、页码、水印、多余空白等噪声

    Args:
        raw_text: 原始文本

    Returns:
        清洗后的文本
    """
    text = raw_text

    # 移除常见页眉页脚模式（如“国家卫生健康委员会”、“第x页”等）
    footer_patterns = [
        r'第?\s*\d+\s*页',
        r'国家卫生健康委员会',
        r'国卫办.*?函〔\d+〕\d+号',
        r'http[s]?://[^\s]+',
        r'www\.[^\s]+',
        r'\b版权所有\b.*',
        r'\(.*?卫健委.*?\)'
    ]
    for pattern in footer_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 移除孤立数字、特殊符号行（可能是页码或分隔符）
    lines = [line.strip() for line in text.split('\n')]
    cleaned_lines = []
    for line in lines:
        # 过滤纯符号行、纯数字行、过短无意义行
        if (len(line) < 5 and (re.fullmatch(r'[·●◆■▲►◆◇]', line) or line.isdigit())) or \
            re.fullmatch(r'[─━═—―]{5,}', line):
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # 统一空白字符
    text = re.sub(r'\n{3,}', "\n\n", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # 移除首尾空白
    return text.strip()


def split_by_headers(documents: List[Document]) -> List[Document]:
    """使用 Markdown 标题分割器按章节分割文档"""
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    header_splits = []
    for doc in documents:
        try:
            splits = markdown_splitter.split_text(doc.page_content)
            
            for split in splits:
                split.metadata.update(doc.metadata)
                
                if split.page_content.strip():
                    header_splits.append(split)
        except Exception as e:
            print(f"分割文档 {doc.metadata.get('filename', 'unknown')} 时出错: {e}")
    
    print(f"按标题分割后得到 {len(header_splits)} 个片段")
    return header_splits


def save_chunks(
    documents: List[Document],
    output_dir: str,
    format: str = "json"
) -> None:
    """保存分块结果到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    if format == "json":
        import json
        
        chunks_data = []
        for doc in documents:
            chunk_info = {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            chunks_data.append(chunk_info)
        
        output_file = os.path.join(output_dir, "chunks.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(chunks_data)} 个片段到: {output_file}")
    
    elif format == "txt":
        for i, doc in enumerate(documents):
            filename = doc.metadata.get("filename", f"doc_{i}")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            output_file = os.path.join(
                output_dir,
                f"{Path(filename).stem}_chunk_{chunk_idx}.txt"
            )
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(doc.page_content)
        
        print(f"已保存 {len(documents)} 个片段到: {output_dir}")


def process_knowledge_base(
    input_dir: str = "./processed",
    output_dir: str = "./final_chunks",
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 50,
    save_format: str = "json",
    model_path: str = None
) -> List[Document]:
    """
    完整的知识库处理流程（使用 Token 计数）
    
    Args:
        input_dir: Markdown 文件输入目录
        output_dir: 分块结果输出目录
        chunk_size_tokens: 每个分块的 token 数量
        chunk_overlap_tokens: 分块重叠的 token 数量
        save_format: 保存格式 ('json' 或 'txt')
        model_path: Qwen 模型路径
    
    Returns:
        分块后的文档列表
    """
    print("=" * 60)
    print("开始处理知识库（Token 模式）...")
    print("=" * 60)
    
    print("\n步骤 0: 加载 Qwen Tokenizer")
    tokenizer = get_qwen_tokenizer(model_path)
    print()
    
    print("步骤 1: 读取 Markdown 文件")
    documents = read_markdown_files(input_dir)
    print(f"共读取 {len(documents)} 个文档\n")
    
    print("步骤 2: 清洗文本内容")
    cleaned_docs = []
    total_chars = 0
    total_tokens = 0
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        cleaned_doc = Document(
            page_content=cleaned_content,
            metadata=doc.metadata.copy()
        )
        cleaned_docs.append(cleaned_doc)
        
        chars = len(cleaned_content)
        tokens = count_tokens(cleaned_content, tokenizer)
        total_chars += chars
        total_tokens += tokens
    
    avg_tokens_per_char = total_tokens / total_chars if total_chars > 0 else 0
    print(f"完成文本清洗")
    print(f"总字符数: {total_chars:,}")
    print(f"总 Token 数: {total_tokens:,}")
    print(f"平均每个字符的 Token 数: {avg_tokens_per_char:.3f}\n")
    
    print("步骤 3: 按章节标题分割")
    header_splits = split_by_headers(cleaned_docs)
    print()
    
    print("步骤 4: 基于 Token 的递归分块")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ".", "!", "?", ";", ",", " ", ""]
    )
    
    final_chunks = text_splitter.split_documents(header_splits)
    
    for i, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_index"] = i % 1000
        chunk.metadata["total_chunks"] = len(final_chunks)
    
    print(f"递归分割后得到 {len(final_chunks)} 个最终片段")
    print()
    
    print("步骤 5: 保存分块结果")
    save_chunks(final_chunks, output_dir, format=save_format)
    print()
    
    print("=" * 60)
    print("知识库处理完成！")
    print(f"输入文件: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"总分块数: {len(final_chunks)}")
    print(f"每块 Token 数: {chunk_size_tokens}")
    print(f"重叠 Token 数: {chunk_overlap_tokens}")
    print("=" * 60)
    
    return final_chunks


if __name__ == "__main__":
    input_directory = "./processed"
    output_directory = "./final_chunks"
    
    chunks = process_knowledge_base(
        input_dir=input_directory,
        output_dir=output_directory,
        chunk_size_tokens=512,
        chunk_overlap_tokens=50,
        save_format="json"
    )
    
    print("\n示例分块预览:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- 分块 {i+1} ---")
        print(f"来源: {chunk.metadata.get('filename', 'N/A')}")
        print(f"标题层级: {[k for k in chunk.metadata.keys() if k.startswith('Header_')]}")
        print(f"内容预览: {chunk.page_content[:200]}...")
        print(f"字符数: {len(chunk.page_content)}")