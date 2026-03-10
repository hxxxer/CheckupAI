#!/bin/bash

# vLLM 启动脚本
# 使用配置文件启动API服务

# 配置文件夹路径
CONFIG_FILE="$(dirname "\$0")/config.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

echo "使用配置文件: $CONFIG_FILE"
echo "启动vLLM服务..."

# 启动vLLM服务
vllm serve --config "$CONFIG_FILE"