#!/bin/bash

# UAV Nest Agent Setup Script
# 自动化设置脚本，简化安装流程

set -e  # 遇到错误立即退出

echo "============================================================"
echo "🚀 UAV Nest Agent - 自动安装脚本"
echo "============================================================"
echo ""

# 检查 Python 3.12 版本
echo "📋 步骤 1/6: 检查 Python 3.12 版本..."

# 优先使用 python3.12，如果不存在则尝试 python3
PYTHON_CMD=""
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    PYTHON_VERSION=$(python3.12 --version | cut -d' ' -f2)
    echo "✅ 找到 Python $PYTHON_VERSION (使用 python3.12)"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    # 检查是否是 Python 3.12
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 12 ]; then
        PYTHON_CMD="python3"
        echo "✅ 找到 Python $PYTHON_VERSION"
    else
        echo "❌ 错误: 需要 Python 3.12"
        echo "   当前系统 python3 版本: $PYTHON_VERSION"
        echo ""
        echo "请安装 Python 3.12："
        echo "   macOS: brew install python@3.12"
        echo "   或访问: https://www.python.org/downloads/"
        exit 1
    fi
else
    echo "❌ 错误: 未找到 Python 3.12"
    echo ""
    echo "请安装 Python 3.12："
    echo "   macOS: brew install python@3.12"
    echo "   或访问: https://www.python.org/downloads/"
    exit 1
fi

# 创建虚拟环境（使用 Python 3.12）
echo ""
echo "📋 步骤 2/6: 创建虚拟环境 (使用 Python 3.12)..."
if [ -d ".venv" ]; then
    echo "⚠️  虚拟环境已存在，跳过创建"
    echo "   如需重新创建，请先删除: rm -rf .venv"
else
    $PYTHON_CMD -m venv .venv
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境
echo ""
echo "📋 步骤 3/6: 激活虚拟环境..."
source .venv/bin/activate
echo "✅ 虚拟环境已激活"

# 升级 pip
echo ""
echo "📋 步骤 4/6: 升级 pip..."
pip install --upgrade pip -q
echo "✅ pip 升级完成"

# 安装依赖
echo ""
echo "📋 步骤 5/6: 安装项目依赖..."
echo "   这可能需要几分钟，请耐心等待..."
pip install -r requirements.txt -q
echo "✅ 依赖安装完成"

# 配置环境变量（可选）
echo ""
echo "📋 步骤 6/6: 配置环境变量（可选）..."
if [ -f ".env" ]; then
    echo "✅ 找到 .env 文件"
    echo "   程序运行时将使用该文件中的 API Key"
else
    echo "ℹ️  未找到 .env 文件"
    echo "   这是正常的！程序运行时会提示你输入 API Key"
    echo ""
    echo "💡 如果你希望避免每次输入 API Key，可以创建 .env 文件："
    echo "   1. 复制示例文件: cp env.example.txt .env"
    echo "   2. 编辑并填入: nano .env"
    echo "   3. 填入 DEEPSEEK_API_KEY=你的实际key"
    echo ""
    echo "   获取 DeepSeek API Key: https://platform.deepseek.com/"
fi

# 检查 PDF 文件
echo ""
echo "📄 检查 PDF 文件..."
if [ -f "Autel-Robotics-Products-Brochure.pdf" ]; then
    echo "✅ 找到 PDF 文件: Autel-Robotics-Products-Brochure.pdf"
else
    echo "⚠️  未找到 PDF 文件"
    echo "   请将 Autel Robotics 产品手册 PDF 文件放在项目根目录"
    echo "   并命名为: Autel-Robotics-Products-Brochure.pdf"
fi

# 完成
echo ""
echo "============================================================"
echo "🎉 安装完成！"
echo "============================================================"
echo ""
echo "📝 后续步骤:"
echo ""
echo "   1. 确保 PDF 文件存在:"
echo "      ls -lh Autel-Robotics-Products-Brochure.pdf"
echo ""
echo "   2. 运行程序（会自动提示输入 API Key）:"
echo "      source ./.venv/bin/activate"
echo "      python uav-nest-agent-by-langchain.py  # 演示模式"
echo ""
echo "💡 提示:"
echo "   - 运行时会要求输入 DeepSeek API Key（输入时隐藏显示）"
echo "   - 或者预先配置 .env 文件避免每次输入"
echo "   - 记得先激活虚拟环境: source .venv/bin/activate"
echo ""
echo "============================================================"

