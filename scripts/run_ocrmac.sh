#!/bin/bash
# -----------------------------------------------------------------------------
# ocrmac 本地运行脚本
# 作用: 自动检测虚拟环境并运行 ocrmac CLI
# 用法: ./run_ocrmac.sh [图片路径] [参数]
# -----------------------------------------------------------------------------

# 获取脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 1. 尝试激活项目下的 .venv 虚拟环境 (推荐)
if [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
# 2. 或者尝试激活 venv 目录
elif [ -f "$DIR/venv/bin/activate" ]; then
    source "$DIR/venv/bin/activate"
fi

# 3. 检查依赖是否安装
python3 -c "import ocrmac" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 错误: 未检测到 ocrmac 模块。"
    echo "请先安装依赖: pip install -e ."
    exit 1
fi

# 4. 运行 CLI
# 使用 python -m ocrmac.cli 可以避免 path 问题
python3 -m ocrmac.cli "$@"
