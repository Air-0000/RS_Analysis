#!/bin/bash
# 变化检测数据集下载脚本
# 支持多个源，自动选择可用镜像
#
# 数据集: LEVIR-CD (建筑变化检测, 256×256 预处理版)

set -e

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DATA_DIR"
TARGET_DIR="LEVIR-CD256"

echo "========================================="
echo "  📥 变化检测数据集下载"
echo "========================================="
echo ""

if [ -d "$TARGET_DIR" ] && [ -f "$TARGET_DIR/list/train.txt" ]; then
    echo "  ✓ 数据集已存在: $TARGET_DIR"
    echo "    训练样本: $(wc -l < $TARGET_DIR/list/train.txt)"
    echo "    验证样本: $(wc -l < $TARGET_DIR/list/val.txt)"
    echo "    测试样本: $(wc -l < $TARGET_DIR/list/test.txt)"
    exit 0
fi

# === 下载源列表（按优先级）===
# 源1: Google Drive (ChangeFormer 预处理版)
# 原 Dropbox 链接 (被墙)
# 源2: 直接从官方源下载原始数据自行裁切
# 源3: 使用 HuggingFace datasets 库
# 源4: 百度网盘 (国内用户推荐)

echo "请选择下载方式:"
echo ""
echo "  1) 百度网盘 (国内用户推荐)"
echo "    链接: https://pan.baidu.com/s/1xyz..."
echo "    提取码: 请查看 README 或项目主页"
echo ""
echo "  2) 手动下载后解压"
echo "    1. 访问 https://justchenhao.github.io/LEVIR/"
echo "    2. 下载 LEVIR-CD 原始数据"
echo "    3. 自行裁切为 256×256"
echo ""
echo "  3) 使用 Python 下载 (尝试所有镜像)"
echo ""

read -p "请选择 [1-3] (默认 3): " CHOICE
CHOICE=${CHOICE:-3}

if [ "$CHOICE" = "1" ]; then
    echo "请手动从百度网盘下载, 然后解压到 $DATA_DIR/$TARGET_DIR/"
    echo "下载后再次运行此脚本确认"
    exit 0
elif [ "$CHOICE" = "2" ]; then
    echo "请手动下载并解压"
    echo "解压后结构应为:"
    echo "  $TARGET_DIR/"
    echo "  ├── A/         # T1 图像"
    echo "  ├── B/         # T2 图像"
    echo "  ├── label/     # 变化标注"
    echo "  └── list/      # 数据划分"
    exit 0
fi

# 选项3: 尝试 Python 下载
echo ""
echo "📦 尝试使用 Python 下载..."
python3 -c "
import urllib.request, os, sys, zipfile

# 尝试多个 Dropbox 镜像
urls = [
    'https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip',
    'https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=1',
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
}

for url in urls:
    try:
        print(f'尝试: {url[:60]}...')
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            print(f'文件大小: {total/1024/1024:.1f} MB')
            
            downloaded = 0
            with open('LEVIR-CD256.zip', 'wb') as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded/total*100
                        sys.stdout.write(f'\r下载中: {downloaded/1024/1024:.1f}/{total/1024/1024:.1f}MB ({pct:.1f}%)')
                        sys.stdout.flush()
            
            print(f'\n下载完成!')
            
            # 解压
            print('解压中...')
            with zipfile.ZipFile('LEVIR-CD256.zip', 'r') as zf:
                zf.extractall('LEVIR-CD256')
            
            os.remove('LEVIR-CD256.zip')
            print('解压完成!')
            
            # 验证
            if os.path.exists('LEVIR-CD256/list/train.txt'):
                with open('LEVIR-CD256/list/train.txt') as f:
                    n = len(f.readlines())
                print(f'训练样本数: {n}')
                print('✅ 数据集准备就绪!')
            sys.exit(0)
    except Exception as e:
        print(f'失败: {e}')
        continue

print()
print('⚠️  所有镜像均无法连接')
print('请手动下载数据集')
" 2>&1

echo ""
echo "若下载失败, 解决方法:"
echo "  1. 使用代理 / VPN"
echo "  2. 从百度网盘下载 (见 README)"
echo "  3. 联系助教获取数据集"
