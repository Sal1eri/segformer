#!/bin/bash
echo "正在激活虚拟环境..."
source seg/bin/activate

echo "开始训练..."
python ../train.py --config ../configs/config.yaml

echo "训练完成！" 