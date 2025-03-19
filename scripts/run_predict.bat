@echo off
echo 正在激活虚拟环境...
call seg\Scripts\activate.bat

echo 开始预测...
python ..\predict.py --checkpoint ..\checkpoints\best_model.pth --input-dir ..\test_images --output-dir ..\predictions --visual

echo 预测完成！结果保存在 predictions 目录中。
pause 