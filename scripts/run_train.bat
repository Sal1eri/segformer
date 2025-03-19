@echo off
echo 正在激活虚拟环境...
call seg\Scripts\activate.bat

echo 开始训练...
python ..\train.py --config ..\configs\config.yaml

echo 训练完成！
pause 