@echo off
echo 正在创建虚拟环境 seg...
python -m venv seg

echo 激活虚拟环境...
call seg\Scripts\activate.bat

echo 安装依赖...
pip install -r ..\requirements.txt

echo 完成！现在可以使用以下命令激活环境:
echo call seg\Scripts\activate.bat
pause 