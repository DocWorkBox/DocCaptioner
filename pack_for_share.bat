@echo off
title DocCaptioner - 打包分享工具
chcp 65001 >nul
cls

echo ========================================================
echo        正在准备发布包 (Clean Release Build)
echo ========================================================
echo.

set "TARGET_DIR=DocCaptioner_Release"

if exist "%TARGET_DIR%" (
    echo [提示] 清理旧的发布文件夹...
    rmdir /s /q "%TARGET_DIR%"
)

echo [1/4] 创建目录结构...
mkdir "%TARGET_DIR%"

echo [2/4] 创建排除列表...
echo \venv\ > exclude.tmp
echo \.git\ >> exclude.tmp
echo \.trae\ >> exclude.tmp
echo \__pycache__\ >> exclude.tmp
echo \Dataset Collections\ >> exclude.tmp
echo \thumbnails\ >> exclude.tmp
echo \config.json\ >> exclude.tmp
echo \exclude.tmp\ >> exclude.tmp
echo \pack_for_share.bat\ >> exclude.tmp
echo \%TARGET_DIR%\ >> exclude.tmp

echo [3/4] 复制项目文件...
xcopy "." "%TARGET_DIR%\" /E /H /C /I /Y /EXCLUDE:exclude.tmp

echo [4/4] 清理临时文件...
del exclude.tmp

echo.
echo ========================================================
echo        打包完成！
echo ========================================================
echo.
echo 请将生成的文件夹 "%TARGET_DIR%" 压缩发送给对方。
echo.
echo 对方收到后的操作步骤：
echo 1. 解压文件夹
echo 2. 双击运行 install.bat (自动安装依赖)
echo 3. 双击运行 start.bat (启动程序)
echo.
pause