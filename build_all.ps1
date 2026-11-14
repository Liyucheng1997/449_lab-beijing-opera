# 一键重建全部贴图、模型和缩略图
# 用法：powershell -ExecutionPolicy Bypass -File build_all.ps1 [-Blender D:\Blender\blender.exe]
param([string]$Blender = "D:\Blender\blender.exe")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "1/4 绘制脸谱、发丝、刺绣贴图…"
python textures\masks.py
python textures\misc.py
python textures\fabrics.py

Write-Host "2/4 Blender：头部 + 脸谱 + 盔头 + 髯口…"
& $Blender -b --factory-startup -P blender\build_heads.py

Write-Host "3/4 Blender：戏服…"
& $Blender -b --factory-startup -P blender\build_costumes.py

Write-Host "4/4 缩略图转 WebP…"
python tools\to_webp.py
Write-Host "完成。运行 python -m http.server 8449 --directory site 后打开 http://localhost:8449"
