"""
PDF 生成工具模块
使用 fpdf2 库生成包含中文内容的 PDF 文件
如果 fpdf2 不可用或找不到中文字体，则回退为保存 .txt 文件
"""
import os
import platform
import logging

logger = logging.getLogger(__name__)

# 常见中文字体路径（按优先级排列）
CHINESE_FONT_PATHS = {
    "Windows": [
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simsun.ttc",   # 宋体
        "C:/Windows/Fonts/simhei.ttf",   # 黑体
        "C:/Windows/Fonts/simkai.ttf",   # 楷体
    ],
    "Linux": [
        "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ],
    "Darwin": [
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ],
}


def find_chinese_font() -> str | None:
    """
    在系统中查找可用的中文字体文件
    返回字体文件路径，找不到则返回 None
    """
    system = platform.system()
    candidates = CHINESE_FONT_PATHS.get(system, [])
    for path in candidates:
        if os.path.exists(path):
            logger.info(f"找到中文字体: {path}")
            return path
    logger.warning(f"未找到中文字体 (系统: {system})")
    return None


def generate_pdf(text: str, output_path: str, title: str = "修改后的文档") -> tuple[bool, str]:
    """
    将文本内容生成为 PDF 文件

    Args:
        text: 要写入 PDF 的文本内容
        output_path: 输出文件路径（应以 .pdf 结尾）
        title: PDF 文档标题

    Returns:
        tuple[bool, str]: (是否成功, 实际输出文件路径)
            如果 fpdf2 不可用或找不到中文字体，会回退保存为 .txt 文件
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 未安装，回退保存为 .txt 文件")
        return _save_as_txt(text, output_path)

    # 查找中文字体
    font_path = find_chinese_font()
    if not font_path:
        logger.warning("找不到中文字体，回退保存为 .txt 文件")
        return _save_as_txt(text, output_path)

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # 添加中文字体
        font_name = "ChineseFont"
        # fpdf2 支持 .ttc 文件，需要指定 style 或 index
        if font_path.endswith(".ttc"):
            # .ttc 是字体集合，fpdf2 2.7+ 支持通过 style 参数
            pdf.add_font(font_name, "", font_path, uni=True)
        else:
            pdf.add_font(font_name, "", font_path, uni=True)

        pdf.set_font(font_name, "", 12)

        # 逐行写入文本
        lines = text.split("\n")
        for line in lines:
            # 处理空行
            if not line.strip():
                pdf.ln(6)
                continue
            # fpdf2 的 multi_cell 会自动换行
            pdf.multi_cell(0, 7, line)

        pdf.output(output_path)
        logger.info(f"PDF 生成成功: {output_path}")
        return True, output_path

    except Exception as e:
        logger.error(f"PDF 生成失败: {e}，回退保存为 .txt 文件")
        return _save_as_txt(text, output_path)


def _save_as_txt(text: str, output_path: str) -> tuple[bool, str]:
    """
    回退方案：将文本保存为 .txt 文件
    自动将 .pdf 扩展名改为 .txt
    """
    if output_path.lower().endswith(".pdf"):
        output_path = output_path[:-4] + ".txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"已保存为文本文件: {output_path}")
    return True, output_path
