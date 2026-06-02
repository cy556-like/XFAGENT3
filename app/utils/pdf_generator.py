"""
PDF 生成工具模块
使用 fpdf2 生成 PDF，支持中文字体
如果 fpdf2 不可用，自动降级为保存 .txt 文件
"""
import os
import platform
import logging

logger = logging.getLogger(__name__)

CHINESE_FONT_PATHS = {
    "Windows": [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ],
    "Darwin": [
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ],
}


def find_chinese_font():
    """查找系统中可用的中文字体"""
    system = platform.system()
    for path in CHINESE_FONT_PATHS.get(system, []):
        if os.path.exists(path):
            return path
    return None


def generate_pdf(text, output_path, title="修改后的文档"):
    """
    生成 PDF 文件

    Args:
        text: 文本内容
        output_path: 输出路径
        title: 文档标题

    Returns:
        tuple: (success: bool, actual_path: str)
            如果 PDF 生成成功，返回 (True, pdf_path)
            如果降级为 txt，返回 (True, txt_path)
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 未安装，降级为 txt 文件")
        return _save_as_txt(text, output_path)

    font_path = find_chinese_font()
    if not font_path:
        logger.warning("未找到中文字体，降级为 txt 文件")
        return _save_as_txt(text, output_path)

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # 添加中文字体
        pdf.add_font("ChineseFont", "", font_path, uni=True)
        pdf.set_font("ChineseFont", "", 12)

        # 写入内容
        for line in text.split("\n"):
            if not line.strip():
                pdf.ln(6)
                continue
            pdf.multi_cell(0, 7, line)

        pdf.output(output_path)
        return True, output_path

    except Exception as e:
        logger.error(f"PDF 生成失败: {e}，降级为 txt 文件")
        return _save_as_txt(text, output_path)


def generate_chat_pdf(messages: list, session_id: str) -> bytes:
    """
    生成对话导出 PDF（返回 bytes）

    Args:
        messages: 对话消息列表 [{"role": "user"/"assistant", "content": "..."}]
        session_id: 会话 ID

    Returns:
        PDF 文件的 bytes 内容
    """
    from fpdf import FPDF

    font_path = find_chinese_font()
    if not font_path:
        raise RuntimeError("未找到中文字体，无法生成 PDF")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 添加中文字体
    pdf.add_font("ChineseFont", "", font_path, uni=True)
    pdf.set_font("ChineseFont", "", 12)

    # 标题
    pdf.set_font("ChineseFont", "", 16)
    pdf.cell(0, 12, "DocAgent 对话记录", ln=True, align="C")
    pdf.set_font("ChineseFont", "", 10)
    pdf.cell(0, 8, f"Session: {session_id[:12]}", ln=True, align="C")
    pdf.ln(8)

    # 写入对话内容
    for msg in messages:
        role = "用户" if msg["role"] == "user" else "助手"
        content = msg.get("content", "")

        # 角色标签
        pdf.set_font("ChineseFont", "", 11)
        if msg["role"] == "user":
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 8, f"  {role}：", ln=True, fill=True)
        else:
            pdf.set_fill_color(245, 245, 255)
            pdf.cell(0, 8, f"  {role}：", ln=True, fill=True)

        # 内容
        pdf.set_font("ChineseFont", "", 10)
        for line in content.split("\n"):
            if not line.strip():
                pdf.ln(3)
                continue
            pdf.multi_cell(0, 6, f"  {line}")

        pdf.ln(4)

    # 输出为 bytes
    return pdf.output()


def _save_as_txt(text, output_path):
    """降级方案：保存为 txt 文件"""
    if output_path.lower().endswith(".pdf"):
        output_path = output_path[:-4] + ".txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return True, output_path
