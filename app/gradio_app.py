"""
Gradio 前端界面
提供一个简洁的聊天界面，挂载到 FastAPI 上
"""
import os
import gradio as gr
from app.agent.core import chat
from app.rag.document import index_document, list_indexed_documents, delete_document
from app.config import settings


def gradio_chat(message: str, history: list, session_id: str) -> str:
    if not message.strip():
        return "请输入您的问题"
    try:
        response = chat(message, session_id=session_id)
        return response
    except Exception as e:
        return f"出错了: {str(e)}\n\n请检查：\n1. API Key 是否正确\n2. 网络是否通畅\n3. 模型名称是否正确"


def gradio_upload(file) -> str:
    if file is None:
        return "请选择文件上传"
    try:
        result = index_document(file.name, os.path.basename(file.name))
        return f"{result['message']}"
    except Exception as e:
        return f"上传失败: {str(e)}"


def gradio_list_docs() -> str:
    docs = list_indexed_documents()
    if not docs:
        return "知识库中暂无文档，请先上传。"
    output = f"知识库中共有 {len(docs)} 个文档：\n\n"
    for i, doc in enumerate(docs, 1):
        output += f"  {i}. {doc}\n"
    return output


def gradio_refresh_doc_dropdown():
    """刷新文档下拉列表，返回更新后的 choices 和默认值"""
    docs = list_indexed_documents()
    if not docs:
        return gr.update(choices=[], value=None)
    return gr.update(choices=docs, value=docs[0])


def gradio_delete_doc(filename) -> tuple:
    """删除选中的文档，返回删除结果、更新后的文档列表、更新后的下拉框"""
    if not filename:
        result_msg = "请先选择要删除的文档"
        dropdown = gradio_refresh_doc_dropdown()
        return result_msg, gradio_list_docs(), dropdown

    try:
        result = delete_document(filename)
        result_msg = result["message"]
    except Exception as e:
        result_msg = f"删除失败: {str(e)}"

    # 刷新列表和下拉框
    dropdown = gradio_refresh_doc_dropdown()
    docs_text = gradio_list_docs()
    return result_msg, docs_text, dropdown


def create_gradio_app():
    with gr.Blocks(title="企业文档智能助手") as demo:

        gr.Markdown("# 企业文档智能助手\n> 基于 LangChain + LangGraph 的 ReAct Agent\n\n你可以问我：公司制度、员工信息、文档内容等任何问题")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="对话", height=500)
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入问题",
                        placeholder="例如：公司年假制度是什么？",
                        scale=4,
                        lines=1,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                session_id = gr.Textbox(
                    label="会话 ID",
                    value="default",
                    visible=True,
                    scale=1,
                )

            with gr.Column(scale=1):
                gr.Markdown("### 文档管理")
                file_input = gr.File(
                    label="上传文档",
                    file_types=[".pdf", ".txt", ".docx", ".xlsx", ".xls"],
                )
                upload_btn = gr.Button("上传到知识库", variant="secondary")
                upload_result = gr.Textbox(label="上传结果", interactive=False)
                list_btn = gr.Button("查看知识库文档")
                docs_list = gr.Textbox(label="文档列表", interactive=False, lines=8)

                gr.Markdown("---")
                gr.Markdown("### 删除文档")
                doc_dropdown = gr.Dropdown(
                    label="选择要删除的文档",
                    choices=[],
                    interactive=True,
                    allow_custom_value=True,
                )
                refresh_btn = gr.Button("刷新文档列表", size="sm")
                delete_btn = gr.Button("删除选中文档", variant="stop")
                delete_result = gr.Textbox(label="删除结果", interactive=False)

        def send_message(message, history, sid):
            response = gradio_chat(message, history, sid)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return "", history

        msg_input.submit(
            send_message,
            inputs=[msg_input, chatbot, session_id],
            outputs=[msg_input, chatbot],
        )

        send_btn.click(
            send_message,
            inputs=[msg_input, chatbot, session_id],
            outputs=[msg_input, chatbot],
        )

        upload_btn.click(
            gradio_upload,
            inputs=[file_input],
            outputs=[upload_result],
        )

        list_btn.click(
            gradio_list_docs,
            outputs=[docs_list],
        )

        refresh_btn.click(
            gradio_refresh_doc_dropdown,
            outputs=[doc_dropdown],
        )

        delete_btn.click(
            gradio_delete_doc,
            inputs=[doc_dropdown],
            outputs=[delete_result, docs_list, doc_dropdown],
        )

        # 页面加载时自动刷新下拉框
        demo.load(
            gradio_refresh_doc_dropdown,
            outputs=[doc_dropdown],
        )

    return demo
