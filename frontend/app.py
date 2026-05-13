"""
Chainlit Frontend for CheckupAI
支持问答模式 / 报告生成模式切换
"""

import chainlit as cl
import aiohttp
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings([
        cl.input_widget.Switch(
            id="report_mode",
            label="报告生成模式（输出结构化体检分析报告）",
            initial=False,
        ),
    ]).send()

    cl.user_session.set("report_mode", False)
    await cl.Message(
        content="欢迎使用 CheckupAI 体检报告分析系统！\n\n"
        "请输入您的健康问题，我将结合体检报告和医学知识为您解答。\n"
        "您也可以在设置中切换为「报告生成模式」以获得结构化的体检分析报告。"
    ).send()


@cl.on_settings_update
async def settings_update(updated_settings):
    cl.user_session.set("report_mode", updated_settings.get("report_mode", False))
    mode_name = "报告生成模式" if updated_settings.get("report_mode") else "问答模式"
    await cl.Message(content=f"已切换为：{mode_name}").send()


@cl.on_message
async def main(message: cl.Message):
    question = message.content.strip()
    if not question:
        await cl.Message(content="请输入您的问题。").send()
        return

    thinking_msg = cl.Message(content="正在检索和分析...")
    await thinking_msg.send()

    report_mode = cl.user_session.get("report_mode", False)
    endpoint = "/api/report" if report_mode else "/api/chat"

    try:
        async with aiohttp.ClientSession() as session:
            payload = {"question": question}
            async with session.post(
                f"{API_URL}{endpoint}", json=payload
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    answer = result.get("answer", "")
                    retrieval = result.get("retrieval", {})

                    formatted = _format_response(answer, retrieval, report_mode)
                    await cl.Message(content=formatted).send()
                else:
                    error = await resp.text()
                    await cl.Message(content=f"请求失败 ({resp.status}): {error}").send()
    except Exception as e:
        await cl.Message(content=f"请求出错: {str(e)}").send()
    finally:
        await thinking_msg.remove()


def _format_response(answer: str, retrieval: dict, report_mode: bool) -> str:
    """格式化回答和检索来源"""
    if report_mode:
        return answer  # 报告模式直接返回，不附加元信息

    parts = [answer]
    rewritten = retrieval.get("rewritten_query", "")
    indicators = retrieval.get("indicators", [])
    report_items = retrieval.get("report_items", [])
    knowledge_count = len(retrieval.get("knowledge_chunks", []))
    qa_count = len(retrieval.get("medical_qa", []))

    parts.append("\n---")
    parts.append(f"查询理解: {rewritten}")
    parts.append(f"涉及指标: {', '.join(indicators) if indicators else '无'}")
    parts.append(f"体检数据: {len(report_items)} 项相关指标")
    parts.append(f"权威知识: {knowledge_count} 条 | 问答参考: {qa_count} 条")

    return "\n".join(parts)
