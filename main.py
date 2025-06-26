"""
main.py - 自助式数据分析（数据分析智能体）

Author: 骆昊
Version: 0.5
Date: 2025/6/26
"""
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd
import streamlit as st
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import docx
import hashlib
from utils import dataframe_agent, qa_cache

# 添加全局样式
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f9;
            color: #333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stTextArea textarea {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
        }
        .stExpander {
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .stExpander>div {
            padding: 20px;
        }
        .stTable {
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def create_chart(input_data, chart_type):
    """生成统计图表"""
    df_data = pd.DataFrame(
        data={
            "x": input_data["columns"],
            "y": input_data["data"]
        }
    ).set_index("x")
    if chart_type == "bar":
        plt.figure(figsize=(8, 5), dpi=120)
        plt.bar(input_data["columns"], input_data["data"], width=0.4, hatch='///')
        st.pyplot(plt.gcf())
    elif chart_type == "line":
        st.line_chart(df_data)

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

def read_txt(file):
    return file.read().decode("utf-8")

def get_data_hash(df):
    """计算DataFrame的哈希值，用于缓存键生成"""
    # 取前1000行和所有列计算哈希，避免大数据集计算缓慢
    sample_df = df.head(1000) if len(df) > 1000 else df
    return hashlib.sha256(pd.util.hash_pandas_object(sample_df).values).hexdigest()

# 初始化会话状态
if "df" not in st.session_state:
    st.session_state["df"] = None

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory()

if "history" not in st.session_state:
    st.session_state["history"] = []

if "current_data_hash" not in st.session_state:
    st.session_state["current_data_hash"] = None

if "sidebar_state" not in st.session_state:
    st.session_state["sidebar_state"] = {
        "data_expanded": True,
        "history_expanded": True,
        "cache_expanded": True
    }

if "selected_question" not in st.session_state:
    st.session_state["selected_question"] = None

if "show_selected_answer" not in st.session_state:
    st.session_state["show_selected_answer"] = False

# 侧边栏
with st.sidebar:
    st.title("💡数据分析智能体")

    # 数据文件类型选择
    st.session_state["sidebar_state"]["data_expanded"] = st.expander(
        "📂数据文件类型", expanded=st.session_state["sidebar_state"]["data_expanded"]
    ).checkbox("展开", value=st.session_state["sidebar_state"]["data_expanded"], key="data_expand_checkbox")

    if st.session_state["sidebar_state"]["data_expanded"]:
        option = st.radio("选择数据类型:", ("Excel", "CSV", "PDF", "DOCX", "TXT"), key="sidebar_data_type")
        file_type = "xlsx" if option == "Excel" else "csv" if option == "CSV" else option.lower()
        data = st.file_uploader(f"上传{option}文件", type=file_type, key="sidebar_uploader")

        if data:
            if option == "Excel":
                wb = openpyxl.load_workbook(data)
                sheet_option = st.radio(label="选择工作表:", options=wb.sheetnames, key="sidebar_sheet")
                st.session_state["df"] = pd.read_excel(data, sheet_name=sheet_option)
            elif option == "CSV":
                st.session_state["df"] = pd.read_csv(data)
            elif option == "PDF":
                text = read_pdf(data)
                st.session_state["df"] = pd.DataFrame({"text": [text]})
            elif option == "DOCX":
                text = read_docx(data)
                st.session_state["df"] = pd.DataFrame({"text": [text]})
            elif option == "TXT":
                text = read_txt(data)
                st.session_state["df"] = pd.DataFrame({"text": [text]})

            # 计算并存储数据哈希
            st.session_state["current_data_hash"] = get_data_hash(st.session_state["df"])

    st.divider()

    # 历史对话记录列表
    st.session_state["sidebar_state"]["history_expanded"] = st.expander(
        "💬历史对话", expanded=st.session_state["sidebar_state"]["history_expanded"]
    ).checkbox("展开", value=st.session_state["sidebar_state"]["history_expanded"], key="history_expand_checkbox")

    if st.session_state["sidebar_state"]["history_expanded"]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"共 {len(st.session_state['history'])//2} 条对话")
        with col2:
            if st.button("清空历史", key="clear_history"):
                st.session_state["history"] = []
                st.success("已清空历史对话")

        if st.session_state["history"]:
            # 显示历史问题列表
            for i, entry in enumerate(st.session_state["history"]):
                if "user" in entry:
                    question = entry["user"]
                    # 创建可点击的历史问题
                    if st.button(f"Q{i//2+1}: {question[:30]}..." if len(question) > 30 else f"Q{i//2+1}: {question}", key=f"hist_{i}"):
                        st.session_state["selected_question"] = question
                        st.session_state["show_selected_answer"] = True
                        # 确保历史记录面板保持展开状态
                        st.session_state["sidebar_state"]["history_expanded"] = True
        else:
            st.info("暂无历史对话")

    st.divider()

    # 缓存状态
    st.session_state["sidebar_state"]["cache_expanded"] = st.expander(
        "💾缓存状态", expanded=st.session_state["sidebar_state"]["cache_expanded"]
    ).checkbox("展开", value=st.session_state["sidebar_state"]["cache_expanded"], key="cache_expand_checkbox")

    if st.session_state["sidebar_state"]["cache_expanded"]:
        cache_size = len(qa_cache)
        st.write(f"当前缓存: {cache_size} 个问答对")

        if cache_size > 0:
            st.write("最近缓存的5个问题：")
            for i, key in enumerate(list(qa_cache.keys())[-5:]):
                answer = qa_cache[key].get("answer", "")
                st.write(f"{i+1}. {answer[:30]}...")

            if st.button("清空缓存", key="clear_cache"):
                qa_cache.clear()
                st.success("已清空缓存")
        else:
            st.info("缓存为空")

# 主界面
st.write("## 💡数据分析智能体")

# 处理从历史记录中选择的问题
if st.session_state["selected_question"] and st.session_state["show_selected_answer"]:
    query = st.session_state["selected_question"]

    with st.spinner("正在处理问题..."):
        # 检查缓存
        if st.session_state["df"] is not None and st.session_state["current_data_hash"]:
            cache_key = f"{st.session_state['current_data_hash']}_{hashlib.sha256(query.encode()).hexdigest()}"
            if cache_key in qa_cache:
                result = qa_cache[cache_key]
                st.success("已从缓存加载回答")
            else:
                st.warning("缓存中未找到此问题的回答，将重新生成")
                result = dataframe_agent(st.session_state["df"], query, st.session_state["memory"])
                # 更新缓存
                qa_cache[cache_key] = result
        else:
            st.error("没有有效的数据可供分析")
            result = {"answer": "请先上传有效的数据文件"}

        # 显示回答
        st.markdown(f"### 问题: {query}")
        if "answer" in result:
            st.markdown(f"**AI**: {result['answer']}")
        if "table" in result:
            st.table(pd.DataFrame(result["table"]["data"],
                                  columns=result["table"]["columns"]))
        if "bar" in result or "line" in result:
            chart_type = "bar" if "bar" in result else "line"
            input_data = result[chart_type]
            st.markdown(f"**AI**: {chart_type} chart")
            create_chart(input_data, chart_type)

    # 添加到历史记录
    if query not in [entry.get("user", "") for entry in st.session_state["history"]]:
        st.session_state["history"].append({"user": query})
        st.session_state["history"].append({"ai": result})

    # 添加返回按钮
    if st.button("返回提问"):
        st.session_state["selected_question"] = None
        st.session_state["show_selected_answer"] = False
        st.experimental_rerun()

else:
    # 正常提问流程
    if st.session_state["df"] is not None:
        with st.expander("📊原始数据"):
            st.dataframe(st.session_state["df"])

        query = st.text_area(
            "请输入你关于以上数据集的问题或数据可视化需求：",
            disabled=st.session_state["df"] is None
        )
        button = st.button("生成回答")

        if button and not query:
            st.warning("请输入问题")
            st.stop()

        if query and button:
            st.session_state["history"].append({"user": query})

            with st.spinner("AI正在思考中，请稍等..."):
                # 检查缓存
                if st.session_state["current_data_hash"]:
                    cache_key = f"{st.session_state['current_data_hash']}_{hashlib.sha256(query.encode()).hexdigest()}"
                    if cache_key in qa_cache:
                        result = qa_cache[cache_key]
                        st.success("已从缓存加载回答")
                    else:
                        result = dataframe_agent(st.session_state["df"], query, st.session_state["memory"])
                        # 更新缓存
                        qa_cache[cache_key] = result
                else:
                    result = dataframe_agent(st.session_state["df"], query, st.session_state["memory"])

                st.session_state["history"].append({"ai": result})

                # 显示回答
                if "answer" in result:
                    placeholder = st.empty()
                    full_answer = ""
                    for chunk in result["answer"]:
                        full_answer += chunk
                        placeholder.write(full_answer)
                if "table" in result:
                    st.table(pd.DataFrame(result["table"]["data"],
                                          columns=result["table"]["columns"]))
                if "bar" in result:
                    create_chart(result["bar"], "bar")
                if "line" in result:
                    create_chart(result["line"], "line")
    else:
        st.info("请在侧边栏上传数据文件")

# 历史对话详情（主界面）
with st.expander("💬历史对话详情"):
    if st.session_state["history"]:
        for i, entry in enumerate(st.session_state["history"]):
            if "user" in entry:
                st.markdown(f"### Q{i//2+1}: {entry['user']}")
            elif "ai" in entry:
                result = entry["ai"]
                if "answer" in result:
                    st.markdown(f"**AI**: {result['answer']}")
                if "table" in result:
                    st.table(pd.DataFrame(result["table"]["data"],
                                          columns=result["table"]["columns"]))
                if "bar" in result or "line" in result:
                    chart_type = "bar" if "bar" in result else "line"
                    input_data = result[chart_type]
                    st.markdown(f"**AI**: {chart_type} chart")
                    create_chart(input_data, chart_type)
                st.divider()
    else:
        st.info("暂无历史对话")