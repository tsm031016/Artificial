"""
utils - 数据分析智能体使用的工具函数

Author: 骆昊
Version: 0.2
Date: 2025/6/26
"""
import json
from turtle import st

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory

PROMPT_TEMPLATE = """你是一位数据分析助手，你的回应内容取决于用户的请求内容，请按照下面的步骤处理用户请求：
1. 思考阶段 (Thought) ：先分析用户请求类型（文字回答/表格/图表），并验证数据类型是否匹配。
2. 行动阶段 (Action) ：根据分析结果选择以下严格对应的格式。
   - 纯文字回答:
     {"answer": "不超过50个字符的明确答案"}

   - 表格数据：
     {"table":{"columns":["列名1", "列名2", ...], "data":[["第一行值1", "值2", ...], ["第二行值1", "值2", ...]]}}

   - 柱状图
     {"bar":{"columns": ["A", "B", "C", ...], "data":[35, 42, 29, ...]}}

   - 折线图
     {"line":{"columns": ["A", "B", "C", ...], "data": [35, 42, 29, ...]}}

3. 格式校验要求
   - 字符串值必须使用英文双引号
   - 数值类型不得添加引号
   - 确保数组闭合无遗漏
   错误案例：{'columns':['Product', 'Sales'], data:[[A001, 200]]}
   正确案例：{"columns":["product", "sales"], "data":[["A001, 200]]}

注意：响应数据的"output"中不要有换行符、制表符以及其他格式符号。

当前用户请求如下：\n"""

# 全局缓存，用于存储已问过的问题和对应的答案
qa_cache = {}


def dataframe_agent(df, query, memory):
    # 检查缓存中是否已有该问题的答案
    cache_key = hash(pd.util.hash_pandas_object(df).sum()) + hash(query)
    if cache_key in qa_cache:
        print(f"从缓存中获取问题: {query} 的答案")
        return qa_cache[cache_key]

    model = ChatOpenAI(
        base_url='https://api.deepseek.com/',
        api_key=st.secrets["API_KEY"],
        model="deepseek-reasoner",
        temperature=0,
        max_tokens=8192
    )
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=32,
        allow_dangerous_code=True,
        verbose=True,
        memory=memory
    )

    prompt = PROMPT_TEMPLATE + query

    try:
        response = agent.invoke({"input": prompt})
        result = json.loads(response["output"])
        # 将结果存入缓存
        qa_cache[cache_key] = result
        return result
    except Exception as err:
        print(err)
        return {"answer": "暂时无法提供分析结果，请稍后重试！"}