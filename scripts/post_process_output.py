import re

def post_process_output(output_text, prompt):
    """
    - 移除 prompt 部分
    - 截断在第一个 "输出结束" 之前
    - 返回两个列表：accusations 和 articles
    """
    # 1. 去除 prompt
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):].lstrip()

    # 2. 截断在 "输出结束" 之前
    if "输出结束" in output_text:
        output_text = output_text.split("输出结束")[0]

    # 3. 初始化存储容器
    accusations = []
    articles = []

    # 4. 逐行扫描并提取罪名和法条
    lines = output_text.strip().split("\n")
    for line in lines:
        line = line.strip()

        # 提取罪名部分
        if line.startswith("罪名："):
            # 去掉 "罪名：" 前缀并分割多个罪名
            extracted_accusations = [acc.strip() for acc in line[3:].split("，")]
            accusations.extend(extracted_accusations)

        # 提取法条部分
        elif line.startswith("法条："):
            # 去掉 "法条：" 前缀并分割多个法条
            extracted_articles = line[3:].split("，")
            for article in extracted_articles:
                # 使用正则提取法条编号（仅数字部分）
                match = re.search(r"第(\d+)条", article)
                if match:
                    articles.append(int(match.group(1)))

    # 5. 去重
    accusations = list(set(accusations))
    articles = list(set(articles))

    return accusations, articles
