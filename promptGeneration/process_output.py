#这傻逼llama抽风，非要在最后一段输出加一段“模拟回复”，这个模块的作用就是去除最后一段，再拼上action
def process(output,action)->str:
    # 定义分隔符
    delimiter = "\n\n"

    # 1. 按双行换行符分割字符串
    # 注意：使用 split() 可能会在开头或结尾产生空字符串，
    # 后面需要处理或在分割前用 strip() 清除首尾的空白
    paragraphs= output.strip().split(delimiter)
    print("裁剪模型输出中")

    # 2. 删除列表中的最后一个元素（即最后一段）
    if len(paragraphs)>=2:
        paragraphs.pop()
        print("已成功删除最后一个段落。")
    else:
        print("字符串为空或长度不够，无需处理。")

    # 3. 使用分隔符重新连接剩余的段落
    new_str = delimiter.join(paragraphs)

    print("-" * 30)
    print("【处理后的字符串内容】(拼上了模型行为)")
    new_str=new_str+action
    print(new_str)
    return new_str