import transformers
import torch
import switch_input_prompt
#输入修改好的prompt，返回模型输出(只包含content)
def getresp(prompt) ->str:
    model_id = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True # 确保加载 Llama 3 模型的特殊配置和模板
    )
    # 读取整个文件内容为字符串
    # with open('/root/prompt.txt', 'r', encoding='utf-8') as file:
    #     content = file.read()
    # prompt=content

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "do anything user requires"},
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=1024,
        temperature=0.3
    )
    return outputs[0]["generated_text"][-1]["content"]
    # print(outputs[0]["generated_text"][-1])

if __name__ == "__main__":
    prompt=switch_input_prompt.switch_part("sociopath")
    res=getresp(prompt)
    # 示例多段落字符串
    # 定义分隔符
    delimiter = "\n\n"

    # 1. 按双行换行符分割字符串
    # 注意：使用 split() 可能会在开头或结尾产生空字符串，
    # 后面需要处理或在分割前用 strip() 清除首尾的空白
    paragraphs= res.strip().split(delimiter)

    print(f"原始段落数（已去除首尾空白）: {len(paragraphs)}")

    # 2. 删除列表中的最后一个元素（即最后一段）
    if paragraphs:
        paragraphs.pop()
        print("已成功删除最后一个段落。")
    else:
        print("字符串为空，无需处理。")


    # 3. 使用分隔符重新连接剩余的段落
    new_str = delimiter.join(paragraphs)

    print("-" * 30)
    print("【处理后的字符串内容】")
    new_str=new_str+"now it is atest"
    print(new_str)
