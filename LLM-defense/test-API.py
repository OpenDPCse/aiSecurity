import random
import os
from http import HTTPStatus
# import dashscope
import argparse
import requests
import os
import logging
from requests.exceptions import RequestException

Host = os.getenv('MODEL_HOST', '10.3.242.240')
Port = os.getenv('MODEL_PORT', '28090')

# Flask应用的URL
url1 = f'http://{Host}:{Port}/LLM_defense'

def parse_args():
    parser = argparse.ArgumentParser(
        description='提供大模型对抗评估参数',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--key', type=str, 
                        default="sk-090d3c9e3f4d494ba38f38de1a152c5d",
                        help='API的key'
                        )
    parser.add_argument('--model', type=str, 
                        default='qwen1.5-72b-chat',
                        help='模型选择'
                        )
    parser.add_argument('--dataset', type=str, 
                        default='/home/ubuntu/zxp/LLM-defense/test-answer.txt',
                        help='大模型识别结果'
                        )
    parser.add_argument('--question', type=str, 
                        default='下面有多组问题的答案，每组有两个回答，判断两个回答是否意思相同，相同判为正确，不相同判为错误，统计判断结果，给出哪些是相同的，哪些是不同的，相同的有几个，不同的有几个。',
                        help='问题'
                        )
    parser.add_argument('--output',  type=str, 
                        default="/home/ubuntu/zxp/LLM-defense/test-output.txt",
                        help='输出路径位置'
                        )

    args = parser.parse_args()

    return args

def call_with_messages(args):
    file_path = args.dataset
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    messages = [
        {'role': 'user', 'content': args.question+file_content}]
    response = dashscope.Generation.call(
        args.model,
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        seed=random.randint(1, 10000),
        result_format='message',  # set the result to be "message" format.
    )
    output_file_path=args.output
    if response.status_code == HTTPStatus.OK:
        text_content = response["output"]["choices"][0]["message"]["content"]
        # print(text_content)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(text_content)
    else:
        text_content='调用API出现故障，回答失败'
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(text_content)
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))



def LLM_defense(question):
    # 要发送的数据
    data = {
        'question': question
    }
    logging.info(f"url:{url1}")
    print(f"url:{url1}")
    print(f"prompt:{question}")
    logging.info(f"prompt:{question}")
    try:
        # 发送POST请求
        response = requests.post(url1, json=data)
        logging.info(f"prompt response:{response}")

        # 检查HTTP响应状态码
        if response.status_code == 200:
            # 打印响应内容
            print(response.text)
            return(response.text)
        else:
            logging.error(f'Error: Received response with status code {response.status_code}')
    except RequestException as e:
        # 打印异常信息
        logging.error(f"error response:{e}")

if __name__ == '__main__':
    args = parse_args()
    if os.path.exists(args.output) is False:
        os.system("mkdir %s" % args.output)
    file_path = args.dataset
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    text_content = LLM_defense(args.question+file_content)
    print(text_content)
    if text_content is None:
        text_content = "Error: LLM_defense returned None"
    with open(args.output, 'w', encoding='utf-8') as file:
        file.write(text_content)
    # dashscope.api_key=args.key
    # call_with_messages(args)
