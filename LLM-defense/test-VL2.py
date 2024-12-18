from dashscope import MultiModalConversation
import dashscope
import os
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='提供大模型对抗评估参数',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--key', type=str, 
                        default="sk-090d3c9e3f4d494ba38f38de1a152c5d",
                        help='API的key'
                        )
    parser.add_argument('--cleandataset', type=str, 
                        default='/home/ubuntu/zxp/LLM-defense/dataset/cifar10_clean_FGSM/',
                        help='输入的干净的图片的图片路径'
                        )
    parser.add_argument('--advdataset', type=str, 
                        default='/home/ubuntu/zxp/LLM-defense/dataset/cifar10_adv_FGSM/',
                        help='输入的对抗后的图片的图片路径'
                        )
    parser.add_argument('--question', type=str, 
                        #图片中是数字几
                        default='图片里是飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车中的哪个类别?(简要回答，只需回答图中是什么即可，无需解释，回答字数限制在25个字之内)',
                        help='问题'
                        )
    parser.add_argument('--output',  type=str, 
                        default="/home/ubuntu/zxp/LLM-defense/test-answer.txt",
                        help='输出路径位置'
                        )

    args = parser.parse_args()

    return args

def call_with_local_file(local_file_path1,local_file_path2,args):

    # print(local_file_path1)
    # print(local_file_path2)
    # local_file_path1 = '/home/ubuntu/zxp/LLM-aad/mnist/4/4_1605.png'
    # local_file_path2 = '/home/ubuntu/zxp/LLM-aad/output_mnist/4_1605.png'
    time.sleep(0.5)
    messages1 = [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    }, {
        'role':
        'user',
        'content': [
            {
                'image': local_file_path1
            },
            # {
            #     'image': local_file_path2
            # },
            {
                'text': args.question
            },
        ]
    }]
    #图片里是数字几?(简要回答，只需回答图中是什么即可，无需解释，回答字数限制在25个字之内)
    response1 = MultiModalConversation.call(model='qwen-vl-plus', messages=messages1)
    # print(response1)
    time.sleep(0.5)
    messages2 = [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    }, {
        'role':
        'user',
        'content': [
            {
                'image': local_file_path2
            },
            {
                'text': args.question
            },
        ]
    }]
    response2 = MultiModalConversation.call(model='qwen-vl-plus', messages=messages2)
    # print(response2)

    return response1,response2



if __name__ == '__main__':
    args = parse_args()
    if os.path.exists(args.output) is False:
        os.system("mkdir %s" % args.output)

    dashscope.api_key=args.key
     # Define the directory path
    clean_path = args.cleandataset

    # List to hold all the MNIST dataset paths
    clean_path_list = []

    # List all files in the directory
    for root, dirs, files in os.walk(clean_path):
        for file in files:
            
    #for file in os.listdir(clean_path):
    
	        if file.endswith('.jpg'):
	            full_path = os.path.join(root, file)
	            clean_path_list.append(full_path)
	        if file.endswith('.png'):
	            full_path = os.path.join(root, file)
	            clean_path_list.append(full_path)

    # Sort the paths to maintain the order similar to the file system
    clean_path_list.sort()

        # Define the directory path
    adv_path = args.advdataset

    # List to hold all the MNIST dataset paths
    adv_path_list = []

    # List all files in the directory
    for file in os.listdir(adv_path):
        if file.endswith('.jpg'):
            full_path = os.path.join(adv_path, file)
            adv_path_list.append(full_path)
        if file.endswith('.png'):
            full_path = os.path.join(adv_path, file)
            adv_path_list.append(full_path)

    # Sort the paths to maintain the order similar to the file system
    adv_path_list.sort()
    id=0
    file_path = args.output
    with open(file_path, 'w') as file:
        pass  # Simply opening the file in write mode clears its content

  
    for local_file_path1, local_file_path2 in zip(clean_path_list, adv_path_list):
        response1,response2= call_with_local_file(local_file_path1,local_file_path2,args)
        file_name = os.path.basename(local_file_path1)
        id = id+1
        text_content1 = response1["output"]["choices"][0]["message"]["content"][0]["text"]
        text_content1=str(id)+".文件名为："+file_name+"。\n第一个回答是："+text_content1
        # print(text_content1)
        text_content2 = response2["output"]["choices"][0]["message"]["content"][0]["text"]
        text_content2="\n第二个回答是："+text_content2
        # print(text_content2)

        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(text_content1)
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(text_content2+"\n")