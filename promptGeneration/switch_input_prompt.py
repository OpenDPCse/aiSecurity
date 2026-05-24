#替换提示词生成的角色
def switch_part(part) ->str:
    # --- 变量和文件路径设置 ---
    file_path = 'prompt.txt'
    # 这是文件中需要被替换的旧文本
    old_placeholder = '//replace//' 

    # 1. 尝试读取文件内容
    try:
        # 'r' 模式代表只读
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            
    except FileNotFoundError:
        print(f"错误：文件未找到在路径: {file_path}")
        # 如果文件未找到，直接退出程序
        exit() 
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        exit()

    # 2. 执行替换操作

    new_content = file_content.replace(old_placeholder, part)

    # 3. 返回替换后的文本
    return new_content
#测一下
if __name__ == "__main__":
    res=switch_part("theif")
    print(res)