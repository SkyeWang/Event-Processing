import pandas as pd
import os
import re
from pathlib import Path

def clean_and_split_csv():
    """
    数据清洗和分割函数
    1. 去除content内容中"内容提取失败"的数据
    2. 对content进行句子分割，以'。''！''？''......'为分割点
    3. 将所有文件的处理结果合并到一个新的CSV文件中，包含num, content, keyword三列
    """
    
    # 设置data文件夹路径
    data_dir = Path('data')
    output_dir = Path('cleaned_data')
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 处理data文件夹下所有csv文件
    csv_files = list(data_dir.glob('*.csv'))
    
    # 存储所有处理后的数据
    all_split_data = []
    sentence_num = 1
    
    for csv_file in csv_files:
        print(f"正在处理文件: {csv_file.name}")
        
        try:
            # 读取CSV文件，指定编码
            df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"原始数据行数: {len(df)}")
            
            # 检查必要的列是否存在
            required_columns = ['content', 'matched_keywords']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"文件 {csv_file.name} 缺少必要的列: {missing_columns}")
                continue
            
            # 1. 去除content内容中"内容提取失败"的数据
            # 去除content为空的数据
            df_cleaned = df.dropna(subset=['content'])
            # 去除content包含"内容提取失败"的数据
            df_cleaned = df_cleaned[~df_cleaned['content'].str.contains('内容提取失败', na=False)]
            # 去除content为空字符串的数据
            df_cleaned = df_cleaned[df_cleaned['content'].str.strip() != '']
            
            print(f"去除无效数据后行数: {len(df_cleaned)}")
            
            # 2. 对content进行句子分割
            for idx, row in df_cleaned.iterrows():
                content = str(row['content'])
                keyword = str(row['matched_keywords'])
                
                # 使用更简单的分割方法
                # 先按常见的句末标点分割
                sentences = []
                current_sentence = ""
                
                for char in content:
                    current_sentence += char
                    # 如果遇到句末标点，就分割
                    if char in '。！？':
                        # 检查是否是省略号的一部分
                        if len(current_sentence) >= 3 and current_sentence[-3:] == '......':
                            continue
                        else:
                            # 完成当前句子
                            if len(current_sentence.strip()) >= 10:  # 至少10个字符
                                sentences.append(current_sentence.strip())
                            current_sentence = ""
                
                # 处理最后一个句子（如果没有以标点结尾）
                if len(current_sentence.strip()) >= 10:
                    sentences.append(current_sentence.strip())
                
                # 将每个句子作为单独的行添加到结果中
                for sentence in sentences:
                    all_split_data.append({
                        'num': sentence_num,
                        'content': sentence,
                        'keyword': keyword
                    })
                    sentence_num += 1
            
            print(f"当前文件处理完成，累计句子数: {len(all_split_data)}")
            
        except Exception as e:
            print(f"处理文件 {csv_file.name} 时出错: {str(e)}\n")
            continue
    
    # 3. 创建包含所有数据的新DataFrame
    result_df = pd.DataFrame(all_split_data)
    
    print(f"所有文件处理完成，总句子数: {len(result_df)}")
    
    # 保存所有结果到同一个CSV文件
    output_path = output_dir / "all_cleaned_data.csv"
    
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"所有数据已合并保存到: {output_path}")
    print(f"处理完成！\n")

def main():
    """主函数"""
    print("开始数据清洗和分割...")
    clean_and_split_csv()
    print("所有文件处理完成！")

if __name__ == "__main__":
    main()
