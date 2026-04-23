import requests

def fetch_lassopred_to_fasta():
    print("==========Download Positive Data==========")
    # 1. 替换为你在 F12 中抓包得到的真实 API URL
    api_url = "https://lassopred.accre.vanderbilt.edu/api/data/?page=1&size=4029" 
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        print("正在拉取数据库数据...")
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        # 假设API返回包含肽信息的 JSON 列表 (如结构嵌套，请根据实际层级提取，如 response.json()['data'])
        data_list = response.json()['data'] 
        
        # 2. 将数据写入 FASTA 文件
        output_filename = "./data/raw_positives.fasta"
        valid_count = 0
        
        with open(output_filename, "w", encoding="utf-8") as fasta_file:
            for item in data_list:
                # 3. 替换为真实的 JSON 字段名
                seq_id = item.get("LP_ID", "")           # 替换 'id'
                sequence = item.get("Precursor_Sequence", "").strip()  # 替换 'sequence'
                
                if sequence:
                    # FASTA 格式规范: >标识符 换行 序列
                    fasta_file.write(f">{seq_id}\n{sequence}\n")
                    valid_count += 1
                    
        print(f"爬取并转换完成！共提取 {valid_count} 条序列，已保存至 {output_filename}")

    except requests.exceptions.RequestException as e:
        print(f"请求失败，请检查网络或 API URL: {e}")
    except Exception as e:
        print(f"数据解析出错: {e}")

def fetch_uniprot_negatives(output_fasta, limit=20000):
    """
    通过 UniProt API 获取细菌中长度在 40-100 之间，
    且不属于套索肽的序列作为负样本。
    """
    print("==========Download Negative Data==========")
    print(f"[*] 正在从 UniProt API 抓取负样本序列...")
    
    # UniProt 查询语法:
    # taxonomy_id:2 (Bacteria 细菌)
    # length:[40 TO 100] (长度限制，贴近套索肽前体长度)
    # NOT family:"lasso peptide" (排除已知的套索肽)
    # reviewed:true (只取经过人工审核的 Swiss-Prot 高质量数据)
    
    query = 'taxonomy_id:2 AND length:[40 TO 100] AND reviewed:true NOT family:"lasso peptide"'
    # query = 'organism_id:9606 AND reviewed:true AND cc_disease:*'
    
    # UniProt REST API URL (请求返回 FASTA 格式)
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "format": "fasta",
        "size": limit  # 注意：stream 接口可能会返回所有匹配项，这里限制只是一种说明，实际可能需要分批截断
    }
    
    try:
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()
        
        count = 0
        with open(output_fasta, 'w') as f:
            # 逐行读取流，避免内存撑爆
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    f.write(decoded_line + '\n')
                    if decoded_line.startswith('>'):
                        count += 1
                    # 达到我们需要的数量就停止
                    if count >= limit:
                        break
                        
        print(f"[+] 成功抓取并保存了 {count} 条负样本序列至 {output_fasta}")
        
    except requests.exceptions.RequestException as e:
        print(f"[-] 网络请求失败: {e}")

if __name__ == "__main__":
    fetch_lassopred_to_fasta()
    fetch_uniprot_negatives("./data/raw_negatives.fasta", limit=20000)