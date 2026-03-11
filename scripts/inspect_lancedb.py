import os
import lancedb
from config import config
from pathlib import Path

def get_db_info(db_path):
    if not os.path.exists(db_path):
        return None
    
    try:
        db = lancedb.connect(db_path)
        table_names = db.table_names()
        
        tables_info = []
        for name in table_names:
            table = db.open_table(name)
            count = table.count_rows()
            
            # 获取维度信息
            schema = table.schema
            dim = "N/A"
            if "vector" in schema.names:
                vector_field = schema.field("vector")
                # 尝试获取固定大小列表的长度
                try:
                    dim = vector_field.type.list_size
                except:
                    # 如果不是固定大小，尝试读取第一行
                    try:
                        first_row = table.head(1).to_pylist()
                        if first_row and "vector" in first_row[0]:
                            dim = len(first_row[0]["vector"])
                    except:
                        dim = "Unknown"
            
            tables_info.append({
                "name": name,
                "count": count,
                "dimension": dim
            })
        
        return {
            "path": db_path,
            "tables": tables_info
        }
    except Exception as e:
        print(f"Error accessing {db_path}: {e}")
        return None

def main():
    # 收集所有可能的 LanceDB 路径
    paths_to_check = set()
    
    # 1. 从 config 获取默认路径
    paths_to_check.add(config.LANCEDB_PATH)
    paths_to_check.add(config.TEXT_LANCEDB_PATH)
    
    # 2. 检查存储根目录下的其他可能目录
    storage_root = config.STORAGE_ROOT
    if os.path.exists(storage_root):
        for item in os.listdir(storage_root):
            full_path = os.path.join(storage_root, item)
            if os.path.isdir(full_path) and ("lancedb" in item.lower() or "db" in item.lower()):
                paths_to_check.add(full_path)
    
    # 3. 检查 benchmark 目录
    benchmark_root = config.BENCHMARK_DB_ROOT
    if os.path.exists(benchmark_root):
        for benchmark in os.listdir(benchmark_root):
            benchmark_path = os.path.join(benchmark_root, benchmark)
            if os.path.isdir(benchmark_path):
                # 检查 benchmark 下的常见子目录
                for sub in ["lancedb", "textdb"]:
                    sub_path = os.path.join(benchmark_path, sub)
                    if os.path.exists(sub_path):
                        paths_to_check.add(sub_path)

    print("=" * 80)
    print(f"{'LanceDB Path':<50} | {'Table':<15} | {'Rows':<10} | {'Dim':<6}")
    print("-" * 80)
    
    found_any = False
    for path in sorted(list(paths_to_check)):
        info = get_db_info(path)
        if info and info["tables"]:
            found_any = True
            path_display = os.path.relpath(path) if os.path.isabs(path) else path
            # 如果路径太长，截断
            if len(path_display) > 48:
                path_display = "..." + path_display[-45:]
                
            for i, table in enumerate(info["tables"]):
                display_path = path_display if i == 0 else ""
                print(f"{display_path:<50} | {table['name']:<15} | {table['count']:<10} | {table['dimension']:<6}")
    
    if not found_any:
        print("No LanceDB databases found in expected locations.")
    print("=" * 80)

if __name__ == "__main__":
    main()
