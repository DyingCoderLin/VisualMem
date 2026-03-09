# LanceDB 索引和时间筛选机制

## 1. 字段索引位置

### 自动索引管理

LanceDB **不需要手动创建索引**，索引是自动管理的：

1. **向量索引（Vector Index）**：
   - 位置：自动为 `vector` 字段创建
   - 类型：IVF (Inverted File Index) 或 HNSW (Hierarchical Navigable Small World)
   - 用途：用于高效的向量相似度搜索
   - 创建时机：首次插入数据时自动创建

2. **标量字段索引**：
   - `timestamp`、`frame_id` 等标量字段**没有显式索引**
   - LanceDB 使用 **Pre-filtering** 机制在向量搜索前过滤
   - 这些字段的过滤是在内存中进行的，对于大规模数据可能较慢

### 代码位置

索引相关的代码在 `core/storage/lancedb_storage.py`：

```python
# 表创建时自动推断 schema 并创建向量索引
self.table = self.db.create_table(self.table_name, data=data)
# 或
self.table = self.db.open_table(self.table_name)
```

**注意**：LanceDB 不会为 `timestamp` 或 `frame_id` 创建 B-tree 索引，这些字段的过滤是通过 Pre-filtering 实现的。

---

## 2. 时间筛选机制

### Pre-filtering 工作原理

LanceDB 使用 **Pre-filtering** 在向量搜索**之前**过滤数据：

1. **时间存储格式**：
   ```python
   timestamp.isoformat()  # 例如: "2026-01-26T18:30:37.123456+00:00"
   ```

2. **过滤实现**（`core/storage/lancedb_storage.py:270-339`）：

```python
def search(
    self, 
    query_embedding: List[float], 
    top_k: int = 5,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict]:
    # 构建搜索查询
    search_query = self.table.search(query_embedding)
    
    # 如果有时间范围，使用 Pre-filtering
    if start_time is not None or end_time is not None:
        conditions = []
        if start_time is not None:
            start_iso = start_time.isoformat()
            conditions.append(f"timestamp >= '{start_iso}'")
        if end_time is not None:
            end_iso = end_time.isoformat()
            conditions.append(f"timestamp <= '{end_iso}'")
        
        if conditions:
            where_clause = " AND ".join(conditions)
            search_query = search_query.where(where_clause)  # Pre-filtering
            logger.debug(f"Applying time filter: {where_clause}")
    
    # 执行搜索（先过滤，再向量搜索）
    results = search_query.limit(top_k).to_list()
```

### 工作流程

```
1. 用户查询: query_embedding + start_time + end_time
   ↓
2. Pre-filtering: 使用 where() 过滤 timestamp 范围
   ↓
3. 向量搜索: 在过滤后的数据上进行向量相似度搜索
   ↓
4. 返回 top_k 结果
```

### 性能特点

**优点**：
- ✅ 比先搜索再过滤更高效（减少向量计算量）
- ✅ 比先过滤再搜索更高效（利用向量索引）
- ✅ 原生支持，无需额外配置

**限制**：
- ⚠️ `timestamp` 字段没有 B-tree 索引，过滤是顺序扫描
- ⚠️ 对于超大规模数据（百万级），时间过滤可能较慢
- ⚠️ 字符串比较（ISO 格式）比数值比较稍慢

### 应用/窗口名称筛选

**代码位置**: `core/storage/lancedb_storage.py:270-377` (`search` 方法)

```python
# 支持按应用名称和窗口名称过滤
results = lancedb_storage.search(
    query_embedding=embedding,
    top_k=10,
    start_time=start_time,
    end_time=end_time,
    app_name="微信",        # 精确匹配应用名称
    window_name="聊天窗口"  # 精确匹配窗口名称
)
```

**字段说明**：
- `app_name`: 
  - frame（全屏帧）: 空字符串 `""`
  - sub_frame（窗口帧）: 填写应用名称，如 `"微信"`, `"Chrome"` 等
- `window_name`:
  - frame（全屏帧）: 空字符串 `""`
  - sub_frame（窗口帧）: 填写窗口名称，如 `"聊天窗口"`, `"新标签页"` 等

**筛选示例**：
```python
# 只搜索微信应用的窗口
results = lancedb_storage.search(
    query_embedding=embedding,
    app_name="微信"
)

# 只搜索特定窗口
results = lancedb_storage.search(
    query_embedding=embedding,
    app_name="微信",
    window_name="聊天窗口"
)

# 只搜索全屏帧（app_name 和 window_name 都为空）
results = lancedb_storage.search(
    query_embedding=embedding,
    app_name="",  # 空字符串表示全屏帧
    window_name=""  # 空字符串表示全屏帧
)
```

### 优化建议

如果需要更高效的时间筛选，可以考虑：

1. **添加标量索引**（如果 LanceDB 支持）：
   ```python
   # 注意：LanceDB 可能不支持手动创建标量索引
   # 需要查看最新文档
   ```

2. **使用数值时间戳**：
   ```python
   # 存储 Unix timestamp (int) 而不是 ISO 字符串
   timestamp_int = int(timestamp.timestamp())
   # 但需要修改现有代码
   ```

3. **分区表**（按日期或应用）：
   ```python
   # 按日期创建不同的表
   table_name = f"frames_{date_str}"  # 例如: frames_20260126
   
   # 或按应用创建不同的表
   table_name = f"frames_{app_name}"  # 例如: frames_微信
   ```

---

## 3. 当前实现总结

### 表结构

```python
{
    "frame_id": str,           # 无索引，通过 where() 过滤
    "timestamp": str,          # 无索引，ISO 格式字符串，Pre-filtering
    "image_path": str,         # 无索引
    "vector": List[float],     # ✅ 自动向量索引（用于相似度搜索）
    "ocr_text": str,           # 无索引
    "metadata": str,           # 无索引
    "app_name": str,           # 无索引，标量字段，支持 Pre-filtering（frame 为空，sub_frame 填写）
    "window_name": str         # 无索引，标量字段，支持 Pre-filtering（frame 为空，sub_frame 填写）
}
```

### 查询流程

```python
# 1. 向量搜索 + 时间过滤 + 应用/窗口过滤
results = table.search(query_embedding)
    .where("timestamp >= '2026-01-26T00:00:00' AND timestamp <= '2026-01-26T23:59:59' AND app_name = '微信'")
    .limit(10)
    .to_list()

# 2. 使用 search 方法的参数（推荐）
results = lancedb_storage.search(
    query_embedding=embedding,
    top_k=10,
    start_time=start_time,
    end_time=end_time,
    app_name="微信",        # 可选：按应用名称过滤
    window_name="聊天窗口"  # 可选：按窗口名称过滤
)

# 3. 纯时间查询（不推荐，LanceDB 不是为这个设计的）
# 应该使用 SQLite 进行时间范围查询
```

### 最佳实践

1. **向量搜索 + 时间过滤 + 应用/窗口过滤**：使用 LanceDB 的 `search()` 方法
   ```python
   results = lancedb_storage.search(
       query_embedding=embedding,
       top_k=10,
       start_time=start_time,
       end_time=end_time,
       app_name="微信",        # 可选：按应用过滤
       window_name="聊天窗口"  # 可选：按窗口过滤
   )
   ```

2. **纯时间范围查询**：使用 SQLite（有 B-tree 索引）
   ```python
   # SQLite 有索引: idx_frames_timestamp (B-tree)
   frames = sqlite_storage.get_frames_in_timerange(
       start_time=start_time,
       end_time=end_time,
       limit=1000
   )
   ```

3. **混合查询**（推荐用于复杂场景）：
   - 先用 SQLite 按时间过滤获取 frame_id 列表（利用 B-tree 索引）
   - 再用 LanceDB 对这些 frame_id 进行向量搜索
   ```python
   # 步骤1: SQLite 时间过滤（快速，有索引）
   frame_ids = sqlite_storage.get_frames_in_timerange(start, end)
   frame_id_set = {f["frame_id"] for f in frame_ids}
   
   # 步骤2: LanceDB 向量搜索（在结果中过滤 frame_id）
   results = lancedb_storage.search(query_embedding, top_k=100)
   filtered_results = [r for r in results if r["frame_id"] in frame_id_set]
   ```

---

## 4. SQLite 时间索引（对比）

### SQLite 索引位置

SQLite 有**显式的 B-tree 索引**用于时间查询：

**代码位置**: `core/storage/sqlite_storage.py:250-253`

```python
# 创建时间索引（B-tree，高效）
cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_frames_timestamp 
    ON frames(timestamp)
""")
```

### SQLite 时间查询

**代码位置**: `core/storage/sqlite_storage.py:611-670`

```python
def get_frames_in_timerange(
    self,
    start_time: Union[datetime, str],
    end_time: Union[datetime, str],
    limit: int = 10000
) -> List[Dict]:
    # 使用 B-tree 索引进行高效查询
    cursor.execute("""
        SELECT f.*, o.text as ocr_text
        FROM frames f
        LEFT JOIN ocr_text o ON f.frame_id = o.frame_id
        WHERE f.timestamp >= ? AND f.timestamp < ?  -- 使用 idx_frames_timestamp 索引
        ORDER BY f.timestamp ASC
        LIMIT ?
    """, (start_str, end_str, limit))
```

### 性能对比

| 特性 | LanceDB | SQLite |
|------|---------|--------|
| 时间字段索引 | ❌ 无（Pre-filtering） | ✅ B-tree 索引 |
| 时间查询性能 | 中等（顺序扫描） | 快速（索引查找） |
| 向量搜索 | ✅ 高效（向量索引） | ❌ 不支持 |
| 适用场景 | 向量相似度搜索 + 时间过滤 | 纯时间范围查询 |

---

## 5. 实际使用场景

### 场景 1: RAG 查询（向量搜索 + 时间过滤）

**代码位置**: `gui_backend_server.py:1146-1151`

```python
# 使用 LanceDB Pre-filtering
res = vector_storage.search(
    emb,
    top_k=top_k,
    start_time=start_time,  # Pre-filtering
    end_time=end_time,      # Pre-filtering
)
```

**工作流程**：
1. 向量搜索在时间过滤后的数据上进行
2. 返回 top_k 个最相似的结果

### 场景 2: 时间轴浏览（纯时间查询）

**代码位置**: `gui_backend_server.py:1657-1661`

```python
# 使用 SQLite B-tree 索引
all_frames = sqlite_storage.get_frames_in_timerange(
    start_time=start_time_str,
    end_time=end_time_str,
    limit=100000
)
```

**工作流程**：
1. 使用 `idx_frames_timestamp` 索引快速查找
2. 按时间排序返回所有结果

### 场景 3: 最近帧查询

**代码位置**: `gui_backend_server.py:1462`

```python
# 使用 SQLite 时间索引
frames = sqlite_storage.get_frames_in_timerange(
    start_time=start_time,
    end_time=end_time
)
```

---

## 6. 相关代码位置

- **LanceDB 表创建**: `core/storage/lancedb_storage.py:58-93` (`_setup_table`)
- **LanceDB 时间筛选**: `core/storage/lancedb_storage.py:270-339` (`search` 方法)
- **LanceDB OCR 时间筛选**: `core/storage/lancedb_storage.py:385-446` (`search_ocr` 方法)
- **SQLite 时间索引创建**: `core/storage/sqlite_storage.py:250-253` (B-tree 索引)
- **SQLite 时间查询**: `core/storage/sqlite_storage.py:611-670` (`get_frames_in_timerange` 方法)
- **RAG 查询使用**: `gui_backend_server.py:1146-1151` (`query_rag_with_time`)
- **时间轴浏览使用**: `gui_backend_server.py:1657-1661` (`get_frames_by_date_range`)
