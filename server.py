from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import json
import datetime
from typing import List, Dict, Any, Optional
import asyncio
from dotenv import load_dotenv  # 添加dotenv模块
from contextlib import asynccontextmanager

# 加载环境变量
load_dotenv()

# 导入RAG检索模块的功能
from RAG检索 import (
    load_documents, 
    get_embedding, 
    get_relevant_docs, 
    generate_answer, 
    ConversationHistory,
    check_knowledge_base_changed,
    get_embedding_async  # 添加这个导入
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时的操作
    load_conversation_histories()
    yield
    # 应用关闭时的操作（如果需要）
    save_conversation_histories()

app = FastAPI(title="RAG检索系统", lifespan=lifespan)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储所有用户的对话历史
conversation_histories = {}

# 加载已保存的对话历史
def load_conversation_histories():
    conversations_dir = os.path.join(os.path.dirname(__file__), 'conversations')
    if not os.path.exists(conversations_dir):
        os.makedirs(conversations_dir)
    
    store_path = os.path.join(conversations_dir, 'conversation_store.json')
    if os.path.exists(store_path):
        try:
            with open(store_path, 'r', encoding='utf-8') as f:
                stored_conversations = json.load(f)
                for conv in stored_conversations:
                    conv_id = conv['id']
                    history = ConversationHistory()
                    history.conversation_id = conv_id
                    history.history = conv['history']
                    conversation_histories[conv_id] = history
        except Exception as e:
            print(f"加载对话历史时出错: {str(e)}")

# 保存对话历史
def save_conversation_histories():
    conversations_dir = os.path.join(os.path.dirname(__file__), 'conversations')
    store_path = os.path.join(conversations_dir, 'conversation_store.json')
    
    try:
        conversations_data = []
        for conv_id, history in conversation_histories.items():
            conversations_data.append({
                "id": conv_id,
                "history": history.history,
                "timestamp": str(datetime.datetime.now())
            })
        
        with open(store_path, 'w', encoding='utf-8') as f:
            json.dump(conversations_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存对话历史时出错: {str(e)}")

# 上传文件到知识库
@app.post("/api/knowledge_base/upload")
async def upload_file(file: UploadFile = File(...)):
    knowledge_base_path = os.path.join(os.path.dirname(__file__), 'knowledge_base')
    
    # 确保知识库文件夹存在
    if not os.path.exists(knowledge_base_path):
        os.makedirs(knowledge_base_path)
    
    # 检查文件类型
    if not file.filename.endswith(('.txt', '.docx', '.pdf')):  # 添加pdf支持
        raise HTTPException(status_code=400, detail="只支持.txt、.docx和.pdf文件")
    
    # 保存文件
    file_path = os.path.join(knowledge_base_path, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"message": f"文件 {file.filename} 上传成功"}

# 获取知识库文件列表
@app.get("/api/knowledge_base")
async def get_knowledge_base():
    knowledge_base_path = os.path.join(os.path.dirname(__file__), 'knowledge_base')
    
    # 确保知识库文件夹存在
    if not os.path.exists(knowledge_base_path):
        os.makedirs(knowledge_base_path)
    
    files = []
    for filename in os.listdir(knowledge_base_path):
        if filename.endswith(('.txt', '.docx', '.pdf')):  # 添加pdf支持
            file_path = os.path.join(knowledge_base_path, filename)
            file_size = os.path.getsize(file_path)
            files.append({
                "name": filename,
                "size": file_size,
                "last_modified": os.path.getmtime(file_path)
            })
    
    return {"files": files}

# 删除知识库文件
@app.delete("/api/knowledge_base/{filename}")
async def delete_file(filename: str):
    knowledge_base_path = os.path.join(os.path.dirname(__file__), 'knowledge_base')
    file_path = os.path.join(knowledge_base_path, filename)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件 {filename} 不存在")
    
    # 删除文件
    os.remove(file_path)
    
    return {"message": f"文件 {filename} 删除成功"}

# 手动更新向量数据库
@app.post("/api/knowledge_base/update")
async def update_vector_db():
    try:
        # 直接调用并等待完成
        result = await reload_documents()
        return {"message": result["message"]}
    except Exception as e:
        return {"message": f"更新失败: {str(e)}"}

# 重新加载文档和更新向量数据库
async def reload_documents():
    # 导入必要的模块
    import faiss
    import numpy as np
    import pickle
    import os
    import asyncio
    from RAG检索 import (
        load_documents, 
        get_embedding, 
        get_relevant_docs, 
        generate_answer, 
        ConversationHistory,
        check_knowledge_base_changed,
        get_embedding_async
    )
    
    # 在app初始化后添加一个启动事件处理器
    @app.on_event("startup")
    async def startup_event():
        """应用启动时执行的操作"""
        # 确保知识库文件夹存在
        knowledge_base_path = os.path.join(os.path.dirname(__file__), 'knowledge_base')
        if not os.path.exists(knowledge_base_path):
            os.makedirs(knowledge_base_path)
            print(f"创建知识库文件夹: {knowledge_base_path}")
        
        # 检查知识库是否为空
        files = [f for f in os.listdir(knowledge_base_path) if f.endswith(('.txt', '.docx', '.pdf'))]
        if not files:
            print("警告：知识库为空，系统将使用有限功能运行")
            print(f"建议在此文件夹中添加文本文件、Word文档或PDF文件: {knowledge_base_path}")
    
    # 标记知识库已更改
    changed = check_knowledge_base_changed()
    
    # 向量数据库文件路径
    vector_db_path = os.path.join(os.path.dirname(__file__), 'vector_db.pkl')
    
    # 强制重新加载文档
    print("正在重新加载文档...")
    documents = load_documents()
    if not documents:
        print("警告：没有加载到任何文档，将创建空向量库")
        # 创建一个默认的空文档，以便系统能够继续运行
        documents = ["这是一个默认文档，请上传知识库文件以获得更好的回答。"]
    
    # 重新构建向量数据库
    print("正在重新构建向量数据库...")
    try:
        # 异步获取文档的embeddings
        print(f"开始异步处理 {len(documents)} 个文档块的嵌入向量...")
        
        # 创建异步任务列表
        embedding_tasks = [get_embedding_async(doc) for doc in documents]
        
        # 使用asyncio.gather并发执行所有任务
        embeddings = await asyncio.gather(*embedding_tasks)
        
        print(f"已完成 {len(embeddings)} 个文档块的嵌入向量处理")
        
        embedding_dim = len(embeddings[0])
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(embeddings, dtype=np.float32))
        
        # 保存向量数据库
        with open(vector_db_path, 'wb') as f:
            pickle.dump({'documents': documents, 'embeddings': embeddings}, f)
        
        # 更新全局变量
        import sys
        module = sys.modules['RAG检索']
        module.documents = documents
        module.embeddings = embeddings
        module.index = index
        
        print(f"向量数据库已更新，包含 {len(documents)} 个文档块")
        return {"message": f"知识库已更新，包含 {len(documents)} 个文档块"}
    except Exception as e:
        print(f"更新向量数据库时出错: {str(e)}")
        return {"message": f"知识库更新失败: {str(e)}"}

# 创建新的对话
@app.post("/api/conversations")
async def create_conversation():
    conversation_id = str(uuid.uuid4())
    conversation_histories[conversation_id] = ConversationHistory()
    conversation_histories[conversation_id].conversation_id = conversation_id
    save_conversation_histories()
    return {"conversation_id": conversation_id}

# 获取对话历史
@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    if conversation_id not in conversation_histories:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    history = conversation_histories[conversation_id].history
    return {"history": history}

# 发送消息并获取回答
@app.post("/api/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, message: Dict[str, str]):
    if conversation_id not in conversation_histories:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    query = message.get("content", "")
    if not query:
        raise HTTPException(status_code=400, detail="消息内容不能为空")
    
    # 生成回答
    conversation_history = conversation_histories[conversation_id]
    answer = generate_answer(query, conversation_history)
    
    # 保存对话历史
    save_conversation_histories()
    
    return {
        "answer": answer,
        "relevant_docs": get_relevant_docs(query)
    }

# 删除对话
@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if conversation_id not in conversation_histories:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    del conversation_histories[conversation_id]
    save_conversation_histories()
    return {"message": "对话已删除"}

# 获取所有对话列表
@app.get("/api/conversations")
async def list_conversations():
    conversations = []
    for conversation_id, history in conversation_histories.items():
        if history.history:
            conversations.append({
                "id": conversation_id,
                "last_message": history.history[-1]["query"] if history.history else "",
                "timestamp": "现在"  # 这里可以添加实际的时间戳
            })
    
    return {"conversations": conversations}

# 挂载静态文件服务
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# 启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)