from openai import OpenAI
import faiss
import numpy as np
from typing import List
import os
import pickle
import docx  # 添加docx库
# 在文件顶部添加新的导入
import PyPDF2
from io import BytesIO

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY', 'your_default_api_key'),  # 从环境变量读取API密钥
    base_url=os.getenv('OPENAI_BASE_URL', 'https://xiaoai.plus/v1')  # 从环境变量读取Base URL
)

# 从知识库文件夹加载文档
def estimate_tokens(text: str) -> int:
    """更准确地估算文本的token数量"""
    # 对于中文文本，我们需要更保守的估算
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_words = len([w for w in text.split() if all(c.isascii() for c in w)])
    # 中文字符通常每个字符是一个token，英文单词通常是0.75个token
    return chinese_chars + int(english_words * 0.75)

def split_text(text: str, max_tokens: int = 500) -> List[str]:
    """将文本分割成更小的块，使用更小的块大小"""
    sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_tokens = estimate_tokens(sentence)
        
        # 如果单个句子就超过了最大限制，需要进一步分割
        if sentence_tokens > max_tokens:
            words = sentence.split()
            temp_chunk = []
            temp_size = 0
            for word in words:
                word_tokens = estimate_tokens(word)
                if temp_size + word_tokens > max_tokens:
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                    temp_chunk = [word]
                    temp_size = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_size += word_tokens
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
        # 如果当前块加上这个句子会超过限制，创建新块
        elif current_size + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_size += sentence_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def load_documents() -> List[str]:
    documents = []
    knowledge_base_path = os.path.join(os.path.dirname(__file__), 'knowledge_base')
    
    # 检查知识库文件夹是否存在，如果不存在则创建
    if not os.path.exists(knowledge_base_path):
        os.makedirs(knowledge_base_path)
        print(f"创建知识库文件夹: {knowledge_base_path}")
        print("请在知识库文件夹中添加.txt、.docx或.pdf文件")
        return documents

    # 获取文件列表，支持txt、docx和pdf
    files = [f for f in os.listdir(knowledge_base_path) if f.endswith(('.txt', '.docx', '.pdf'))]
    if not files:
        print("知识库文件夹中没有找到.txt、.docx或.pdf文件")
        print(f"请在此文件夹中添加文本文件、Word文档或PDF文件: {knowledge_base_path}")
        return documents

    for filename in files:
        file_path = os.path.join(knowledge_base_path, filename)
        
        # 处理PDF文档
        if filename.endswith('.pdf'):
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text()
                    if content:
                        chunks = split_text(content)
                        documents.extend(chunks)
                        print(f"成功加载PDF文档: {filename}")
                        print(f"文档被分割为 {len(chunks)} 个块")
                    else:
                        print(f"警告: PDF文档 {filename} 内容为空")
            except Exception as e:
                print(f"读取PDF文档 {filename} 时出错: {str(e)}")
                continue
        
        # 处理Word文档
        elif filename.endswith('.docx'):
            try:
                doc = docx.Document(file_path)
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
                if content:
                    chunks = split_text(content)
                    documents.extend(chunks)
                    print(f"成功加载Word文档: {filename}")
                    print(f"文档被分割为 {len(chunks)} 个块")
                else:
                    print(f"警告: Word文档 {filename} 内容为空")
            except Exception as e:
                print(f"读取Word文档 {filename} 时出错: {str(e)}")
                continue
        
        # 处理txt文件
        elif filename.endswith('.txt'):
            try:
                with open(file_path, 'rb') as f:
                    raw_content = f.read()
                    
                encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'big5', 'ascii']
                for encoding in encodings:
                    try:
                        content = raw_content.decode(encoding).strip()
                        if content:
                            chunks = split_text(content)
                            documents.extend(chunks)
                            print(f"成功加载文件: {filename} (使用 {encoding} 编码)")
                            print(f"文档被分割为 {len(chunks)} 个块")
                            break
                        else:
                            print(f"警告: 文件 {filename} 内容为空")
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"使用 {encoding} 解码文件 {filename} 时出错: {str(e)}")
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {str(e)}")
                continue
    
    if not documents:
        print("警告：没有成功加载任何文档内容")
    else:
        print(f"成功加载了 {len(documents)} 个文档块")
    
    return documents

# 获取文档的 embeddings
# 在文件顶部添加asyncio导入
import asyncio

# 添加同步版本的get_embedding函数
def get_embedding(text: str) -> List[float]:
    """同步获取文本的嵌入向量"""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"获取嵌入向量时出错: {str(e)}")
        # 如果出错，返回一个零向量（text-embedding-3-large的维度是3072）
        return [0.0] * 3072

# 添加异步版本的get_embedding函数
async def get_embedding_async(text: str) -> List[float]:
    """异步获取文本的嵌入向量"""
    try:
        # 使用asyncio.to_thread将同步函数转换为异步
        response = await asyncio.to_thread(
            client.embeddings.create,
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"获取嵌入向量时出错: {str(e)}")
        # 如果出错，返回一个零向量（text-embedding-3-large的维度是3072）
        return [0.0] * 3072

# 修改加载文档并检查是否成功的部分
documents = load_documents()
if not documents:
    print("警告：没有加载到任何文档，将使用空知识库继续运行")
    # 创建一个默认的空文档，以便系统能够继续运行
    documents = ["这是一个默认文档，请上传知识库文件以获得更好的回答。"]
    # 不再调用exit()

# 向量数据库文件路径
vector_db_path = os.path.join(os.path.dirname(__file__), 'vector_db.pkl')
knowledge_base_path = os.path.join(os.path.dirname(__file__), 'knowledge_base')

# 检查知识库是否有变化
def check_knowledge_base_changed() -> bool:
    """检查知识库是否有变化，如果有变化或没有元数据文件则返回True"""
    metadata_path = os.path.join(os.path.dirname(__file__), 'kb_metadata.pkl')
    
    # 获取当前知识库文件列表和修改时间
    current_files = {}
    for filename in os.listdir(knowledge_base_path):
        if filename.endswith(('.txt', '.docx')):
            file_path = os.path.join(knowledge_base_path, filename)
            current_files[filename] = os.path.getmtime(file_path)
    
    # 如果没有元数据文件，创建一个并返回True
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'wb') as f:
            pickle.dump(current_files, f)
        return True
    
    # 加载上次的元数据
    try:
        with open(metadata_path, 'rb') as f:
            previous_files = pickle.load(f)
        
        # 比较文件列表和修改时间
        if set(previous_files.keys()) != set(current_files.keys()):
            # 文件列表不同
            with open(metadata_path, 'wb') as f:
                pickle.dump(current_files, f)
            return True
        
        # 检查文件修改时间
        for filename, mtime in current_files.items():
            if filename not in previous_files or mtime != previous_files[filename]:
                # 文件被修改
                with open(metadata_path, 'wb') as f:
                    pickle.dump(current_files, f)
                return True
        
        # 没有变化
        return False
    
    except Exception as e:
        print(f"检查知识库元数据时出错: {str(e)}")
        # 出错时重建元数据并返回True
        with open(metadata_path, 'wb') as f:
            pickle.dump(current_files, f)
        return True

# 检查是否存在向量数据库文件以及知识库是否有变化
if os.path.exists(vector_db_path) and not check_knowledge_base_changed():
    print("正在加载现有向量数据库...")
    try:
        with open(vector_db_path, 'rb') as f:
            saved_data = pickle.load(f)
            documents = saved_data['documents']
            embeddings = saved_data['embeddings']
            embedding_dim = len(embeddings[0])
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(np.array(embeddings, dtype=np.float32))
        print(f"成功加载向量数据库，包含 {len(documents)} 个文档块")
    except Exception as e:
        print(f"加载向量数据库时出错: {str(e)}")
        print("将重新构建向量数据库...")
        # 如果加载失败，重新构建向量数据库
        embeddings = [get_embedding(doc) for doc in documents]
        embedding_dim = len(embeddings[0])
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(embeddings, dtype=np.float32))
        
        # 保存向量数据库
        with open(vector_db_path, 'wb') as f:
            pickle.dump({'documents': documents, 'embeddings': embeddings}, f)
        print("向量数据库已保存")
else:
    if os.path.exists(vector_db_path):
        print("检测到知识库内容有变化，将重新构建向量数据库...")
    else:
        print("正在构建向量数据库...")
    # 获取文档的 embeddings
    embeddings = [get_embedding(doc) for doc in documents]
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings, dtype=np.float32))
    
    # 保存向量数据库
    with open(vector_db_path, 'wb') as f:
        pickle.dump({'documents': documents, 'embeddings': embeddings}, f)
    print("向量数据库已保存")

# 删除重复的代码
# 构建向量数据库
# embeddings = [get_embedding(doc) for doc in documents]
# embedding_dim = len(embeddings[0])
# index = faiss.IndexFlatL2(embedding_dim)
# index.add(np.array(embeddings, dtype=np.float32))

# 检索相关文档
def get_relevant_docs(query: str, k: int = 5) -> List[str]:  # 将默认值从2改为5
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k)
    return [documents[i] for i in I[0]]

# 添加对话历史记录类
class ConversationHistory:
    def __init__(self, max_history: int = 100):
        self.history = []
        self.max_history = max_history
    
    def add(self, query: str, answer: str):
        self.history.append({"query": query, "answer": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self) -> str:
        if not self.history:
            return ""
        return "\n".join([
            f"用户: {h['query']}\n助手: {h['answer']}"
            for h in self.history
        ])

# 修改生成回答的函数
# 修改生成回答的函数
def generate_answer(query: str, conversation_history: ConversationHistory) -> str:
    try:
        relevant_docs = get_relevant_docs(query)
        
        # 检查是否有有效的相关文档
        has_valid_docs = any(doc != "这是一个默认文档，请上传知识库文件以获得更好的回答。" for doc in relevant_docs)
        
        if not has_valid_docs:
            # 如果没有有效文档，使用特殊提示
            system_prompt = "你是一个助手。当前知识库为空，请提醒用户上传文档。"
            context = "知识库为空，无法提供基于知识库的回答。"
        else:
            # 正常情况
            system_prompt = "你是一个助手。请基于以下背景信息回答问题："
            context = "\n".join(relevant_docs)
        
        chat_history = conversation_history.get_context()
        
        messages = [
            {"role": "system", "content": f"{system_prompt}\n{context}"},
        ]
        
        # 如果有对话历史，添加到消息中
        if chat_history:
            messages.append({"role": "system", "content": f"之前的对话历史：\n{chat_history}"})
        
        messages.append({"role": "user", "content": query})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        answer = response.choices[0].message.content
        
        # 如果知识库为空，添加提示信息
        if not has_valid_docs:
            answer += "\n\n[系统提示：当前知识库为空，请上传文档以获得更准确的回答。]"
            
        conversation_history.add(query, answer)
        return answer
    except Exception as e:
        # 添加错误处理
        error_msg = str(e)
        if "sensitive_words_detected" in error_msg:
            return "抱歉，您的问题或知识库内容可能包含敏感信息，无法生成回答。请尝试修改您的问题或联系管理员检查知识库内容。"
        else:
            return f"生成回答时出错: {error_msg}"

# 修改主程序
if __name__ == "__main__":
    conversation_history = ConversationHistory()
    while True:
        query = input("\n请输入你的问题（输入 'quit' 退出）: ")
        if query.lower() == 'quit':
            break
        answer = generate_answer(query, conversation_history)
        print(f"\nAI回答: {answer}")
