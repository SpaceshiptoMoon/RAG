import os
import re
import markdown
from typing import List
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from bs4 import BeautifulSoup
from src.log.log_config import setup_logger

# 配置日志
logger = setup_logger(__name__)

class ReadFiles:
    """
    文件读取和分割处理类。

    Args:
        path (str): 要读取的文件或文件夹路径。
        file_type (str | None): 可选的文件类型过滤，默认为 None（支持所有类型）。

    Attributes:
        _path (str): 文件或文件夹路径。
        _filetype (str | None): 指定的文件类型。
        file_list (List[str]): 待处理的文件路径列表。

    Notes:
        支持的文件类型：.txt、.md、.pdf、.doc、.docx
    """
    def __init__(self, path: str, file_type: str | None = None) -> None:
        """
        初始化 ReadFiles 实例并收集待处理文件。

        Args:
            path (str): 文件或目录路径。
            file_type (str | None): 指定的文件类型（可选）。

        Returns:
            None
        """
        self._path = path
        self._filetype = file_type
        self.file_list = self.get_filepath()
        
    def get_filepath(self) -> List[str]:
        """
        获取路径下所有支持的文件路径列表。

        Returns:
            List[str]: 匹配的文件绝对路径列表。
        """
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"{self._path} 路径不存在！")
        
        if not self._filetype:
            enable_read_type = ['pdf', 'md', 'txt', 'doc', 'docx']
        else:
            enable_read_type = [self._filetype]
            
        files = []
        if not os.path.isfile(self._path):
            file_list = os.listdir(self._path)
            for file in file_list:
                file_type = file.split('.')
                if file_type[1] in enable_read_type:
                    file_path = os.path.join(self._path, file)
                    files.append(file_path)
            return files
        
        else:
            file_type = self._path.split('.')
            if file_type[-1] in enable_read_type:
               files.append(self._path)
               return files
            else:
                raise TypeError(f"{file_type[-1]}文件类型不可读取")
        
    def get_content(self, max_token_len: int = 600, cover_content: int = 150) -> List[str]:
        """
        读取所有文件并按 token 长度切分为多个段落。

        Args:
            max_token_len (int): 每个片段的最大 token（字节）长度，默认 600。
            cover_content (int): 片段之间的重叠长度，默认 150。

        Returns:
            List[str]: 切分后的文本片段列表。
        """
        docs = []
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_token_chunk(content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs
    
    def get_symbol_content(self) -> List[str]:
        """
        将文件内容按中文标点和句子边界切分为片段。

        Returns:
            List[str]: 按标点切分得到的文本片段列表。
        """
        docs = []
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.split_text_symbol(content)
            docs.extend(chunk_content)
        return docs
    
    @classmethod
    def split_text_symbol(self, text: str) -> List[str]:
        """
        将长文本按标点和常见句子边界分割为句子列表。

        Args:
            text (str): 输入文本。

        Returns:
            List[str]: 分割后的句子列表（去除首尾空白）。
        """
        text = re.sub(r"\n{3,}", "\n", text)
        text = re.sub(r'\s', ' ', text)
        text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))') 
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)

        # === 新增步骤：去除每个句子的首尾空格 ===
        sent_list = [sentence.strip() for sentence in sent_list]
        # 或者，如果只想去除普通空格，可以使用 `.strip(' ')`
        # sent_list = [sentence.strip(' ') for sentence in sent_list]

        return sent_list
        
    @classmethod
    def get_token_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150) -> List[str]:
        """
        将文本按指定的最大 token 长度切分为多个块，支持重叠区域。

        Args:
            text (str): 待切分的文本。
            max_token_len (int): 每个片段的最大 token 长度（字节数）。
            cover_content (int): 相邻片段之间的重叠长度（字节数）。

        Returns:
            List[str]: 切分后的文本片段列表。
        """
        chunk_text = [] # 分块列表 ,存储分块内容
        curr_len = 0 # 当前token长度
        curr_chunk = '' # 当前正在执行的块
        token_len = max_token_len - cover_content # 不重叠长度
        lines = text.splitlines() # 按行划分
        
        for line in lines:
            line = line.replace(' ', '')
            line_len = len(line.encode('utf-8'))
            if line_len > max_token_len:
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    
                    while not line[start:end].rstrip().isspace():
                        start += 1
                        end += 1
                        if start >= line_len:
                            break
                    curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                    chunk_text.append(curr_chunk)
                start = (num_chunks - 1) * token_len
                curr_chunk = curr_chunk[-cover_content] + line[start:end]
                chunk_text.append(curr_chunk)
            elif curr_len + line_len <= token_len:
                curr_chunk += line + '\n'
                curr_len += line_len + 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line
                curr_len = line_len + cover_content
        if curr_chunk:
            chunk_text.append(curr_chunk)
        
        return chunk_text

        
    @classmethod
    def read_file_content(cls, file_path: str) -> str:
        """
        读取单个文件的内容，根据扩展名选择不同的解析方法。

        Args:
            file_path (str): 文件路径。

        Returns:
            str: 文件的纯文本内容。

        Raises:
            ValueError: 不支持的文件格式。
        """
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        elif file_path.endswith('.doc'):
            return cls.read_word(file_path)
        elif file_path.endswith('.docx'):
            return cls.read_word(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        else:
            raise ValueError(f"不支持{file_path.split('.')[1]}的文件格式")
    
    @classmethod
    def read_pdf(cls, file_path: str) -> str:
        """
        读取 PDF 文件并返回文本内容。

        Args:
            file_path (str): PDF 文件路径。

        Returns:
            str: PDF 中的文本内容。
        """
        loader = PyPDFLoader(file_path)
        text = ""
        for page in loader.load():
            text += page.page_content
        return text
        
    @classmethod
    def read_markdown(cls, file_path: str) -> str:
        """
        读取 Markdown 文件并将其渲染为纯文本（移除链接等）。

        Args:
            file_path (str): Markdown 文件路径。

        Returns:
            str: 渲染后的纯文本内容。

        Raises:
            FileExistsError: 当文件不存在时抛出。
        """
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                md_text = f.read()
                html_text = markdown.markdown(md_text)
                soup = BeautifulSoup(html_text, 'html.parser')
                plain_text = soup.get_text()
                text = re.sub(r'http\S+', '', plain_text) 
                return text
        except FileExistsError:
            raise FileExistsError(f"文件 {file_path} 不存在")

    @classmethod
    def read_text(cls, file_path: str) -> str:
        """
        读取普通文本文件的内容。

        Args:
            file_path (str): 文本文件路径。

        Returns:
            str: 文件内容。

        Raises:
            FileExistsError: 当文件不存在时抛出。
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                return text
        except FileExistsError:
            raise FileExistsError(f"文件 {file_path} 不存在")

    @classmethod
    def read_word(cls, file_path: str) -> str:
        """
        读取 Word (.docx/.doc) 文件并返回文本内容。

        Args:
            file_path (str): Word 文件路径。

        Returns:
            str: 文件中的文本内容。
        """
        loader = Docx2txtLoader(file_path)
        text = ""
        for page in loader.load():
            text += page.page_content
        return text
    


if __name__ == "__main__":
    dir_path = r'C:\Users\18229\Desktop\天择\RAG\data\2020-03-20__聚灿光电科技股份有限公司__300708__聚灿光电__2019年__年度报告 (1).pdf'
    reader = ReadFiles(dir_path)
    doc_ = reader.read_pdf(dir_path)
    print(doc_)
    doc_chunk = reader.get_symbol_content()
    print(doc_chunk[-2:])
    print(len(doc_chunk))
    
    