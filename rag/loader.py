from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """ 统一的文档数据结构 """
    page_content: str
    metadata: dict

def load_txt(file_path: str | Path) -> Document:
    """读取单个 TXT 文件， 并返回 Document 对象 """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"file not existed in :{path}")

    if not path.is_file():
        raise ValueError(f"指定路径不是文件：{path}")

    if path.suffix.lower() != ".txt":
        raise ValueError(f"暂不支持该文件类型:{path.suffix}")

    content = path.read_text(encoding="utf-8")

    # path.resolve(): resolved absolute path
    return Document(
        page_content=content,
        metadata={
            "source": str(path.resolve()),
            "file_name":path.name,
            "file_type":path.suffix.lower(),
        }
    )

    