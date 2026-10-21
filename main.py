
from rag.loader import load_txt
from rag.splitter import split_text


def main():
    document = load_txt("data/documents/rag_intro.txt")

    chunks = split_text(
        document,
        chunk_size=200,
        overlap=30,
    )

    print(f"原始文档：{document.metadata['file_name']}")
    print(f"原始文本长度：{len(document.page_content)}")
    print(f"切分后 Chunk 数量：{len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"\n--- Chunk {index + 1} ---")
        print(chunk.page_content)
        print("元数据：", chunk.metadata)


if __name__ == "__main__":
    main()





