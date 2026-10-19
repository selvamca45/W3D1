

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 1. Create a sample text file to work with (or use an existing one)
sample_text = """
Neo4j is a graph database management system developed by Neo4j, Inc. 
LangChain is a framework designed to simplify the creation of applications using large language models. 
Alice works at Acme Corp in New York. Acme Corp partners with TechGlobal.
"""

with open("sample_knowledge.txt", "w") as f:
    f.write(sample_text.strip())

# 2. Load the document using LangChain's TextLoader
loader = TextLoader("sample_knowledge.txt")
documents = loader.load()

# 3. Split the document into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

print(f"Loaded {len(docs)} document chunk(s).")
print("First chunk content:")
print(docs[0].page_content)