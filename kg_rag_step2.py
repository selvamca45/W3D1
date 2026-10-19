from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


import os
import sys

from langchain_openai import ChatOpenAI

# Check if the environment variable OPENAI_API_KEY exists
from dotenv import load_dotenv

load_dotenv(override=True)        # reads OPENAI_API_KEY from the .env file

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Set the OPENAI_API_KEY environment variable before running this script.")
# Default temperature lets ideas vary between runs; temperature=0 makes them repeatable.
# 1. Load document (from Step 1)
sample_text = """
Neo4j is a graph database management system developed by Neo4j, Inc. 
LangChain is a framework designed to simplify the creation of applications using large language models. 
Alice works at Acme Corp in New York. Acme Corp partners with TechGlobal.
"""
with open("sample_knowledge.txt", "w") as f:
    f.write(sample_text.strip())

loader = TextLoader("sample_knowledge.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 2. Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Create an Extraction Prompt
extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are an expert knowledge graph builder. Extract entities and relationships "
        "from the provided text as triplets in the format: Entity1 | Relationship | Entity2. "
        "Put each triplet on a new line. Do not include markdown code blocks or extra text."
    ),
    ("human", "Text:\n{text}")
])

# 4. Create the Chain using LCEL (|)
extraction_chain = extraction_prompt | llm

# 5. Run the chain on our document chunk
print("\nExtracting entities and relationships...")
response = extraction_chain.invoke({"text": docs[0].page_content})
print("\nExtracted Triplets:")
print(response.content)