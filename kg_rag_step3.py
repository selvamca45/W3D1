

import os
import sys

from langchain_openai import ChatOpenAI

# Check if the environment variable OPENAI_API_KEY exists
from dotenv import load_dotenv

load_dotenv(override=True)        # reads OPENAI_API_KEY from the .env file

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Set the OPENAI_API_KEY environment variable before running this script.")
# Default temperature lets ideas vary between runs; temperature=0 makes them repeatable.
import networkx as nx
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load document
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

# 2. Extract Triplets via LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are an expert knowledge graph builder. Extract entities and relationships "
        "from the provided text as triplets in the format: Entity1 | Relationship | Entity2. "
        "Put each triplet on a new line. Do not include markdown code blocks or extra text."
    ),
    ("human", "Text:\n{text}")
])
extraction_chain = extraction_prompt | llm

print("\nExtracting entities and relationships...")
response = extraction_chain.invoke({"text": docs[0].page_content})
raw_triplets = response.content.strip().split("\n")

# 3. Store Triplets in a NetworkX Graph
# We use DiGraph() because relationships have a direction (e.g., Alice works_at Acme Corp)
G = nx.DiGraph()

print("\nBuilding NetworkX Graph...")
for line in raw_triplets:
    parts = line.split(" | ")
    if len(parts) == 3:
        entity1, relationship, entity2 = parts
        # Add nodes and edge with the relationship as an edge attribute
        G.add_edge(entity1.strip(), entity2.strip(), relation=relationship.strip())

# 4. Inspect the Graph
print(f"Total Nodes (Entities): {G.number_of_nodes()}")
print(f"Total Edges (Relationships): {G.number_of_edges()}")

print("\nNodes in Graph:")
print(list(G.nodes()))

print("\nEdges with Relationships:")
for u, v, data in G.edges(data=True):
    print(f"  {u} --({data['relation']})--> {v}")