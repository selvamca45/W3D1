import os
import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables
load_dotenv(override=True)

if not os.getenv("OPENAI_API_KEY"):
  raise ValueError(
      "Set the OPENAI_API_KEY environment variable before running this script."
  )

# 1. Sample Data Loading & Document Preparation
sample_text = """
Neo4j is a graph database management system developed by Neo4j, Inc. 
LangChain is a framework designed to simplify the creation of applications using large language models. 
Alice works at Acme Corp in New York. Acme Corp partners with TechGlobal.
"""
with open("sample_knowledge.txt", "w") as f:
  f.write(sample_text.strip())

loader = TextLoader("sample_knowledge.txt")
docs = CharacterTextSplitter(chunk_size=300, chunk_overlap=0).split_documents(
    loader.load()
)

# 2. Robust Entity & Relationship Extraction using LangChain's LLMGraphTransformer
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)

print("Extracting nodes and relationships from documents...")
graph_documents = llm_transformer.convert_to_graph_documents(docs)

# 3. Store Graph Structures in NetworkX (In-Memory)
graph = NetworkxEntityGraph()
nx_g = graph._graph

for gdoc in graph_documents:
  for node in gdoc.nodes:
    nx_g.add_node(node.id, type=node.type)
  for rel in gdoc.relationships:
    nx_g.add_edge(rel.source.id, rel.target.id, type=rel.type)

# 4. Inspect Results
print("\nSuccessfully stored graph in NetworkX!")
print("Nodes in Graph:")
for node, data in nx_g.nodes(data=True):
  print(f" - {node} (Type: {data.get('type', 'Unknown')})")

print("\nEdges (Relationships) in Graph:")
for u, v, data in nx_g.edges(data=True):
  print(f" - {u} --[{data.get('type')}]--> {v}")

# Optional: Visualize the graph locally
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(nx_g, seed=42)
nx.draw(
    nx_g,
    pos,
    with_labels=True,
    node_size=3500,
    node_color="lightblue",
    font_size=9,
    font_weight="bold",
    edge_color="gray",
)
edge_labels = nx.get_edge_attributes(nx_g, "type")
nx.draw_networkx_edge_labels(nx_g, pos, edge_labels=edge_labels, font_color="red")
plt.title("Knowledge Graph Pipeline (NetworkX)", fontsize=14)
plt.show()