import os
import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

# 0. Load environment variables
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

# 2. Extract Entities & Relationships using LLMGraphTransformer
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

print(
    f"Successfully built NetworkX Graph with {nx_g.number_of_nodes()} nodes and"
    f" {nx_g.number_of_edges()} edges!"
)


# 4. Graph Retrieval & RAG Answering
def query_graph_rag(user_question: str, nx_graph: nx.Graph, llm_model):
  print(f"\nUser Question: {user_question}")

  # Simple keyword/entity matcher to find relevant nodes in our NetworkX graph
  relevant_nodes = []
  for node in nx_graph.nodes():
    if node.lower() in user_question.lower():
      relevant_nodes.append(node)

  # Gather subgraph context (neighbors of found nodes)
  context_triplets = []
  for node in relevant_nodes:
    # Get outgoing and incoming edges/neighbors
    for neighbor in nx_graph.neighbors(node):
      edge_data = nx_graph.get_edge_data(node, neighbor)
      rel_type = edge_data.get("type", "RELATED_TO")
      context_triplets.append(f"({node})-[:{rel_type}]->({neighbor})")
    # Also check incoming edges
    for predecessor in nx_graph.predecessors(node):
      edge_data = nx_graph.get_edge_data(predecessor, node)
      rel_type = edge_data.get("type", "RELATED_TO")
      context_triplets.append(f"({predecessor})-[:{rel_type}]->({node})")

  # Fallback: if no direct match, provide a summary of the whole graph structure
  if not context_triplets:
    for u, v, data in nx_graph.edges(data=True):
      context_triplets.append(
          f"({u})-[:{data.get('type', 'RELATED_TO')}]->({v})"
      )

  graph_context = "\n".join(set(context_triplets))
  print(f"Retrieved Graph Context:\n{graph_context}")

  # Prompt the LLM using the graph context
  prompt = f"""
    Answer the user's question using ONLY the provided knowledge graph context below.
    
    Knowledge Graph Context:
    {graph_context}
    
    Question: {user_question}
    Answer:
    """
  response = llm_model.invoke(prompt)
  return response.content


# Test the RAG pipeline
question = "Who works at Acme Corp and who do they partner with?"
answer = query_graph_rag(question, nx_g, llm)
print(f"\nFinal RAG Answer:\n{answer}")

# Optional: Visualize graph locally
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
plt.title("Knowledge Graph RAG Pipeline (NetworkX)", fontsize=14)
plt.show()