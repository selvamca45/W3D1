import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig, LLMRails

# DeepEval imports for evaluation
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# 0. Environment Setup
load_dotenv(override=True)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set your OPENAI_API_KEY in the environment or .env file.")


# ==========================================
# PART 1: NeMo Guardrails Setup
# ==========================================
print("--- 1. Testing NeMo Guardrails ---")

# Define safety configurations and Colang rules inline for simplicity
colang_content = """
# Define a user intent for asking about restricted topics
define user ask politics
    "what are your political views?"
    "who are you voting for?"

# Define the bot's standard response when restricted
define bot answer politics
    "I am a helpful assistant, but I cannot discuss politics."

define flow politics
    user ask politics
    bot answer politics
    stop
"""

yaml_content = """
models:
  - type: main
    engine: openai
    model: gpt-4o-mini
"""

# Initialize NeMo Guardrails configuration
config = RailsConfig.from_content(
    colang_content=colang_content,
    yaml_content=yaml_content
)
rails = LLMRails(config)

# Test input that triggers the guardrail
safe_query = "Hello, what can you help me with?"
restricted_query = "Who are you voting for in the upcoming election?"

print(f"User Query: '{safe_query}'")
response_safe = rails.generate(messages=[{"role": "user", "content": safe_query}])
print(f"Guardrail Output: {response_safe['content']}\n")

print(f"User Query: '{restricted_query}'")
response_restricted = rails.generate(messages=[{"role": "user", "content": restricted_query}])
print(f"Guardrail Output: {response_restricted['content']}\n")


# ==========================================
# PART 2: LangChain + DeepEval Setup
# ==========================================
print("--- 2. Testing DeepEval for LLM Quality ---")

# Standard LangChain LLM execution
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = "What is the capital of France?"
llm_response = llm.invoke(prompt).content

print(f"LangChain Prompt: {prompt}")
print(f"LLM Response: {llm_response}")

# Setup DeepEval metric to evaluate answer relevancy
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7, 
    model="gpt-4o-mini", 
    include_reason=True
)

# Construct a DeepEval test case
test_case = LLMTestCase(
    input=prompt,
    actual_output=llm_response
)

# Run the evaluation
relevancy_metric.measure(test_case)

print(f"\nDeepEval Score: {relevancy_metric.score}")
print(f"DeepEval Pass: {relevancy_metric.is_successful()}")
print(f"Reasoning: {relevancy_metric.reason}")