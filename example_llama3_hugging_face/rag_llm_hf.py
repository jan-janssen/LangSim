# Adapted from:
#https://www.fahdmirza.com/2024/02/tutorial-to-implement-rag-with-gemma.html

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import torch
from huggingface_hub import login
login()# add access token from profile
from llama_index.core import PromptTemplate
import gradio as gr
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", type=str, default="data/")
  return parser.parse_args()

def main():
  
  args = parse_args()
  data_dir = args.data_dir
  
  embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
  Settings.embed_model = embed_model
  Settings.chunk_size = 512
  system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
  query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

  llm = HuggingFaceLLM(
      context_window=8192,
      max_new_tokens=256,
      generate_kwargs={"temperature": 0.7, "do_sample": False},
      system_prompt=system_prompt,
      query_wrapper_prompt=query_wrapper_prompt,
      tokenizer_name="meta-llama/Meta-Llama-3-8B",
      model_name="meta-llama/Meta-Llama-3-8B",
      device_map="auto",
      tokenizer_kwargs={"max_length": 4096},
      model_kwargs={"torch_dtype": torch.float16}
      )

  Settings.llm = llm
  Settings.chunk_size = 512
  documents = SimpleDirectoryReader(data_dir).load_data()
  index = VectorStoreIndex.from_documents(documents)
  
  def predict(input, history):
    response = query_engine.query(input)
    return str(response)
  
  query_engine = index.as_query_engine()
  gr.ChatInterface(predict).launch(share=True)



if __name__ == "__main__":
  main()