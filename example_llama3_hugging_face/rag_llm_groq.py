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

from llama_index.core import PromptTemplate
import gradio as gr
import argparse
from llama_index.llms.groq import Groq



def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", type=str, default="data/")
  parser.add_argument("--groq_api_key", type=str)
  return parser.parse_args()

def main():
  args = parse_args()
  data_dir = args.data_dir
  api_key = args.groq_api_key
  
  embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
  Settings.embed_model = embed_model
  Settings.chunk_size = 512
  system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
  query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

  llm = Groq(model="Llama3-70b-8192", api_key=api_key)

  Settings.llm = llm
  Settings.chunk_size = 512
  documents = SimpleDirectoryReader(data_dir).load_data()#SimpleDirectoryReader, which creates documents out of every file in a given directory. It is built in to LlamaIndex and can read a variety of formats including Markdown, PDFs, Word documents, PowerPoint decks, images, audio and video.
  index = VectorStoreIndex.from_documents(documents)#method which accepts an array of Document objects and will correctly parse and chunk them up. 
  #Under the hood, this splits your Document into Node objects, which are similar to Documents (they contain text and metadata) but have a relationship to their parent Document.
  
  
  def predict(input, history):
    response = query_engine.query(input)
    return str(response)
  
  query_engine = index.as_query_engine()
  gr.ChatInterface(predict).launch(share=True)



if __name__ == "__main__":
  main()