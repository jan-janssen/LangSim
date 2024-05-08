#https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine/
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
     Settings,
)
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import argparse
import os

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", type=str, default="data/")
  parser.add_argument("--groq_api_key", type=str)
  return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    api_key = args.groq_api_key
    # load data
    ASE_doc1 = SimpleDirectoryReader(
        input_files=[f"{data_dir}ASE2.pdf"]
    ).load_data()
    ASE_doc2 = SimpleDirectoryReader(
        input_files=[f"{data_dir}ASE.pdf"]
    ).load_data()

    # build index
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    llm = Groq(model="Llama3-70b-8192", api_key=api_key)# add app_key

    Settings.llm = llm
    Settings.chunk_size = 512
    ase1_index = VectorStoreIndex.from_documents(ASE_doc1, show_progress=True)# this uses the embed model internally
    ase2_index = VectorStoreIndex.from_documents(ASE_doc2)

    # persist index
    os.makedirs("storage", exist_ok=True)
    ase1_index.storage_context.persist(persist_dir="storage/ASE_doc1")
    ase2_index.storage_context.persist(persist_dir="storage/ASE_doc2")




    ase1_engine = ase1_index.as_query_engine(similarity_top_k=3)
    ase2_engine = ase2_index.as_query_engine(similarity_top_k=3)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=ase1_engine,
            metadata=ToolMetadata(
                name="ASE1",
                description=(
                    "ASE documentation 1 "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=ase2_engine ,
            metadata=ToolMetadata(
                name="ASE2",
                description=(
                    "ASE documentation 2 "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),

    ]

    ## setup ReAct Agent

    # [Optional] Add Context
    context = """\
    You are a Molecular dynamics simulation expert in python.\
        You will answer questions about writeing python snippets for getting properties of materials\
        as a material  scientist.\
    """



    # llm = OpenAI(model="gpt-3.5-turbo-0613")


    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
        context=context
    )

    response = agent.chat("give python code for: calculate the bulk modulus of AL using ASE with EMT simulation code")
    print(str(response))

if __name__ == "__main__":
    main()