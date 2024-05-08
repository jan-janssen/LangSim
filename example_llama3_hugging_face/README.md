# 1.RAG (Retrieval Augmented Generation) on PDFs(present in data folder)

![Example Output](assets/example.png)

## Model Specification
**Model Used:** LLAMA3 8B  
**Hugging Face Models:** Any model from [Hugging Face](https://huggingface.co/models) can be used.

## Installation and Execution Steps

1. **Install Required Packages**  
   Run `pip install -r requirements.txt` to install the necessary dependencies.

2. **Execute the Script**  
   Use the command: python rag_llm.py --data_dir data/
- Defaults to using the LLAMA3 8B model.
- Defaults to retrieving data from the 'data' folder.
- You will be prompted to enter your Hugging Face API token in the terminal.

3. **Access the UI**  
Upon successful execution, a user interface will be available on port 7860 in the browser, allowing interaction with the LLM and the processed PDFs.

![Document Access](assets/document_access.png)

**Note:** 
- The model download may take some time during the first execution.

- Tried on Nvidia V100 32GB GPU  
- The 8B LLAMA model requires approximately 20GB of GPU memory.

Adapted from : https://www.fahdmirza.com/2024/02/tutorial-to-implement-rag-with-gemma.html

# 2.react_agent_with_query_engine:
Adapted from : https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine/

## youll need a groq account to run this example(however you can use any LLM you have access to) - https://groq.com/
- python react_agent.py --data_dir data/ --groq_api_key <your_key>