import gradio as gr
import argparse
import os

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

SALES_BOTS = {}

# Initialize the vector store
def init_vector_store(file_path_and_name: str, vector_store_dir: str):
    # Check if index.faiss and index.pkl files exist in vector_store_dir
    if os.path.exists(os.path.join(vector_store_dir, "index.faiss")) and os.path.exists(os.path.join(vector_store_dir, "index.pkl")):
        return
    # Read the dataset file
    with open(file_path_and_name) as f:
        file_read = f.read()
    # Convert the dataset file into documents object
    docs = text_splitter.create_documents([file_read])
    # Initialize the vector store for the corresponding dataset file
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    # Persist the vector store data
    db.save_local(vector_store_dir)
    print(vector_store_dir + " has loaded in faiss successfully...")


# Initialize the corresponding sales bot
def initialize_sales_bot(vector_store: str):
    # 初始化矢量存储数据
    init_vector_store("data_set/" + vector_store + "_data.txt", "vector_store/" + vector_store)
    db = FAISS.load_local("vector_store/" + vector_store, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    SALES_BOTS[vector_store] = RetrievalQA.from_chain_type(llm,
                                                           retriever=db.as_retriever(
                                                               search_type="similarity_score_threshold",
                                                               search_kwargs={"score_threshold": 0.8}))
    # Return the retrieval results from the vector store
    SALES_BOTS[vector_store].return_source_documents = True
    return SALES_BOTS[vector_store]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Get command line arguments for enabling chat
def get_arguments_enable_chat():
    # Get from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_chat', type=str2bool, default=False)
    args = parser.parse_args()
    return args.enable_chat


ENABLE_CHAT = get_arguments_enable_chat()


# Chat retrieval
def sales_chat(sales_type, message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    print(f"[SALES_TYPE]{sales_type}")
    print(f"[ENABLE_CHAT]{ENABLE_CHAT}")
    ans = SALES_BOTS[sales_type]({"query": message})
    # If there are retrieval results or chat mode with large model is enabled
    # Return the results combined by RetrievalQA combine_documents_chain
    if ans["source_documents"] or ENABLE_CHAT:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # Otherwise, output a scripted response
    else:
        return "I need to ask my supervisor about this question."


def launch_gradio():
    sales_type = gr.Dropdown(
        choices=[
            ('Real Estate Sales Consultant', "real_estate_sales"),
            ("Electrical Appliance Sales Consultant", "electrical_appliance_sales"),
            ("Home Decoration Sales Consultant", "home_decoration_sales"),
            ("Education Sales Consultant", "education_sales")
        ],
        value="real_estate_sales",
        label="Sales Consultant Type",
        info="Select the type of sales consultant (default: Real Estate Sales Consultant)"
    )

    def wrapper_fn(message, history, sales_type):
        return sales_chat(sales_type, message, history)

    chat_interface = gr.ChatInterface(
        additional_inputs=[sales_type],
        fn=wrapper_fn,
        title="Sales Chatbot",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )
    chat_interface.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # Initialize the real estate sales bot
    initialize_sales_bot("real_estate_sales")
    # Initialize the electrical appliance sales bot
    initialize_sales_bot("electrical_appliance_sales")
    # Initialize the home decoration sales bot
    initialize_sales_bot("home_decoration_sales")
    # Initialize the education sales bot
    initialize_sales_bot("education_sales")
    # Launch the Gradio service
    launch_gradio()