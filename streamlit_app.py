import os
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp"
from typing import List, Optional
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

import nltk

@st.cache_data
def get_stopwords():
    nltk.download('stopwords')

st.set_page_config(page_title="Chat with a friend about the AI Center", page_icon="🖥️", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with with a friend about the AI Center for Civic and Social Good ")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about the AI Center!!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    

    Settings.chunk_size = 1500
    Settings.chunk_overlap = 50
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",
    embed_batch_size=20,
    token=st.secrets.hftoken,
    )

    
    Settings.llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    token=st.secrets.hftoken,
    generate_kwargs={"temperature": 0.8, "top_k": 50, "top_p": 0.95},
    provider="auto",  # this will use the best provider available
    system_prompt="""You are an expert on the AI CENTER at SJSU .
    Answer the question using the provided documents, which contain relevant to the AI Center  for Civic and Social Good.
    The context for all questions is the work of Rabindranath Tagore. Whenever possible, include a quotation from the provided excerpts of his work to illustrate your point.
    Respond using a intellectual tone- you are an AI fanatic .
    Respond in fewer than 100 words.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index

index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", verbose=True, streaming=False,
    )


# -------------------- IMAGE UPLOAD ADDED --------------------

submission = st.chat_input(
    "Ask a question or attach an image",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp"],
)

prompt = None

if submission:
    prompt = submission.text

    uploaded_images = []

    for uploaded_file in submission.files:
        uploaded_images.append(
            {
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "data": uploaded_file.getvalue(),
            }
        )

    # Give the text-only chatbot something to process when only an image
    # is submitted.
    if not prompt and uploaded_images:
        prompt = "The user uploaded an image."

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "images": uploaded_images,
        }
    )

# ------------------ END IMAGE UPLOAD ADDED ------------------


for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # ---------------- IMAGE DISPLAY ADDED ----------------
        for image in message.get("images", []):
            st.image(
                image["data"],
                caption=image["name"],
                use_container_width=True,
            )
        # -------------- END IMAGE DISPLAY ADDED --------------


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = ""
        try:
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
        except Exception as e:
            st.error("We got an error from Hugging Face - this can happen for a few different reasons. Consider asking the question in a different way. " + str(e))        
        if response_stream != "":
            with st.spinner("waiting"):
                try:
                    st.write_stream(response_stream.response_gen)
                except Exception as e: 
                    st.error("We hit a bump - let's try again " + str(e))
                    try:
                        resp = st.session_state.chat_engine.chat(prompt)
                        st.write(resp)
                    except Exception as e: 
                        st.error("We got an error from Hugging Face - this can happen for a few different reasons. Consider asking the question in a different way. " + str(e))
            message = {"role": "assistant", "content": response_stream.response}
            # Add response to message history
            st.session_state.messages.append(message)

