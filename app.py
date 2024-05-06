import streamlit as st
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# from langchain_community.llms import HuggingFaceEndpoint

torch.random.manual_seed(0)

authors = {
    "J. D. Salinger": "third person narration, authentic dialogue, complex symbolism, descriptive setting, and character inner thoughts",
    "James Joyce": "first person narration, stream of consciousness, and major climax and resolution",
    "Lucia Berlin": "laid back first person storytelling, marginalized main character, and life experiences",
}


def process_input(input):
    messages = [
        {"role": "user", "content": input},
    ]
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    output = st.session_state.pipe(messages, **generation_args)
    return output[0]["generated_text"]


def main():
    load_dotenv()
    st.set_page_config(page_title="Short Story Generator", page_icon=":books:")
    # st.write(css, unsafe_allow_html=True)
    if "model" not in st.session_state:
        st.session_state.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )
    if "pipe" not in st.session_state:
        st.session_state.pipe = pipeline(
            "text-generation",
            model=st.session_state.model,
            tokenizer=st.session_state.tokenizer,
        )

    st.header("Generate Short Stories in your Favorite Author's Style!")

    # Create a form for user input
    with st.form(key="story_form"):
        # Multiple choice question for author selection
        author = st.selectbox("Select an author:", authors.keys())
        # Text input for topic
        topic = st.text_input("Enter a topic for the story:", "")
        # Submit button
        submit_button = st.form_submit_button(label="Generate Story")

    if submit_button:
        prompt = f"Write a detailed story in the style of {author} on {topic}. Include a lot of {authors[author]}\n\nStory:\n"
        with st.spinner("Generating story..."):
            st.write(prompt)
            response = process_input(prompt)
            print(response)
            st.write(response)


if __name__ == "__main__":
    main()
