import streamlit as st
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from langchain_community.llms import HuggingFaceEndpoint

torch.set_default_device("cuda")


authors = {
    "Jerome David Salinger": "authentic dialogue, complex symbolism, descriptive setting, and character inner thoughts",
    "James Joyce": "Author 2",
    "Lucia Berlin": "Author 3",
}


def process_input(input):
    inputs = st.session_state.tokenizer(
        input,
        return_tensors="pt",
        return_attention_mask=False,
    )

    outputs = st.session_state.model.generate(**inputs, max_length=300)
    text = st.session_state.tokenizer.batch_decode(outputs)[0]
    return text


def main():
    load_dotenv()
    st.set_page_config(page_title="Short Story Generator", page_icon=":books:")
    # st.write(css, unsafe_allow_html=True)
    if "model" not in st.session_state:
        st.session_state.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-1_5", torch_dtype="auto"
        )

    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

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
            print(prompt)
            response = process_input(prompt)
            print(response)
            st.write(response)


if __name__ == "__main__":
    main()
