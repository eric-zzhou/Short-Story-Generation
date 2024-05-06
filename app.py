import streamlit as st
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from langchain_community.llms import HuggingFaceEndpoint

torch.random.manual_seed(0)

authors = {
    "J. D. Salinger": "third person narration, authentic dialogue, complex symbolism, descriptive setting, and character inner thoughts",
    "James Joyce": "first person narration, stream of consciousness, and major climax and resolution",
    "Lucia Berlin": "laid back first person storytelling, marginalized main character, and life experiences",
}


def process_input(input):
    prompt = [{"role": "user", "content": input}]

    inputs = st.session_state.tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_tensors="pt"
    )

    tokens = st.session_state.model.generate(
        inputs.to(st.session_state.model.device),
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )
    output = st.session_state.tokenizer.decode(
        tokens[:, inputs.shape[-1] :][0], skip_special_tokens=False
    )

    return output


def main():
    load_dotenv()
    st.set_page_config(page_title="Short Story Generator", page_icon=":books:")
    # st.write(css, unsafe_allow_html=True)
    if "model" not in st.session_state:
        st.session_state.model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stablelm-2-1_6b-chat",
            device_map="auto",
        )

    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/stablelm-2-1_6b-chat"
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
