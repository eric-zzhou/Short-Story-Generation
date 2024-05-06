import streamlit as st
from dotenv import load_dotenv
import pickle

# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint

authors = {
    "Jerome David Salinger": "Author 1",
    "James Joyce": "Author 2",
    "Lucia Berlin": "Author 3",
}


def get_model():
    return llm


def main():
    load_dotenv()
    st.set_page_config(page_title="Short Story Generator", page_icon=":books:")
    # st.write(css, unsafe_allow_html=True)
    if "model" not in st.session_state:
        st.session_state.model = HuggingFaceEndpoint(
            repo_id="google/flan-t5-xxl", temperature=0.5
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
        prompt = f"You are a professional creative writer focused on short stories. Write the beginning of a short story in the style of {author} on the following topics: {topic}"
        print(prompt)
        response = st.session_state.model.invoke(prompt)
        print(response)
        st.write(response)


if __name__ == "__main__":
    main()
