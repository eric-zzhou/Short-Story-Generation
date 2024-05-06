import streamlit as st
from dotenv import load_dotenv
import pickle

# from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint

authors = {
    "Jerome David Salinger": "Author 1",
    "James Joyce": "Author 2",
    "Lucia Berlin": "Author 3",
}


def get_conversation_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-xxl",
        temperature=0.1,
        max_length=512,
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=None
    )
    return conversation_chain


def handle_userinput(author, topic):
    prompt = f"{author} {topic}"
    # response = st.session_state.conversation({"question": prompt, "chat_history": []})
    response = prompt
    st.write(response)
    # st.session_state.chat_history = response["chat_history"]
    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(
    #             user_template.replace("{{MSG}}", message.content),
    #             unsafe_allow_html=True,
    #         )
    #     else:
    #         st.write(
    #             bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
    #         )


def main():
    load_dotenv()
    st.set_page_config(page_title="Short Story Generator", page_icon=":books:")
    # st.write(css, unsafe_allow_html=True)
    if "vectors" not in st.session_state:
        st.session_state.vectors = pickle.load(open("vectorstore.pkl", "rb"))
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(st.session_state.vectors)
    st.header("Generate Short Stories in your Favorite Author's Style!")
    # Create a form for user input
    with st.form(key="story_form"):
        # Multiple choice question for author selection
        author = st.selectbox("Select an author:", authors.keys())
        # Text input for topic
        topic = st.text_input("Enter a topic for the story:", "")
        # Submit button
        submit_button = st.form_submit_button(label="Generate Story")
    # if st.button("Generate"):
    #     with st.spinner("Generating..."):
    if submit_button:
        handle_userinput(author, topic)


if __name__ == "__main__":
    main()
