import streamlit as st

# from transformers import pipeline

# Load the Hugging Face pipeline for text generation
# text_generator = pipeline("text-generation")

# Define the options for the 'author' multiple choice question
author_options = ["Author 1", "Author 2", "Author 3"]


# Streamlit app starts here
def main():
    st.title("Short Story Generator")

    # Create a form for user input
    with st.form(key="story_form"):
        # Multiple choice question for author selection
        author = st.selectbox("Select an author:", author_options)

        # Text input for topic
        topic = st.text_input("Enter a topic for the story:", "")

        # Submit button
        submit_button = st.form_submit_button(label="Generate Story")

    # When the form is submitted
    if submit_button:
        st.write("Generating story...")

        # Generate a prompt based on user inputs
        prompt = (
            f"Once upon a time, {author} wrote a story about {topic}. The story goes:"
        )

        # Generate a short story using the prompt
        # generated_story = text_generator(prompt, max_length=100)[0]['generated_text']
        generated_story = prompt

        # Display the generated story
        st.write("Generated Story:")
        st.write(generated_story)


# Run the Streamlit app
if __name__ == "__main__":
    main()
