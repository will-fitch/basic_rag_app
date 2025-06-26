import streamlit as st
from app_state import AppState

def initialize_session_state():
    """Initialize session state with application state."""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "I am here to help YOU!"}]

def display_sources(sources):
    """Display source citations in a consistent format."""
    if sources:
        with st.popover("**Sources**", use_container_width=True):
            for i, source in enumerate(sources):
                with st.expander(f"{i+1} - {source['filename']}"):
                    st.markdown(source['content'])

def display_message(message):
    """Display a single message with its sources."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            display_sources(message["sources"])

def handle_file_upload():
    """Handle file upload when a new file is selected."""
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        uploaded_file = st.session_state.uploaded_file
        file_name = uploaded_file.name
        file_content = uploaded_file.getbuffer()
        try:
            with st.spinner("Processing file..."):
                num_chunks = st.session_state.app_state.upload_file(file_name, file_content)
                st.sidebar.success(f"✅ Added {num_chunks} chunks from '{file_name}'")
        except Exception as e:
            st.sidebar.error(f"❌ Error processing file: {str(e)}")

def render_file_upload():
    """Render the file upload section in the sidebar."""
    st.sidebar.header("📁 Upload Documents")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['txt', 'pdf'],
        help="Upload a .txt or .pdf file to add to the knowledge base",
        key="uploaded_file",
        on_change=handle_file_upload
    )
    render_uploaded_files()

def render_uploaded_files():
    """List previously uploaded files and allow downloading or deleting them."""
    st.sidebar.markdown("### Documents")
    files = st.session_state.app_state.list_uploaded_files()
    for filename in files:
        file_content = st.session_state.app_state.load_file_content(filename)
        col1, col2, col3 = st.sidebar.columns([3, 1, 1])
        with col1:
            st.markdown(f"**{filename}**")
        with col2:
            st.download_button(
                label="🡇",
                data=file_content,
                file_name=filename,
                mime="text/plain",
                key=f"download_{filename}"
            )
        with col3:
            if st.button("🗑️", key=f"delete_{filename}"):
                st.session_state.app_state.delete_file(filename)
                st.rerun()

def render_chat_interface():
    """Render the chat interface."""
    # Display all messages
    for message in st.session_state.messages:
        display_message(message)

def handle_chat_input():
    """Handle chat input and generate responses."""
    if prompt := st.chat_input("Type here..."):
        # Add user message
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        display_message(user_message)
        
        # Generate and add assistant response
        try:
            with st.spinner("Thinking..."):
                rag_response = st.session_state.app_state.query_model(prompt)
            
            # Create assistant message with sources
            assistant_message = {
                "role": "assistant", 
                "content": rag_response.content,
                "sources": rag_response.sources
            }
            st.session_state.messages.append(assistant_message)
            display_message(assistant_message)
                            
        except Exception as e:
            error_message = {
                "role": "assistant", 
                "content": f"Sorry, I encountered an error: {str(e)}"
            }
            st.session_state.messages.append(error_message)
            display_message(error_message)

def main():
    """Main application function."""
    st.title("RAG Chatbot")
    initialize_session_state()
    render_file_upload()
    render_chat_interface()
    handle_chat_input()

if __name__ == "__main__":
    main()
