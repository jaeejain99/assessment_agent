import streamlit as st
from agent import WorldBankAgentSession

st.set_page_config(
    page_title="World Bank Assistant",
    page_icon="🌍",
    layout="wide"
)

st.title("World Bank Intelligence Assistant")

# Initialize agent session in state
if "agent_session" not in st.session_state:
    try:
        st.session_state.agent_session = WorldBankAgentSession()

        st.session_state.init_error = None
    except Exception as e:
        st.session_state.agent_session = None
        st.session_state.init_error = str(e)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Halt input if initialization failed
if st.session_state.init_error:
    st.error(f"Failed to initialize agent: {st.session_state.init_error}")
    st.stop()

# Render chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])

# User input handling
if user_input := st.chat_input("Ask a question about World Bank projects..."):
    # Display user query immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent_session.ask(user_input)
                if response:
                    bot_text = response
                else:
                    bot_text = "No response received. Please check your agent configuration."
            except Exception as e:
                bot_text = f"Error: {str(e)}"
            
            st.markdown(bot_text)

    # Persist interaction
    st.session_state.chat_history.append({"user": user_input, "bot": bot_text})