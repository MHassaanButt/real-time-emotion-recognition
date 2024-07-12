import streamlit as st
import tempfile
import os
import base64
from utils.emotions import emotion_process
# Create an output folder if it doesn't exist
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Function to convert video file to base64
def video_to_base64(file_path):
    if file_path is None:
        return None
    try:
        with open(file_path, "rb") as video_file:
            video_bytes = video_file.read()
        return base64.b64encode(video_bytes).decode("utf-8")
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# Streamlit app
st.title("Real-Time Emotion Recognition")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    # Choose output type
    output_type = st.selectbox("Choose output type", ["Video", "Graph"])

    # Display input video
    st.write("Input")
    input_video_base64 = video_to_base64(input_path)
    st.markdown(
        f"""
        <video width="640" height="480" controls>
            <source src="data:video/mp4;base64,{input_video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """,
        unsafe_allow_html=True
    )

    # Process video on button click
    if st.button("Run"):
        # Call emotion_process function
        out_path = emotion_process(input_path, output_option=output_type)

        # Display output based on output_type
        st.write("Output")
        if output_type == "Video":
            if out_path is not None:
                output_video_base64 = video_to_base64(out_path)
                if output_video_base64 is not None:
                    st.markdown(
                        f"""
                        <video width="640" height="480" controls>
                            <source src="data:video/mp4;base64,{output_video_base64}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error("Failed to load output video.")
            else:
                st.error("No output video generated.")
        elif output_type == "Graph":
            if out_path is not None and os.path.exists(out_path):
                st.image(out_path, width=640)  # Display graph image
            else:
                st.error("No graph image found.")

    # Clean up temporary files
    tfile.close()
    os.remove(input_path)
