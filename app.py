import streamlit as st
import cv2
import tempfile
import os
import base64
from utils.emotions import process_video
# Create an output folder if it doesn't exist
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Function to process video (placeholder for your real processing logic)
def process_video(input_path, output_path):
    # For demonstration, just copying the input video to output
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()


# Function to convert video file to base64
def video_to_base64(file_path):
    with open(file_path, "rb") as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode("utf-8")


# Streamlit app
st.title("Real-Time Emotion Recognition")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, dir=output_folder)
    tfile.write(uploaded_file.read())
    input_path = tfile.name
    input_name = os.path.basename(input_path)

    # Paths for the output video and graph
    output_video_path = os.path.join(output_folder, f"{input_name}_out.mp4")
    output_graph_path = os.path.join(output_folder, f"{input_name}_out.png")

    # Choose output type
    output_type = st.selectbox("Choose output type", ["Video", "Graph"])

    # Display input video
    col1, col2 = st.columns(2)
    with col1:
        st.write("Input")
        input_video_base64 = video_to_base64(input_path)
        st.markdown(
            f"""
            <video width="320" height="240" controls>
                <source src="data:video/mp4;base64,{input_video_base64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """,
            unsafe_allow_html=True
        )

    # Process video on button click
    if st.button("Run"):
        process_video(input_path, output_video_path)

        # Display output video or graph
        with col2:
            st.write("Output")
            if output_type == "Video":
                output_video_base64 = video_to_base64(output_video_path)
                st.markdown(
                    f"""
                    <video width="320" height="240" controls>
                        <source src="data:video/mp4;base64,{output_video_base64}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    """,
                    unsafe_allow_html=True
                )
            elif output_type == "Graph":
                st.image(output_graph_path, width=320)  # Replace with actual graph output

    # Clean up temporary files
    tfile.close()
    os.remove(input_path)
