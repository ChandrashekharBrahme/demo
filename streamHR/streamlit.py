import streamlit as st
import numpy as np
import cv2
from datetime import datetime
import video

def main():
    st.title("Vital-Sense")

    # Flag to control capturing
    capturing = False

    # Start and stop buttons
    start_button = st.button("Start Capture")
    stop_button = st.button("Stop Capture")

    if start_button:
        capturing = True
        st.session_state.start_time = datetime.now()  # Store start time
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Unable to open webcam")

    if stop_button:
        capturing = False
        st.session_state.end_time = datetime.now()  # Store end time
        if 'cap' in st.session_state:
            cap.release()
            del st.session_state['cap']

    # Display video stream in a rectangular box
    video_placeholder = st.empty()

    # List to store frames
    frames_list = st.session_state.get('frames_list', [])

    # Capture video from webcam and store frames
    while capturing:
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Unable to capture frame")
            break

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Append frame to the list
        frames_list.append(frame)

        # Store frames_list in session state
        st.session_state['frames_list'] = frames_list

        # Display frame
        video_placeholder.image(frame, channels="RGB")

        # Check if 'q' key is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate duration of video captured
    if 'start_time' in st.session_state and 'end_time' in st.session_state:
        duration = (st.session_state.end_time - st.session_state.start_time).total_seconds()
        st.write(f"Duration of video captured: {duration:.2f} seconds")

    # Convert frames list to NumPy array
    np_frames = np.asarray(frames_list)

    st.write("Shape of captured frames:", np_frames.shape)

    # Perform calculations using frames and display results
    st.sidebar.title("Calculations")
    if st.button("Perform Calculations"):
        st.sidebar.write("Performing calculations...")
        hr_pred = video.frame_calculation(np_frames)
        if hr_pred:
            st.sidebar.write(f"HR Predictions: {hr_pred}")
            st.sidebar.write("Calculations completed.")

if __name__ == "__main__":
    main()
