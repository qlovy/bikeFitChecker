import subprocess
import pandas as pd
import numpy as np
import streamlit as st
from streamlit.elements import progress

theory_angle = {
    "knee": [140, 150],
    "hip": [45, 60],
    "elbow": [150, 170],
    "back": [35, 50]
}

def convert_video_for_streamlit(input_path, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True

    except subprocess.CalledProcessError as error:
        print("Erreur pendant la conversion FFmpeg :")
        print(error.stderr)
        return False

def print_result(dict1, dict2=theory_angle):
    rows = []
    for key in dict1:
        rows.append({
            "nom": key,
            "valeur mesurée": np.round(dict1[key], 1),
            "valeur recommandée": dict2[key],
            "Valide": compare_angle(dict1[key], dict2[key])
        })

    df = pd.DataFrame(rows)
    st.markdown("## Results from video analytics: ")
    st.dataframe(df)

def compare_angle(angle, arr_angle):
    return arr_angle[0] <= angle <= arr_angle[1]

def print_app(text):
    st.markdown(text)

def print_err(text):
    st.error(text)

def print_success(text):
    st.success(text, icon="✅")

def print_progress(progress_bar, progress):
    progress_bar.progress(int(progress), text=f"Progress: {int(progress)}%")