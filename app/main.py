import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def get_clean_data():
    data = pd.read_csv("dataset/data.csv")

    data = data.drop(["Unnamed: 32", "id"], axis=1)

    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    return data


def add_sidebar():
    st.sidebar.title("Cell Nuclei Details")

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
        )
    # print(input_dict)
    return input_dict


def scaled_values(input_data):
    # scaler = StandardScaler()
    data = get_clean_data()
    data.drop(["diagnosis"], axis=1, inplace=True)

    scaled_values = {}

    for key, value in input_data.items():
        max_val = data[key].max()
        min_val = data[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_values[key] = scaled_value

    return scaled_values


def get_radar_chart(input_data):
    input_data = scaled_values(input_data)
    categories = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[value for key, value in input_data.items() if "_mean" in key],
            theta=categories,
            fill="toself",
            name="Mean Value",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[value for key, value in input_data.items() if "_se" in key],
            theta=categories,
            fill="toself",
            name="Satndard Error",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[value for key, value in input_data.items() if "_worst" in key],
            theta=categories,
            fill="toself",
            name="Worst",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True
    )

    return fig


def add_prediction(input_data):
    model = pickle.load(open("./model/model.pkl", "rb"))
    scaler = pickle.load(open("./model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    # print(prediction)
    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is")
    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malicious")

    st.write(
        "Probability of Benign: %f" % model.predict_proba(input_array_scaled)[0][0]
    )
    st.write(
        "Probability of Malicious: %f" % model.predict_proba(input_array_scaled)[0][1]
    )


def main():
    # page configuration
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":femal-docker:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    input_data = add_sidebar()
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write(
            "Please connect this to your cytology lab help diagnose breast cancer form your tissue.\n"
        )

    col1, col2 = st.columns([4, 1])  # first column is biggest then 1 column

    with col1:
        # https://plotly.com/python/radar-chart/
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_prediction(input_data)


if __name__ == "__main__":
    main()
