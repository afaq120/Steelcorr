# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:29:54 2025

@author: Afaq Ahmad
"""

import streamlit as st
from PIL import Image

st.title("Corrosion Detection and Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    img_path = "uploaded_image.jpg"
    image.save(img_path)
    img, edge_map, corrosion_mask, corrosion_percentage, predicted_label = predict_corrosion_level(img_path)
    st.write(f"Corrosion Percentage: {corrosion_percentage:.2f}%")
    st.write(f"Predicted Level: {predicted_label[0]}")