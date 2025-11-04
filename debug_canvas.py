"""Debug canvas functionality"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2

st.title("Canvas Debug Test")

# Simple canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",
    stroke_width=30,
    stroke_color="rgb(255, 255, 255)",
    background_color="rgb(0, 0, 0)",
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="debug_canvas",
    update_streamlit=True,
)

st.write("### Canvas Result Info:")
if canvas_result is not None:
    st.write(f"- canvas_result type: {type(canvas_result)}")
    st.write(f"- Has image_data: {canvas_result.image_data is not None}")
    
    if canvas_result.image_data is not None:
        st.write(f"- Image shape: {canvas_result.image_data.shape}")
        st.write(f"- Image dtype: {canvas_result.image_data.dtype}")
        st.write(f"- Image min/max: {canvas_result.image_data.min()}/{canvas_result.image_data.max()}")
        
        # Check alpha channel
        alpha = canvas_result.image_data[:, :, -1]
        st.write(f"- Alpha channel shape: {alpha.shape}")
        st.write(f"- Alpha min/max: {alpha.min()}/{alpha.max()}")
        st.write(f"- Non-zero pixels: {np.count_nonzero(alpha)}")
        
        if alpha.max() > 0:
            st.success("✓ Drawing detected!")
            
            # Try preprocessing
            coords = cv2.findNonZero(alpha)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                st.write(f"- Bounding box: x={x}, y={y}, w={w}, h={h}")
                st.success("✓ Preprocessing would work!")
            else:
                st.error("✗ cv2.findNonZero returned None")
        else:
            st.info("Canvas is empty - draw something!")
else:
    st.info("Canvas result is None")
