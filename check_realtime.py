"""
Quick diagnostic - Check if Streamlit is updating
Open the app and draw something, then check this output
"""
import time
print("Monitoring app activity...")
print("Draw something on the canvas at http://localhost:8501")
print("If you see predictions appearing, it's working!")
print("\nTroubleshooting:")
print("1. Make sure you DRAW on the canvas (drag your mouse)")
print("2. The canvas should be BLACK with WHITE strokes")
print("3. Prediction should appear immediately on the RIGHT side")
print("4. If nothing appears, try:")
print("   - Refresh the browser (F5)")
print("   - Draw a larger digit")
print("   - Check if 'Has drawing: True' appears in debug info")
