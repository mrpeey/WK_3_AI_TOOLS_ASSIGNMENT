"""
Screenshot Helper Script
Automatically creates a screenshots folder and provides reminders
"""

import os
import time
from pathlib import Path

# Create screenshots directory
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

print("="*80)
print("📸 SCREENSHOT CAPTURE HELPER")
print("="*80)
print(f"\nScreenshots will be saved to: {SCREENSHOT_DIR}")
print("\n🎯 REQUIRED SCREENSHOTS (8 total):\n")

screenshots = [
    {
        "num": 1,
        "name": "01_web_app_homepage.png",
        "title": "Web App Homepage",
        "instructions": [
            "Open http://localhost:8501 in browser",
            "Wait for app to fully load",
            "Show full interface with sidebar",
            "Press Windows + Shift + S to capture"
        ]
    },
    {
        "num": 2,
        "name": "02_drawing_digit.png",
        "title": "Drawing a Digit",
        "instructions": [
            "Click on the canvas area",
            "Draw a clear digit (try '7')",
            "Make it large and centered",
            "Capture while digit is visible"
        ]
    },
    {
        "num": 3,
        "name": "03_prediction_result.png",
        "title": "Real-Time Prediction",
        "instructions": [
            "Wait for prediction to appear (instant)",
            "Show both canvas and prediction",
            "Ensure confidence % is visible",
            "Capture the preprocessed 28x28 image too"
        ]
    },
    {
        "num": 4,
        "name": "04_probability_chart.png",
        "title": "Probability Distribution Chart",
        "instructions": [
            "Scroll down to see the full chart",
            "All 10 bars should be visible",
            "Predicted digit should be highlighted green",
            "Capture clear bar chart with percentages"
        ]
    },
    {
        "num": 5,
        "name": "05_top3_predictions.png",
        "title": "Top-3 Predictions",
        "instructions": [
            "Find 'Top 3 Predictions' section",
            "Show medals (🥇🥈🥉)",
            "Display all three with percentages",
            "Capture clearly"
        ]
    },
    {
        "num": 6,
        "name": "06_performance_metrics.png",
        "title": "Model Performance Metrics",
        "instructions": [
            "Scroll to bottom metrics section",
            "Show all 4 metric cards",
            "Test Accuracy, Training Time, Parameters, Dataset",
            "Capture the dashboard"
        ]
    },
    {
        "num": 7,
        "name": "07_different_digit.png",
        "title": "Different Digit Example",
        "instructions": [
            "Click 'Clear Canvas' button",
            "Draw a different digit (try '3' or '8')",
            "Wait for new prediction",
            "Capture to show consistency"
        ]
    },
    {
        "num": 8,
        "name": "08_full_page_overview.png",
        "title": "Full Page Overview",
        "instructions": [
            "Use browser extension 'GoFullPage' OR",
            "Take multiple screenshots and stitch",
            "Show entire app from top to bottom",
            "This gives complete overview"
        ]
    }
]

def display_screenshot_guide():
    """Display step-by-step screenshot guide"""
    for shot in screenshots:
        print(f"\n{'='*80}")
        print(f"📸 Screenshot #{shot['num']}: {shot['title']}")
        print(f"{'='*80}")
        print(f"Filename: {shot['name']}")
        print(f"\n📋 Steps:")
        for i, instruction in enumerate(shot['instructions'], 1):
            print(f"   {i}. {instruction}")
        
        # Wait for user
        print(f"\n⏸️  Press ENTER when you've captured this screenshot...")
        input()
        
        # Check if file exists
        screenshot_path = SCREENSHOT_DIR / shot['name']
        if screenshot_path.exists():
            print(f"   ✅ Found: {shot['name']}")
        else:
            print(f"   ⚠️  Not found. Please save as: {shot['name']}")
        
        print(f"\n✓ Screenshot #{shot['num']} complete!")
        time.sleep(0.5)

def show_summary():
    """Show summary of captured screenshots"""
    print(f"\n\n{'='*80}")
    print("📊 SCREENSHOT SUMMARY")
    print(f"{'='*80}\n")
    
    captured = []
    missing = []
    
    for shot in screenshots:
        screenshot_path = SCREENSHOT_DIR / shot['name']
        if screenshot_path.exists():
            captured.append(shot['name'])
            print(f"✅ {shot['name']}")
        else:
            missing.append(shot['name'])
            print(f"❌ {shot['name']} - MISSING")
    
    print(f"\n{'='*80}")
    print(f"Captured: {len(captured)}/8")
    print(f"Missing: {len(missing)}/8")
    print(f"{'='*80}\n")
    
    if len(captured) == 8:
        print("🎉 SUCCESS! All screenshots captured!")
        print(f"\n📁 Location: {SCREENSHOT_DIR}")
        print("\n✅ Ready for submission!")
    else:
        print("⚠️  Some screenshots are missing.")
        print("\nMissing files:")
        for filename in missing:
            print(f"   - {filename}")
        print("\n💡 Tip: Save screenshots to the correct folder with exact filenames.")

def show_bonus_screenshots():
    """Show optional bonus screenshots"""
    print(f"\n\n{'='*80}")
    print("🌟 BONUS SCREENSHOTS (Optional)")
    print(f"{'='*80}\n")
    
    bonus_shots = [
        {
            "name": "09_code_quality.png",
            "description": "Code in IDE (mnist_streamlit_app.py open in VS Code)"
        },
        {
            "name": "10_ethics_analysis.png",
            "description": "Ethics report showing bias analysis"
        },
        {
            "name": "11_debugging_comparison.png",
            "description": "Side-by-side buggy vs fixed code"
        }
    ]
    
    print("These are optional but impressive:\n")
    for i, shot in enumerate(bonus_shots, 1):
        print(f"{i}. {shot['name']}")
        print(f"   {shot['description']}\n")

def main():
    """Main function"""
    print("\n🚀 Starting screenshot capture guide...\n")
    print("This helper will guide you through capturing all 8 required screenshots.")
    print(f"Screenshots will be saved to: {SCREENSHOT_DIR}\n")
    
    print("📝 Important Notes:")
    print("   • Use Windows + Shift + S to capture (Snipping Tool)")
    print("   • Save each screenshot with the EXACT filename shown")
    print(f"   • Save to: {SCREENSHOT_DIR}")
    print("   • Ensure high resolution (1080p+)")
    print("   • Keep browser window size consistent\n")
    
    choice = input("Ready to start? (y/n): ").lower()
    
    if choice == 'y':
        print("\n🎬 Let's begin!\n")
        time.sleep(1)
        display_screenshot_guide()
        show_summary()
        show_bonus_screenshots()
        
        print("\n" + "="*80)
        print("✅ SCREENSHOT CAPTURE COMPLETE")
        print("="*80)
        print(f"\n📁 All screenshots saved to: {SCREENSHOT_DIR}")
        print("\n📋 Next Steps:")
        print("   1. Review all screenshots for quality")
        print("   2. Retake any unclear images")
        print("   3. Add to submission package")
        print("   4. Optional: Create demo video")
        print("\n🎉 Great work! Your submission is almost ready!")
        
    else:
        print("\n👋 Exiting. Run this script when you're ready!")
        print("\n💡 Quick Tip: Make sure your Streamlit app is running first:")
        print("   streamlit run mnist_streamlit_app.py")

if __name__ == "__main__":
    main()
