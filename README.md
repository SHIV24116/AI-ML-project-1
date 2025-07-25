This is my first AI ML project
<br>
# AI-Powered Posture Coach

This project is a real-time AI system that analyzes exercise postures — starting with push-ups — and provides **live feedback** to improve your form. It combines **Computer Vision**, **Machine Learning**, and **Text-to-Speech** to guide users with audio-visual corrections.

---

##  Features

-  Uses **MediaPipe** to detect body landmarks in real-time  
-  Calculates key **joint angles**
-  Classifies **push-up posture** as Good or Bad using a trained **ML model**
-  Gives **live feedback** with color-coded skeleton and text tips
-  Adds **audio guidance** using **Text-to-Speech**
-  **Suggests which joint needs correction** when posture is incorrect

---

##  Dataset

I built and trained my own dataset of realistic posture samples, carefully labeled and balanced across various **Good** and **Bad** postures for different exercises.  

>  Note: The current dataset is small — this is a known limitation, and expanding the dataset is a priority for future development.

---

##  Tech Stack

- **Python**
- **OpenCV + MediaPipe** (for body tracking)
- **scikit-learn** (for posture classification model)
- **pyttsx3** (for audio feedback)
- **NumPy**, **Pandas** (for data processing and analysis)

---

 
## 🤝 Contributions

Feel free to fork, improve, or suggest features!  
Your contributions are welcome.

---

## 📬 Contact

If you'd like to collaborate or have suggestions, feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/shivendra-singh-).
