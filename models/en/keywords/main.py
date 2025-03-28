import random


keywords = {
    "sad": ["I'm sorry to hear that you're feeling sad. Things will get better.", "It's okay to feel down sometimes. You're not alone."],
    "angry": ["Take a deep breath and try to calm down. Everything will be okay.", "I understand you're upset. Let's work through this together."],
    "anxious": ["Try to relax. Things are not as bad as they seem.", "Take it one step at a time. You've got this!"],
    "happy": ["That's great to hear! Keep up the good mood!", "I'm so glad you're feeling happy!"],
    "scared": ["Don't be afraid. I'm here to support you.", "You're stronger than you think. You can face this!"],
    "disappointed": ["Disappointments are temporary. There are always new opportunities.", "Don't lose hope. Every setback is a step toward success."]
}


def detect_emotion(text):
    for emotion, responses in keywords.items():
        if emotion in text.lower():
            return random.choice(responses)
    return "I understand how you feel. If you'd like, you can tell me more about it."


def main():
    print("Hello! I'm your AI assistant. How are you feeling today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "bye", "goodbye"]:
            print("AI: Goodbye! I hope you have a wonderful day!")
            break
        response = detect_emotion(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()