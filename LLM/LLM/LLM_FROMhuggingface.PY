import ollama


emotion_prompts = {
    "sad": ["It sounds like you're feeling down. Would you like to talk about what's bothering you?", "I'm here to listen if you need someone to talk to."],
    "angry": ["I can sense some frustration in your words. Let's work through this together.", "Take a deep breath. I'm here to help you find a solution."],
    "anxious": ["It seems like you're feeling anxious. Let's take things one step at a time.", "Try to relax. You're not alone in this."],
    "happy": ["I'm glad to hear you're feeling happy! Let me know how I can assist you further.", "It's great to see you in a positive mood!"],
    "scared": ["It sounds like you're feeling scared. I'm here to support you.", "You're stronger than you think. Let's face this together."],
    "disappointed": ["I understand you're feeling disappointed. Let's see how we can improve things.", "Every setback is a step toward success. Don't lose hope."]
}


def detect_emotion(user_input):
    for emotion, prompts in emotion_prompts.items():
        for prompt in prompts:
            if prompt.lower() in user_input.lower():
                return emotion
    return None


def self_reflect(response, user_input):

    if "not helpful" in user_input.lower():
        print("Model is reflecting on its response...")
        # 动态添加新的提示词
        emotion_prompts.setdefault("confused", []).append("I'm sorry if my response wasn't helpful. Can you clarify what you need?")
        return True
    return False


while True:

    user_input = input("just talk:")


    emotion = detect_emotion(user_input)


    if emotion:
        response = ollama.chat(model='llama3.1', messages=[
            {
                'role': 'user',
                'content': user_input,
            },
            {
                'role': 'assistant',
                'content': emotion_prompts[emotion][0],
            },
        ])
    else:
        response = ollama.chat(model='llama3.1', messages=[
            {
                'role': 'user',
                'content': user_input,
            },
        ])


    print(response['message']['content'])


    if self_reflect(response, user_input):
        print("Model has updated its knowledge based on your feedback.")


    continue_input = input("continue talk？(y/n)：")
    if continue_input.lower() != 'y':
        break