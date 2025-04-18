import ollama
import time
from typing import Dict, List


class ConversationManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.emotion_prompts = {
            "sad": ["It sounds like you're feeling down. Would you like to talk about what's bothering you?",
                    "I'm here to listen if you need someone to talk to."],
            "angry": ["I can sense some frustration in your words. Let's work through this together.",
                      "Take a deep breath. I'm here to help you find a solution."],
            "anxious": ["It seems like you're feeling anxious. Let's take things one step at a time.",
                        "Try to relax. You're not alone in this."],
            "happy": ["I'm glad to hear you're feeling happy! Let me know how I can assist you further.",
                      "It's great to see you in a positive mood!"],
            "scared": ["It sounds like you're feeling scared. I'm here to support you.",
                       "You're stronger than you think. Let's face this together."],
            "disappointed": ["I understand you're feeling disappointed. Let's see how we can improve things.",
                             "Every setback is a step toward success. Don't lose hope."]
        }

    def start_new_session(self, user_id: str) -> str:
        """Initialize a new session"""
        session_id = f"{user_id}_{time.time()}"
        self.sessions[session_id] = {
            "messages": [],
            "context": {
                "detected_emotion": None,
                "user_sentiment": None,
                "feedback_received": False
            }
        }
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the conversation history"""
        if session_id in self.sessions:
            self.sessions[session_id]["messages"].append({
                "role": role,
                "content": content
            })

    def get_context(self, session_id: str) -> List[Dict]:
        """Get the conversation context"""
        return self.sessions[session_id]["messages"] if session_id in self.sessions else []

    def update_context(self, session_id: str, key: str, value):
        """Update context information for a session"""
        if session_id in self.sessions:
            self.sessions[session_id]["context"][key] = value

    def detect_emotion(self, user_input: str) -> str:
        """Detect emotion from user input"""
        for emotion, prompts in self.emotion_prompts.items():
            for prompt in prompts:
                if prompt.lower() in user_input.lower():
                    return emotion
        return None

    def self_reflect(self, session_id: str, user_input: str) -> bool:
        """
        Model self-reflection: adjust emotional prompts based on user feedback
        """
        if session_id not in self.sessions:
            return False

        if "not helpful" in user_input.lower():
            print("Model is reflecting on its response...")
            self.emotion_prompts.setdefault("confused", []).append(
                "I'm sorry if my response wasn't helpful. Can you clarify what you need?"
            )
            self.update_context(session_id, "feedback_received", True)
            return True
        return False


# Initialize conversation manager
conv_manager = ConversationManager()


# Main program
def main():
    # Start a new session
    user_id = "user_001"  # In real applications, this can be retrieved from a login system
    session_id = conv_manager.start_new_session(user_id)

    print("Welcome to the conversation assistant! Type 'quit' to exit.")

    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            break

        # Add to conversation history
        conv_manager.add_message(session_id, "user", user_input)

        # Detect emotion
        emotion = conv_manager.detect_emotion(user_input)
        if emotion:
            conv_manager.update_context(session_id, "detected_emotion", emotion)

        # Prepare conversation context
        messages = conv_manager.get_context(session_id)

        # If emotion is detected, add an emotional prompt
        if emotion:
            messages.append({
                'role': 'system',
                'content': f"User seems to be feeling {emotion}. Respond appropriately."
            })

        # Get model response
        response = ollama.chat(
            model='llama3.1',
            messages=messages
        )

        # Extract response content
        response_content = response['message']['content']
        print(f"Assistant: {response_content}")

        # Add assistant response to conversation history
        conv_manager.add_message(session_id, "assistant", response_content)

        # Model self-reflection
        if conv_manager.self_reflect(session_id, user_input):
            print("Model has updated its knowledge based on your feedback.")

        # Show prompt
        print("(Type your next message or 'quit' to exit)")


if __name__ == "__main__":
    main()
