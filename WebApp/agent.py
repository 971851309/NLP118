import os
import json
import datetime as dt
from crewai import Task, Agent, Crew, Process
from crewai_tools import SerperDevTool
from models import google_model, local_model

# assign serper tool to a variable
web_search = SerperDevTool()

# customer service information
customer_service_contact = {
    "name": "Customer Service Contact",
    "email": "customerservice@amazon.com",
    "phone": "+1-800-123-4567",
    "address": "123 Amazon Way, Seattle, WA 98101",
}

# each sentiment considerations
# Positive, Negative, Neutral considerations
positive_considerations = [
    "Write a sentence sincerely thanking the customer.",
    "Note that success comes from valued customers like them.",
    "Express hope that the product provides lasting benefits.",
    "Lightly encourage them to shop again with humor."
]

negative_considerations = [
    "Apologize for the issue.",
    "Thank the customer for sharing dissatisfaction and acknowledge feedback.",
    "Elaborate on the issue (e.g., product quality, delivery issues).",
    "Offer a potential solution."
]

neutral_considerations = [
    "Thank the customer for their feedback.",
    "Express appreciation for their loyalty.",
    "Compliment the product they purchased.",
    "Ask what more can be done to improve their experience."
]

# Sentiment-specific expectations
positive_expectations = [
    "A cheerful, empathetic response under 100 words"
]

negative_expectations = [
    "A polite, solution-oriented response under 150 words"
]

neutral_expectations = [
    "A friendly, engaging response under 100 words"
]

# Common response guidelines
common_response_guidelines = [
    "Never say i cannot help you or i am not able to help you",
    "You do not need to address the customer by name",
    "Do not give any greetings or salutations in the response, just start with the response",
    "keep in mind that you need to reply in the same language as the review  ",
    "Do not answer questions that involve offensive language, illegal activities, sensitive information, manipulative intent, or are vague and nonsensical, and politely reject, ask for clarification, or redirect as needed.",
    "Keep it warm, personal, and sweet, like you're chatting with a best friend",
    "If the review is unclear, ask for more details with light humor",
    "Offer solutions or help in a casual, enthusiasm, and approachable way.",
    "Make it easy to read: use short sentences, simple warm words, and a friendly tone",
    "Is is mandatory to make your response easy to read: use short sentences, clear separation between paragraphs, and a friendly tone",
    "use maximum 25 words per sentence",
    "use maximum 3 paragraphs",
    "If needed, includes reference to contemporary fictional character from web searches",
    "Use emojis to add a warm and friendly touch where relevant",
    "End with a positive, open note: 'Let us know if you need anything!'",
]

# general_response_guidelines = [
#     "Never say i cannot help you or i am not able to help you",
#     "keep in mind that you need to reply in the same language as the reply  ",
#     "Keep it warm, personal, and sweet, like you're chatting with a best friend",
#     "If the reply is unclear, ask for more details with light humor",
#     "Offer solutions or help in a casual, enthusiasm, and approachable way.",
#     "Make it easy to read: use short sentences, simple warm words, and a friendly tone",
#     "use maximum 30 words per sentence",
#     "use maximum 3 paragraphs",
#     "Use emojis to add a warm and friendly touch where relevant",
#     "End with a positive, open note: 'Let us know if you need anything!'",
# ]

return_policy = {
    "policy": ["customer can return the product within 30 days of purchase for a full refund.",
               "customer cannot return the product after 30 days of purchase."
               ],
    "conditions": [
        "The product must be in its original condition and packaging.",
        "customer must provide proof of purchase, such as a receipt or order confirmation.",
        "Certain items, such as electronics or personalized products, may have different return policies.",
        "If the product is defective or damaged, you may be eligible for a replacement or repair.",
        "For more information, please visit our return policy page on the Amazon website."]
}

order_inquiry = [
    "If the customer ask about their order status, follow up with order status inquiry or politely ask them to contact customer service for assistance.",
    "If the customer ask to cancel their order, follow up with order cancellation inquiry or politely ask them to contact customer service for assistance.",
    "If the customer ask to talk to agent or customer service, follow up with giving them the customer service contact information."
]

# get current date

current_date = dt.datetime.now().strftime("%Y-%m-%d")

def run_agent(agent_input):
    
    # Defining Agents
    
    # Sentiment Analysis Agent
    
    name = agent_input.get("cust_name", "")
    purch_date = agent_input.get("purch_date", "")
    product = agent_input.get("product", "")
    review = agent_input.get("review", "")
    
    sentiment_agent = Agent(
        role="Sentiment Analysis Agent",
        goal=(
            "Accurately classify text sentiment as Positive, Negative, or Neutral. "
            "Identify the dominant emotion to guide tailored responses. "
            "Deliver clear and reliable sentiment analysis."
        ),
        backstory=(
            "Trained on vast datasets of human expression, you excel at nuanced text understanding. "
            "You transform raw feedback into empathetic, actionable insights."
        ),
        llm=google_model.gemini_2_flash_lite(),
        verbose=False
    )

    # Sentiment Review Agent
    sentiment_review_agent = Agent(
        role="Sentiment Review Agent",
        goal=(
            f"Review the snetiment and emotion analysis from the review: '{review}'. "
            "Ensure sentiment analysis is precise and contextually appropriate. "
            "Confirm sentiment as Positive, Negative, or Neutral. "
            "Capture the dominant emotion for response relevance."
        ),
        backstory=(
            "With years of scrutinizing text analysis, your deep understanding of linguistic nuances "
            "ensures reliable evaluations for meaningful customer interactions."
        ),
        llm=google_model.gemini_2_flash(),
        verbose=True,
        max_iterations=2
    )

    # Response Generation Agent
    response_agent = Agent(
        role="Response Generation Agent",
        goal=(
            "Generate tailored responses for customer reviews based on sentiment and the same language as the review. "
            "Generate empathetic, helpful responses based on sentiment and emotion analysis. "
            "Address concerns appropriately, offering solutions for negative feedback. "
            "Strengthen customer trust and satisfaction."
        ),
        backstory=(
            "Experienced in customer interactions, you craft meaningful responses reflecting emotions like joy or frustration. "
            "You uphold business values through compassionate replies."
        ),
        llm=google_model.gemini_2_flash_lite(),
        max_iterations=2
    )

    # Reviewer Agent
    reviewer_agent = Agent(
        role="Response Reviewer Agent",
        goal=(
            "Generate final tailored responses for customer reviews based on sentiment and the same language as the review. "
            "Review and adjust responses for empathy, politeness, and conciseness. "
            "Ensure responses address concerns with effective solutions. "
            "Deliver polished replies within 200-350 words."
        ),
        backstory=(
            "Your expertise in evaluating customer communications ensures every response meets high standards of empathy and clarity. "
            "You foster trust through thoughtful refinements."
        ),
        llm=google_model.gemini_2_flash(),
        verbose=True,
        tools=[web_search],
        max_iterations=2
    )
    
        # Defining Tasks

    # Sentiment Analysis Task
    sentiment_task = Task(
        description=(
            f"Analyze the sentiment of the text: '{review}'. "
            "Classify as Positive, Negative, or Neutral. "
            "Identify the dominant emotion expressed."
        ),
        expected_output=json.dumps({
            "sentiment": "Positive, Negative, or Neutral",
            "emotion": "e.g., happy, sad, angry, excited"
        }, indent=2),
        agent=sentiment_agent
    )

    # Sentiment Review Task
    sentiment_review_task = Task(
        description=(
            "Review the sentiment analysis for accuracy and contextual relevance. "
            "Validate the emotion to ensure it reflects the text’s tone. "
            "Adjust classifications if discrepancies are found."
        ),
        expected_output=json.dumps({
            "sentiment": "Positive, Negative, or Neutral",
            "emotion": "e.g., happy, sad, angry, excited",
            "review": "Explanation of validation or adjustments"
        }, indent=2),
        agent=sentiment_review_agent,
        context=[sentiment_task]
    )

    # Response Generation Task
    response_task = Task(
        description=(
            f"Customer information provided: name:'{name}', product:'{product}', purchasedate:'{purch_date}'. "
            f" current date: {current_date} "
            f"Order guideline: {order_inquiry} "
            f"Generate a tailored response for the review: '{review}'. "
            "Follow sentiment-specific guidelines:\n"
            f"- Positive: {', '.join(positive_considerations)}\n"
            f"- Negative: {', '.join(negative_considerations)}\n"
            f"- Neutral: {', '.join(neutral_considerations)}\n"
            f"Expectations:\n"
            f"- Positive: {', '.join(positive_expectations)}\n"
            f"- Negative: {', '.join(negative_expectations)}\n"
            f"- Neutral: {', '.join(neutral_expectations)}"
        ),
        expected_output=(
            "A response string with the following characteristics:\n"
            f"- {', '.join(common_response_guidelines)}\n"
            f"you need to decide whether customer eligible for return or not by following the {return_policy['policy']} and {return_policy['conditions']} guideline \n"
            "- Reflects the sentiment (Positive, Negative, or Neutral).\n"
            "- Incorporates empathy and solutions (if negative).\n"
            "- If necessary, search Amazon for product details."
        ),
        agent=response_agent,
        context=[sentiment_task, sentiment_review_task]
    )

    # Reviewer Task
    reviewer_task = Task(
        description=(
            f"Customer information provided: name:'{name}', product:'{product}', purchasedate:'{purch_date}'. "
            f" current date: {current_date} "
            f"Order guideline: {order_inquiry} "
            "Represent the Amazon Customer Service Team to refine the response. "
            f"Review the response for the input: '{review}'. "
            "Ensure empathy, clarity, and alignment with Amazon standards."
        ),
        expected_output=(
            "A polished empathetic response string with the following characteristics:\n"
            f"- {', '.join(common_response_guidelines)}\n"
            f"you need to decide whether customer eligible for return or not by following the {return_policy['policy']} and {return_policy['conditions']} guideline \n"
            "- Addresses sentiment and emotion, within 30-50 words.\n"
            "- For negative sentiment, includes solutions (e.g., new product for faulty items, delivery review for delays).\n"
            "- For positive sentiment, invites repeat shopping with light humor.\n"
            f"- Includes contact details: {customer_service_contact['name']} if the customer needs further assistance or related to negative sentiment.\n"
            f" {customer_service_contact['email']}, {customer_service_contact['phone']}.\n"
            "- If needed, includes a link to product recommendations or solutions from Amazon or web searches."
            " Ends with a warm, positive thank-you note"
        ),
        agent=reviewer_agent,
        context=[sentiment_task, sentiment_review_task, response_task],
        tools=[web_search]
    )
    
    # Crew Setup
    # Crew Execution

    crew = Crew(
        agents=[sentiment_agent, sentiment_review_agent, response_agent, reviewer_agent],
        tasks=[sentiment_task, sentiment_review_task, response_task, reviewer_task],
        verbose=True,
        process=Process.sequential
    )

    crew.kickoff()
    
    # output results
    # will be displayed in streamlit app

    result = {
        "name": name,
        "purchase_date": purch_date,
        "product": product,
        "review": review,
        "sentiment": sentiment_task.output.raw,
        "sentiment_review": sentiment_review_task.output.raw,
        "response": response_task.output.raw,
        "reviewed_response": reviewer_task.output.raw,
        "Used_Model": (
            f"for sentiment analysis: {sentiment_agent.llm.model}, "
            f"for sentiment review: {sentiment_review_agent.llm.model}, "
            f"for response generation: {response_agent.llm.model}, "
            f"for reviewer agent: {reviewer_agent.llm.model}"
        )
    }

    return result

def process_reply(agent_input:dict, customer_reply:str, response_result:str):
    
    name = agent_input.get("cust_name", "")
    purch_date = agent_input.get("purch_date", "")
    product = agent_input.get("product", "")
    review = agent_input.get("review", "")
    
    
    general_response_agent = Agent(
        role="General Response Agent",
        goal=(
            "Your true goal is to provide any solution or suggestion to the customer. "
            "You will need to review the response from the [reviewer agent]"
            "Ensure that the response is concise, clear, solution-oriented, and friendly. "
            "You will need to respond to any customer inquiries or concerns with warm, friendly, and helpful tone. "
        ),
        backstory=(
            "Experienced in customer interactions, you craft meaningful responses reflecting emotions like joy or frustration. "
            "You uphold business values through compassionate replies."
        ),
        llm =google_model.gemini_2_flash_lite(),
    )
    
    # response task
    
    general_response_task = Task(
        description=(
            f"Your main task is to respond to the {customer_reply} after the output from the {response_result} "
            f"Customer information provided: name:'{name}', product:'{product}', purchasedate:'{purch_date}'. "
            "Represent the Amazon Customer Service Team to refine the response. "
            f"Review the response for the input: '{response_result}'. "
            "- dont end the conversation if the solution has not been agreed by the customer"
        ),
        expected_output=(
            f"- {', '.join(general_response_guidelines)}"
            f"make sure you do not repeat the same response as the {response_result}"
            "- Craft the reply to be more short sentence, clear, solution-oriented, and friendly."
            "- If needed, includes a link to product recommendations or solutions from Amazon or web searches."
            "- respond to any customer inquiries or concerns with warm, friendly, and helpful tone"
            "- respond to any customer inquiries or concerns with solutions or suggestions"
            "- ensure that the response is concise, clear, solution-oriented, and friendly"
            "- only end the conversation if the solution has been agreed by the customer"
            "- ends with a warm, positive thank-you note"
        ),
        agent=general_response_agent,
        tools=[web_search]
    )
    
    crew = Crew(
        agents=[general_response_agent],
        tasks=[general_response_task],
        verbose=True,
        process=Process.sequential
    )
    
    crew.kickoff()
    
    result = {
        "general_response": general_response_task.output.raw,
        "Used_Model": f"for general response: {general_response_agent.llm.model}"
    }
    
    return result

    
    
    
    