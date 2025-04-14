import os
import json
from crewai import Task, Agent, Crew, LLM, Process
from dotenv import load_dotenv, find_dotenv
from crewai_tools import SerperDevTool


# Load environment variables from .env file
dotenv_path = find_dotenv()
load_dotenv()

# configure the tools

web_search = SerperDevTool()

# Configure the language models.

# google
file_path = ''

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert the credentials to a JSON string
vertex_credentials_json = json.dumps(vertex_credentials)

gemini = LLM(
    model="gemini/gemini-2.0-flash-lite",
    temperature=0.7,
    vertex_credentials=vertex_credentials_json
)

gemini2 = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    vertex_credentials=vertex_credentials_json
)

# local model
llama = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434",
    temperature=0.7,
)

gemma = LLM(
    model="ollama/gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.7,
)

mistral = LLM(
    model="ollama/mistral:latest",
    base_url="http://localhost:11434",
    temperature=0.7,
)

cogito = LLM(
    model="ollama/cogito:latest",
    base_url="http://localhost:11434",
    temperature=0.7,
)

exaone = LLM(
    model="ollama/exaone-deep:latest",
    base_url="http://localhost:11434",
    temperature=0.7,
)

# Consideration and expectation strings.
positive_considerations = [
    "For positive sentiment, consider the following points: "
    "Write sentence sincerely thanking the customer. "
    "Write sentence noting that your success comes from valued customers like them. "
    "Write sentence expressing hope that the product provides lasting benefits. "
    "Write sentence lightly encouraging them to shop again with humor. "
]

negative_considerations = [
    "for negative sentiment, consider the following points: "
    "Write sentence apologizing for the issue. "
    "Write sentence thanking the customer for sharing their dissatisfaction and acknowledging their feedback. "
    "Write sentence elaborating on the issue (e.g., product quality, delivery issues, performance concerns). "
    "Write sentence offering a potential solution. "
]

neutral_considerations = [
    "for neutral sentiment, consider the following points: "
    "Write sentence thanking the customer for their feedback. "
    "Write sentence expressing appreciation for their loyalty. "
    "Write sentence complimenting the product they purchased. "
    "Write sentence asking what more can be done to improve their experience. ",
]

positive_expectations = [
    "For positive sentiment: "
    "A cheerful, empathetic response under 500 words with five paragprahs."
]

negative_expectations = [
    "For negative sentiment: "
    "A polite, solution-oriented response under 500 words with five paragprahs."
]

neutral_expectations = [
    "for neutral sentiment: "
    "A friendly, engaging response under 500 words with five paragprahs."
]

# Define the sentiment analysis agent.
sentiment_agent = Agent(
    role="You are a sentiment analysis agent with deep expertise in text evaluation"
         "You excel at identifying and classifying sentiments with precision"
         "You can detect emotions such as joy, anger, sadness, or excitement embedded in text",
    goal="To accurately classify the sentiment of user-provided text as Positive, Negative, or Neutral"
         "To identify the dominant emotion to guide tailored responses."
         "To deliver clear and reliable sentiment analysis for customer oriented responses.",
    backstory="You are an expert sentiment analysis specialist, trained on vast datasets of human expression."
              "Your experience spans diverse contexts, enabling nuanced understanding of text."
              "You thrive on transforming raw feedback into empathetic, actionable insights for the customer",
    llm=gemini
)

# sentiment review agent
sentiment_review_agent = Agent(
    role="You are a sentiment review agent with a keen eye for detail."
        "You specialize in validating sentiment analysis for accuracy and context."
        "You adeptly identify emotions like anger, joy, sadness, or excitement in text.",
    goal="To ensure the sentiment analysis output is precise and contextually appropriate."
         "To confirm the sentiment as Positive, Negative, or Neutral with confidence."
         "To capture the dominant emotion to enhance response relevance.",
    backstory="You are a meticulous expert, honed by years of scrutinizing text analysis."
              "Your deep understanding of linguistic nuances ensures reliable evaluations."
              "You take pride in refining insights to support meaningful customer interactions.",
    llm=gemini2,
    verbose=True,
    max_iterations=10,
)

# Define the response generation agent.
response_agent = Agent(
    role="You are a response agent with expertise in crafting thoughtful replies"
         "You excel at addressing customer feedback with empathy and precision"
         "You adapt responses to reflect sentiments and emotions like joy or frustration",
    goal="To generate empathetic and helpful responses based on sentiment and emotion analysis"
         "To address customer concerns appropriately, offering solutions for negative feedback"
         "To ensure every reply strengthens customer trust and satisfaction."
         "User available tool to craft the most empathetic, helpful, solution-oriented response.",
    backstory="You are a well-experienced response agent. "
              "You are well-experienced in generating responses based on the sentiment and emotion analysis."
              "Your experience in customer interactions ensures meaningful engagement."
              "You are driven to uphold business values through compassionate responses.",
    llm=gemini,
    max_iterations=25
)

# Define the reviewer agent.
reviewer_agent = Agent(
    role="You are a reviewer agent with a sharp focus on response quality."
         "You specialize in refining customer replies to meet high standards of empathy and clarity."
         "You ensure alignment with business standards and customer needs.",
    goal="To review and adjust responses to be empathetic, polite, and concise."
         "To ensure responses address customer concerns with effective solutions"
         "To deliver polished replies, maintaining 200 to 350 words."
         "Use the provided tools to enhance response quality"
         "Return a final response that is clear, concise, empathetic, warm note, helpful, and solution-oriented.",
    backstory="You are a meticulous reviewer agent with expertise in evaluating and perfecting customer communications."
            "Your keen insight ensures every response meets high standards of empathy and clarity."
            "You are dedicated to fostering trust through thoughtful refinements.",
    llm=gemini2,
    verbose=True,
    max_iterations=25,
)

# search_agent = Agent(
#     role="You are a search agent with expertise in finding relevant information online."
#          "You excel at retrieving accurate and up-to-date data to support customer inquiries."
#             "You can search for product details, solutions, and recommendations.",
#     goal="To conduct web searches to find relevant information for customer inquiries."
#             "To provide accurate and timely data to enhance customer interactions.",
#     backstory="You are a skilled search agent, adept at navigating the web for precise information."
#             "Your experience ensures that you deliver relevant and helpful data to support customer needs.",
#     llm=gemini2,
#     verbose=True,
#     max_iterations=25,
#     tools=[web_search],
# )

# Function to run the CrewAI tasks using the provided input.
def run_agent(combined_input):
    # Define the sentiment analysis task.
    sentiment_task = Task(
        description=f"Analyze the sentiment of the following text: {combined_input}"
                    "Classify the sentiment as Positive, Negative, or Neutral."
                    "Identify the dominant emotion expressed in the text.",
        expected_output="A JSON object containing two keys:"
                   "- 'sentiment': Classified as 'Positive', 'Negative', or 'Neutral'."
                   "- 'emotion': The dominant emotion, such as 'happy', 'sad', 'angry', or 'excited'.",
        agent=sentiment_agent,
    )

    # Define the sentiment review task.
    sentiment_review_task = Task(
        description="Review the sentiment analysis output for accuracy and contextual relevance."
                     "Validate the emotion analysis to ensure it reflects the text’s true tone."
                     "Adjust classifications if discrepancies are found.",
        expected_output="A JSON object with three keys:"
                    "- 'sentiment': Confirmed as 'Positive', 'Negative', or 'Neutral'."
                    "- 'emotion': Validated emotion, e.g., 'happy', 'sad', 'angry', or 'excited'."
                    "- 'review': A brief explanation of the validation or adjustments made.",
        agent=sentiment_review_agent,
        context=[sentiment_task]
    )
        
    # Define the response generation task.
    response_task = Task(
        description=f"Guidelines: {positive_expectations}, {negative_expectations}, {neutral_expectations}"
                     "Generate a tailored response based on sentiment and emotion analysis"
                     "Address the customer review with empathy and relevance."
                     "Follow sentiment-specific guidelines to ensure appropriateness.",
        expected_output="A response string with the following characteristics:"
                     f"- Get customer name, purchase date, product, review, sentiment, and emotion from {combined_input}."
                     "- If you find the customer review is ambiguous or not clear, you can ask further clarification"
                     "- if you are not sure how to respond the review just give light humor response."
                     "- never use square brackets or any other placeholders in the response."
                     "- never mention about sentiment analysis process."
                     "- If you find the review sarcastic or not clear, you can ask further clarification."
                     "If you find the review is irrelevant or not clear, you can reply with light humour."
                    "- No single placeholders are allowed in the response eg [your name]."
                    "- Starts with 'Dear Valued Customer' or the customer’s name."
                    "- Reflects the sentiment (Positive, Negative, or Neutral) per guidelines."
                    "- Incorporates empathy and solutions (if negative), within 350 words."
                    "- Structured in five paragraphs, adhering to business standards."
                    "- Ends with a warm, positive note."
                    "- If necessary you can search amazon for the product details",
        agent=response_agent,
        context=[sentiment_task, sentiment_review_task]
    )

    # Define the reviewer task for fine-tuning the response.
    reviewer_task = Task(
        description="Represent the Amazon Customer Service Team to refine customer responses."
                "Extract customer name, purchase date, product, review, sentiment, and emotion from input."
                "Review and adjust the response to ensure empathy, clarity, and alignment with Amazon standards.",
        expected_output="A polished empathetic response string with the following characteristics:"
                    f"- Get customer name, purchase date, product, review, sentiment, and emotion from {combined_input}."
                    "- If you find the customer review is ambiguous or not clear, you can ask further clarification"
                     "- if you are not sure how to respond the review just give light humor response."
                     "- If you find the review sarcastic or not clear, you can ask further clarification."
                     "If you find the review is irrelevant or not clear, you can reply with light humour."
                    "- never mention about sentiment analysis process."
                    "- never use square brackets or any other placeholders in the response."
                    "- Starts with 'Dear Valued Customer' or the customer’s name."
                    "- Addresses customer concern, sentiment, and emotion, within 200-300 words."
                    "- Uses clear sentence separation with newlines for readability."
                    "- For negative sentiment, includes solutions (e.g., new product for faulty items, delivery review for delays)."
                    "- For positive sentiment, invites repeat shopping with light humor."
                    "- Includes contact details: Customer Service Team, customer.satisfaction@amazon.com, 1-800-555-0199."
                    "- Ends with a warm, positive thank-you note"
                    "- Adheres to Amazon’s business standards of empathy and professionalism."
                    "- If needed, you can search amazon for the product recommendation or details"
                    "- if needed you can search the web for the product recommendation."
                    "- If needed you can search the web for the solution for the customer review."
                    "- You need to provide the link for the product recommendation or solution."
                    " - Do not forget to provide the link for the product recommendation or solution when necessary.",
        agent=reviewer_agent,
        context=[sentiment_task, sentiment_review_task, response_task],
        tools=[web_search]
    )
    
    # # define search task
    # search_task = Task(
    #     description=f"You need to learn about the customer review : {combined_input}"
    #                 "You need to get the response from the reviewer agent."
    #                 "If needed, from both the condition of customer review and reviewer agent."
    #                 "You need to provide the link for the product recommendation or solution.",
    #     expected_output="A response in a string format with the following characteristics:"
    #                      "Should have numbering or bullet points."
    #                      "Should have clear separation between the sentences."
    #                      "Provide simple explanation of the provided link."
    #                      "Provide the link for the product recommendation or solution."
    #                      "No need to provide any other information than what has described above.",
    #     agent=search_agent,
    #     context=[reviewer_task],
    #     tools=[web_search],
    # )

    # Create a Crew with the defined tasks and agents.
    crew1 = Crew(
        agents=[sentiment_agent, sentiment_review_agent, response_agent, reviewer_agent,],# search_agent],
        tasks=[sentiment_task, sentiment_review_task, response_task, reviewer_task,],# search_task],
        verbose=True,
        process=Process.sequential,
    )

    # Run the tasks.
    crew1.kickoff()

    # Return results as a dictionary so that the UI can display them.
    result = {
        "user_input": combined_input,
        "sentiment": sentiment_task.output.raw,
        "sentiment_review": sentiment_review_task.output.raw,
        "response": response_task.output.raw,
        "reviewed_response": reviewer_task.output.raw,
        "Used_Model": f"for sentiment analysis: {sentiment_agent.llm.model},"
                     f"for sentiment review: {sentiment_review_agent.llm.model},"
                     f"for response generation: {response_agent.llm.model},"
                     f"for reviewer agent: {reviewer_agent.llm.model}",
        # "search_task": search_task.output.raw,
    }
    
    return result