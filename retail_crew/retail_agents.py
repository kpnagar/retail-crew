from crewai import Agent
from langchain_community.llms import Ollama
import tools as t
from datetime import datetime
from crewai import Task, Crew, Process

llm = Ollama(model="mistral")

chitchat_llm = Ollama(model="gemma:2b")

personal_assistant = Agent(
    role='Personal Conversational Assistant',
    goal='Engage in conversations with the user',
    backstory="""You're a personal assistant who likes to talk.
You're responsible for keeping the user engaged in the conversation.
You're always truthful and respectful.""",
    llm=chitchat_llm,
    memory=True,
)

purchase_assistant = Agent(
    role='Product Purchase Assistant',
    goal='Help user purchase a product',
    backstory=f"""You're a personal assistant who helps the user purchase a product.
You're responsible for buying the product on behalf of the user.
You're always truthful and respectful. Today is: {datetime.utcnow()}""",
    llm=llm,
    memory=True,
    tools=[t.product_purchase, t.schedule_purchase],
    max_iter=3
)

tracking_assistant = Agent(
    role='Price&Order Tracking Assistant',
    goal='Help user track change in price of a product and their order status',
    backstory=f"""You're a personal assistant who helps the user track change in price of a product and the status of 
    the order placed by them. You're always truthful and respectful. Today is: {datetime.utcnow()}""",
    llm=llm,
    memory=True,
    tools=[t.price_tracking, t.order_tracking],
    max_iter=3
)

product_purchase = Task(
    description="Purchase the product immediately for the user",
    expected_output='Appropriate message for the user as per the success or failure of the purchase.',
    tools=[t.product_purchase],
    agent=purchase_assistant,
)

schedule_purchase = Task(
    description="Schedule the purchase of a product in the time mentioned by the user.",
    expected_output='Appropriate message for the user as per the success or failure of the purchase.',
    tools=[t.schedule_purchase],
    agent=purchase_assistant,
)

price_tracking = Task(
    description="Track the price of a particular product and notify the user when they go down",
    expected_output='Appropriate message for the user as per the success or failure of the action.',
    tools=[t.price_tracking],
    agent=tracking_assistant,
)

order_tracking = Task(
    description="Track the status of the order placed by user.",
    expected_output='Appropriate message informing the status of the order and the product purchased by him.',
    tools=[t.order_tracking],
    agent=tracking_assistant,
)

chatting = Task(
    description="Chatting with the user and engaging in conversations with them",
    expected_output="Appropriate response to the user's message. It should always be honest and respectful",
    tools=[],
    agent=personal_assistant,
)

crew = Crew(
    tasks=[chatting, product_purchase, schedule_purchase, price_tracking, order_tracking],
    agents=[personal_assistant, purchase_assistant, tracking_assistant],
    manager_llm=llm,
    process=Process.hierarchical
)

crew.kickoff({'user_query': "I want to buy an iPhone"})
