import boto3
from langchain_aws import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode
from typing import Annotated, Literal, Sequence, TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrock
from functools import partial

bedrock_runtime = boto3.client( 
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_runtime,
        region_name="us-east-1",
    )

index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
    )

loader = PyPDFLoader("Ml_2.pdf")
index_from_loader = index_creator.from_loaders([loader])
index_from_loader.vectorstore.save_local("/tmp")

faiss_index = FAISS.load_local("/tmp", embeddings, allow_dangerous_deserialization=True)
retriever=faiss_index.as_retriever()

retriever_tool=create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return the the documents related to question",
    )

tools=[retriever_tool]
retrieve=ToolNode([retriever_tool])



llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs=dict(temperature=0),
    region_name = "us-east-1"
)

llm_with_tools = llm.bind_tools(tools=tools)

# Sample question bank
question_bank = [
    "What is machine learning?",
    "Can you explain the difference between supervised and unsupervised learning?",
    "What is overfitting and how can you prevent it?"
]

# Agent state definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    question_index: int
    validated_answers: list
    current_question: str
    user_answer: str
    answer_validated: bool

def greet_user(state: AgentState) -> AgentState:
    print("🤖 Hi there! I'm your interview assistant. Let's get started.")
    return state

# 2. Wait for user's greeting
def wait_for_user_greeting(state: AgentState) -> AgentState:
    greeting = input("You: ").strip().lower()
    if any(word in greeting for word in ["hi", "hello", "hey", "good", "ready"]):
        return state
    else:
        print("🤖 Just say hi when you're ready.")
        return wait_for_user_greeting(state)

# 3. Ask the current question from the question bank
def ask_question(state: AgentState) -> AgentState:
    index = state.get("question_index", 0)
    question = question_bank[index]
    print(f"🤖 Question {index + 1}: {question}")
    state["current_question"] = question
    return state

# 4. Get the user's answer (CLI input)
def get_user_answer(state: AgentState) -> AgentState:
    state["user_answer"] = input("Your answer: ").strip()
    return state

# 5. Validate the answer using Claude and retriever
def validate_answer(state: AgentState, tools) -> AgentState:
    # Use the retriever tool to fetch relevant docs
    retriever_tool = tools[0]
    retrieved_docs = retriever_tool.invoke({"query": state["current_question"]})
    if isinstance(retrieved_docs, str):
        context = retrieved_docs  # Use the string as-is
    else:
        try:
            # Fallback for list/other types
            if hasattr(retrieved_docs[0], 'page_content'):  # Document objects
                context = "\n".join(doc.page_content for doc in retrieved_docs)
            else:  # Other iterables
                context = "\n".join(str(doc) for doc in retrieved_docs)
        except (TypeError, IndexError, AttributeError):
            context = str(retrieved_docs)  # Final fallback

    prompt = f"""Here is a user's answer to a question.

    Question: {state["current_question"]}
    Answer: {state["user_answer"]}

    Use the following context to validate the answer:
    {context}

    Is the answer correct and complete? Reply 'yes' or 'no' and explain why.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    print("🤖 Validation result:", response.content)

    if "yes" in response.content.lower():
        state["answer_validated"] = True
    else:
        state["answer_validated"] = False
    return state

# 6. Ask follow-up question
def ask_follow_up(state: AgentState) -> AgentState:
    follow_up = f"Can you clarify or expand more on this: {state['current_question']}?"
    print(f"🤖 Follow-up: {follow_up}")
    return state

# 7. Save validated answer and advance to next question
def save_valid_answer(state: AgentState) -> AgentState:
    state["validated_answers"].append({
        "question": state["current_question"],
        "answer": state["user_answer"]
    })
    state["question_index"] += 1
    return state

def wait_for_answer(state: AgentState) -> AgentState:
    state["user_answer"] = input("Your answer: ").strip()
    return state


# 8. End the interview
def end_interview(state: AgentState) -> AgentState:
    print("🤖 That's all for now. Thanks for your answers!")
    print("📄 Summary of your validated answers:")
    for i, entry in enumerate(state["validated_answers"], 1):
        print(f"{i}. {entry['question']} ➡ {entry['answer']}")
    return state


# Create your graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("start", greet_user)
graph.add_node("wait_for_greeting", wait_for_user_greeting)
graph.add_node("ask_question", ask_question)
graph.add_node("wait_for_answer", wait_for_answer)
graph.add_node("validate_answer", partial(validate_answer, tools=tools))
graph.add_node("follow_up", ask_follow_up)
graph.add_node("save_answer", save_valid_answer)
graph.add_node("end", end_interview)

# Define the flow
graph.set_entry_point("start")
graph.add_edge("start", "wait_for_greeting")
graph.add_edge("wait_for_greeting", "ask_question")
graph.add_edge("ask_question", "wait_for_answer")
graph.add_edge("wait_for_answer", "validate_answer")

# Decision logic after validation

def validation_router(state: AgentState) -> tuple[AgentState, str]:
    if state.get("answer_validated", False):
        return state, "save_answer"
    else:
        return state, "follow_up"

graph.add_conditional_edges("validate_answer", validation_router)


# Follow-up response path
graph.add_edge("follow_up", "wait_for_answer")

# Valid path: save then continue or finish
def next_step(state: AgentState) -> str:
    if state["question_index"] + 1 < len(question_bank):
        return "ask_question"
    else:
        return "end"

graph.add_conditional_edges("save_answer", next_step)

# Compile graph
interview_graph = graph.compile()

initial_state = {
    "messages": [],
    "question_index": 0,
    "validated_answers": [],
    "current_question": "What's your name?",
    "user_answer": "",
    "answer_validated": False,
    
}



interview_graph.invoke(initial_state)