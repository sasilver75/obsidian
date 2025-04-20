LangGraph Unit

[[LangGraph]] is a framework that allows you to build production-ready applications by giving you control tools over the flow of your agent.

- This is just an intro; there's a free LangChain academy course: Introduction to LangGraph!
- LangChain provide a standard interface to interact with models and other components
	- Retrieval
	- LLM cals
	- Tool Calls
	- ...
- Classes from LangChain MIGHT be used in LangGraph, but do not HAVE to be used.
- ==LangGraph== is a framework developed by LangChain to manage the control flow of applications that integrate an LLM.


When should we use LangGraph?
- Control vs Freedom
	- ==Freedom== gives your LLM more room to move and be creative and tackle unexpected problems.
	- ==Control== allows you to ensure predictable behavior and maintain guardrails.

Code Agents (like those in smolagents) are VERY free, calling multiple tools in a single step, creating their own tools, etc. But this can make them less predictable and less controllable than a regular Agent working with JSON!

==LangGraph== is on the *other* end of the spectrum -- it shines when you need ==Control== on the execution of your agent!
- LangGraph is particularly valuable when you need Control over your applications, giving you to the tools to build an application that follows a predictable process while still leveraging the power of LLMs.

==If your application involves a series of steps that need to be orchestrated in a particular way, with decisiosn made at each junction point, LangGraph provides the structure you need!==

If we're building an LLM assistant that can answer some questions over some docuemnts:
- Since LLMs understand text the best, before being able to answer the question, you will need to convert other complex modalities (chart, table) into text.


This is a branching that we choose to represent as follows:
- While this branching is deterministic, you can also design branching that are conditioned on the output of an LLM, making them undeterministic.
- ==The key scenarios where LangGraph excels include:==
	- ==Multi-step Reasoning Processes== that need ==explicit control on== the flow
	- Applications requiring ==persistence of state between steps==
	- Systems that ==combine deterministic logic with AI capabilities==
	- Workflows that need ==human-in-the-loop interventions==
	- Complex agent architectures with ==multiple components working together==


# ==LangGraph is the most production-ready agent framework on the market.==


## How does LangGraph work?
- At its core, LangGraph uses a directed graph structure to define the flow of your application.
	- ==Nodes==: Represent ***individual processing steps*** (like calling an LLM, using a tool, making a decision)
	- ==Edges==: Define the ***possible transitions*** between steps
	- ==State==: ***User-defined and maintained and passed between nodes during execution***. 
		- When deciding which node to target next, this is the state we look at.

You might think:
- Why don't I just write regular Python code with if-else statements to handle all these flows, right?
- You could, but LAngGraph has some advantages; it has some easier tools and abstractions for you:
	- Includes
		- States
		- Visualization
		- Logging(traces)
		- Built-in HITL
		- more!

![[Pasted image 20250419185521.png]]

## State

The central concept in LangGraph is ==State==. It represents all information flowing through your app.

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

The state is ==USER DEFINED==, hence the fields should be carefully crafted to contain all data needed for the decision-making process.


## Nodes

Nodes are Python functions. Each node:
- Takes the state as input
- Performs soem operation
- Returns updates to the state

```python
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}
```

For example, Nodes can contain:
- ==LLM== calls: Generate text or making decisions
- ==Tool== calls: Interacting with external steps
- ==Conditional Logic==: Determining next steps
- ==Human interventions==: Getting inputs from users

## Edges
- Edges connect nodes and define the possible paths through your graph:
```python
import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"
```
Edges can be:
- ==Direct==: Always go from node A to node B
- ==Conditional==: Choose the next node based on the current state

## StateGraph
- The state graph is the container that holds your entire agent workflow!

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State) # Where is State coming frmo?
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

  
  

# We can then invoke the graph
graph.invoke({"graph_state": "Hi, this is Lance."})
```


---------------

Building our First LangGraph...

![[Pasted image 20250419190359.png]]


```python
import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Step 1: Define our State
# Our state needs to be comprehensive enough to track all important info, but we need to avoid bloating it.

class EmailState(TypedDict):
    # The email being processed
    email: Dict[str, Any]  # Contains subject, sender, body, etc.

    # Category of the email (inquiry, complaint, etc.)
    email_category: Optional[str]

    # Reason why the email was marked as spam
    spam_reason: Optional[str]

    # Analysis and decisions
    is_spam: Optional[bool]
    
    # Response generation
    email_draft: Optional[str]
    
    # Processing metadata
    messages: List[Dict[str, Any]]  # Track conversation with LLM for analysis


# Step 2: Defining our Nodes
# Now let's make the processing functions that will form our nodes:

# Initialize our LLM
model = ChatOpenAI(temperature=0)


#  BELOW, WE DEFINE 4 FUNCTIONS. YOU'LL SEE THAT THEY'RE BASICALLY JUST THE NODES IN OUR GRAPH!
# EACH NODE MODIFIE STATE (OR NOT)

# Function that takes our Emailstate and apparently makes no changes to state
# It seems to me that return {} means tht the returned dict is MERGED into the existing EmailState state?
def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]
    
    # Here we might do some initial preprocessing
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")
    
    # No state changes needed here
    return {}

#  Function to classify an email...
def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]
    
    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """
    
    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

	##### BELOW CONTENT IS PARSING THE RESOPNSE for is_spam, spam_reason, email_category, messages

    # Simple logic to parse the response (in a real app, you'd want more robust parsing)
    response_text = response.content.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text
    
    # Extract a reason if it's spam
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()
    
    # Determine category if legitimate
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # Return state updates
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }

# Handling Spam
def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")
    
    # We're done processing this email
    return {}

# Drafting Response
def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"
    
    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    This email has been categorized as: {category}
    
    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """
    
    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # Return state updates
    return {
        "email_draft": response.content,
        "messages": new_messages
    }

def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]
    
    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["email_draft"])
    print("="*50 + "\n")
    
    # We're done processing this email
    return {}

```

Next, we need to define our ROUTING LOGIC!

```python
def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"
```

Let's define our STateGraph and Define our Edges:
```python
# Create the graph
email_graph = StateGraph(EmailState)

# Add nodes
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

# Start the edges
email_graph.add_edge(START, "read_email")
# Add edges - defining the flow
email_graph.add_edge("read_email", "classify_email")

# Add conditional branching from classify_email
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_response"
    }
)

# Add the final edges
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

# Compile the graph
compiled_graph = email_graph.compile()
```

And now we can run the application!

```python
# Example legitimate email
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
}

# Example spam email
spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
}

# Process the legitimate email
print("\nProcessing legitimate email...")
legitimate_result = compiled_graph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})

# Process the spam email
print("\nProcessing spam email...")
spam_result = compiled_graph.invoke({
    "email": spam_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})
```


We can also inspect our mail sorting agent with Langfuse

```python
import os
 
# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..." 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region

from langfuse.callback import CallbackHandler

# Initialize Langfuse CallbackHandler for LangGraph/Langchain (tracing)
langfuse_handler = CallbackHandler()

# Process legitimate email
legitimate_result = compiled_graph.invoke(
    input={"email": legitimate_email, "is_spam": None, "spam_reason": None, "email_category": None, "draft_response": None, "messages": []},
    config={"callbacks": [langfuse_handler]}
)
```

## What We’ve Built

We’ve created a complete email processing workflow that:

1. Takes an incoming email
2. Uses an LLM to classify it as spam or legitimate
3. Handles spam by discarding it
4. For legitimate emails, drafts a response and notifies Mr. Hugg


##   
Key Takeaways

- **State Management**: We defined comprehensive state to track all aspects of email processing
- **Node Implementation**: We created functional nodes that interact with an LLM
- **Conditional Routing**: We implemented branching logic based on email classification
- **Terminal States**: We used the END node to mark completion points in our workflow

