from langgraph.graph import StateGraph, END
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import TypedDict
from langchain_core.runnables import RunnableLambda

import logging
import os

# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/dag_run.log"),
        logging.StreamHandler()  # Optional: also print to console
    ]
)


# 1. Define the state schema
class GraphState(TypedDict):
    input_text: str
    prediction: str
    confidence: float
    fallback: bool
    route: str

# 2. Load model + tokenizer
model_path = "models/lora-distilbert-imdb"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

# 3. Inference node
def inference_node(state: GraphState) -> GraphState:
    result = classifier(state["input_text"])[0]
    result.sort(key=lambda x: x["score"], reverse=True)
    return {
        **state,
        "prediction": result[0]["label"],
        "confidence": result[0]["score"]
    }

# 4. Router node (returns dict with 'route')
def confidence_check_node(state: GraphState) -> dict:
    threshold = 0.8
    route = "high" if state["confidence"] >= threshold else "low"
    return {"route": route}

# 5. Fallback node
def fallback_node(state: GraphState) -> GraphState:
    print(f"\n⚠️ Low confidence: {state['confidence']*100:.2f}%")
    print(f"Prediction: {state['prediction']}")
    answer = input("Do you agree with this prediction? (yes/no): ").strip().lower()
    if answer == "no":
        correction = input("Enter the correct label (e.g., LABEL_0 or LABEL_1): ")
        state["prediction"] = correction
    state["fallback"] = True
    return state

# 6. Build graph
graph_builder = StateGraph(GraphState)

graph_builder.add_node("InferenceNode", inference_node)
graph_builder.add_node("ConfidenceCheckNode", confidence_check_node)
graph_builder.add_node("FallbackNode", fallback_node)
graph_builder.add_node("EndNode", lambda x: x)

graph_builder.set_entry_point("InferenceNode")
graph_builder.add_edge("InferenceNode", "ConfidenceCheckNode")

# ✅ Fix: wrap target strings in RunnableLambda
graph_builder.add_conditional_edges("ConfidenceCheckNode", {
    "high": RunnableLambda(lambda x: x).with_config(run_name="EndNode"),
    "low": RunnableLambda(fallback_node).with_config(run_name="FallbackNode")
})

graph_builder.add_edge("FallbackNode", END)
graph_builder.add_edge("EndNode", END)

graph = graph_builder.compile()

# 7. CLI input/output
if __name__ == "__main__":
    input_text = input("Enter a movie review: ")

    final_result = graph.invoke({
        "input_text": input_text,
        "prediction": "",
        "confidence": 0.0,
        "fallback": False,
        "route": ""
    })

    print("\n✅ Final Prediction:", final_result["prediction"])
    print("🔢 Confidence:", round(final_result["confidence"] * 100, 2), "%")
    print("🧭 Fallback Triggered:", final_result["fallback"])
