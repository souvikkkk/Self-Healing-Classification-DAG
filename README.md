# Self-Healing-Classification-DAG
## Task Overview    
   
Building a LangGraph-based classification pipeline that not only performs predictions but also   
incorporates a self-healing mechanism. The goal is to fine-tune a transformer model and    
design a fallback strategy in cases of low prediction confidence, ensuring robustness and     
reliability in human-in-the-loop workflows     

## Objective    
   
Fine-tune a transformer-based text classification model, then integrate it into a LangGraph   
DAG that uses prediction confidence to decide whether to accept, reject, or request clarification   
for its outputs. The design should prioritize correctness over blind automation.      

## Requirements

- Choose any open-access text classification dataset (e.g., sentiment analysis, topic classification,
emotion labeling).     
- Fine-tune a transformer model (e.g., DistilBERT, TinyBERT, etc.) using LoRA or full
fine-tuning.
- Build a LangGraph workflow composed of the following nodes:   
    - InferenceNode: Runs classification using the trained model.   
    - ConfidenceCheckNode: Evaluates the confidence score of the prediction. If 
    below a defined threshold, triggers fallback.   
    - FallbackNode: Avoids incorrect classification by either:   
        - Asking the user for clarification or additional input, or   
        - Escalating to an alternative strategy (e.g., backup model).   
- Maintain a clean CLI interface to handle:   
    - User inputs   
    - Clarification questions   
    - Final outputs with confidence scores
 - Implement structured logging for:   
    - Initial predictions and associated confidence   
    - Fallback activations and corresponding user interactions   
    - Final classification decisions

## Installation  
  ### 🧪 Training the Model   
      - python train_lora.py   
  ### 🚦 Running the Self-Healing DAG      
      - python langgraph_dag.py   

## 💬 Example Interaction   
  Enter a movie review: The movie was painfully slow and boring.    
   
[InferenceNode] Predicted label: Positive | Confidence: 54%   
[ConfidenceCheckNode] Confidence too low. Triggering fallback...   
[FallbackNode] Do you agree with the prediction 'LABEL_1'? (yes/no):   
> no   
Please enter the correct label (e.g., 'LABEL_0' or 'LABEL_1'):   
> LABEL_0    
 
Final Prediction: LABEL_0   
Confidence: 54.0 %   
Fallback Used: True    


