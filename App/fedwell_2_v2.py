import streamlit as st
import pandas as pd
import os
import sys
import ast
import time
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import numpy as np
import torch
import re
import altair as alt
import matplotlib.pyplot as plt
import redis
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import altair
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
 
print(altair.__version__)



#installation_path = "/netscratch/krajshekar/Fedwell" #change here
#cache_dir = "/netscratch/krajshekar/Fedwell" #change here

installation_path = "/netscratch/psaxena/Llama_Inference/CEO_Prediction/App"
cache_dir ="/netscratch/psaxena/Llama_Inference/CEO_Prediction/App"

os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir
sys.path.insert(0, installation_path)

os.environ["HF_TOKEN"] = "your_huggingface_password_here"

# --------------------------
# Streamlit page config
# --------------------------
st.set_page_config(
    page_title="Llama3-8B Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_data" not in st.session_state:
    st.session_state.user_data = {}

st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

redis_conn = redis.Redis(
  host='redis-17861.c300.eu-central-1-1.ec2.redns.redis-cloud.com',
  port=17861,
  password='your_redis_password_here'  # Replace with your actual password
)


tokenizer1 = model1 = tokenizer2 = model2 = None #remove later


@st.cache_resource

def load_model():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
    )
    print("model loading")

    model_name = "meta-llama/Llama-3.1-8B-Instruct" #ebadullah371/llama_3_8b_ft_10_chat_v2
    tokenizer1 = AutoTokenizer.from_pretrained(
        model_name,
        token=os.environ["HF_TOKEN"],
        cache_dir=cache_dir
    )
    model1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ["HF_TOKEN"],
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        # torch_dtype="auto",
    ).to("cuda")

    if tokenizer1.pad_token_id is None:
        tokenizer1.pad_token_id = tokenizer1.eos_token_id
    # """
    # model_name2 = "meta-llama/Llama-3.1-8B-Instruct"
    # tokenizer2 = AutoTokenizer.from_pretrained(
    #     model_name2,
    #     token=os.environ["HF_TOKEN"],
    #     cache_dir=cache_dir
    # )
    # model2 = AutoModelForCausalLM.from_pretrained(
    #     model_name2,
    #     token=os.environ["HF_TOKEN"],
    #     cache_dir=cache_dir,
    #     torch_dtype="auto"
    # ).to("cuda")
    # """

    return tokenizer1, model1 #, tokenizer2, model2

#tokenizer1, model1, tokenizer2, model2 = load_model()  add later
print("model loaded")

def question_answer(messages):
    input_ids = tokenizer1.apply_chat_template(messages, return_tensors="pt").to('cuda') #to.cuda
    outputs = model1.generate(
        input_ids=input_ids,
        top_p=0.9,
        temperature=0.9,
        do_sample=False,
        max_new_tokens=800,
        return_dict_in_generate=True,
        pad_token_id=tokenizer1.eos_token_id
    )
    output_text = tokenizer1.decode(outputs.sequences[0])
    print('Response SPECIFIC: ', output_text)
    return output_text

def question_answer_main(prompt: str, max_new_tokens=20) -> str:
    inputs = tokenizer1(prompt, return_tensors="pt") #tokenizer2
    inputs = {k: v.to("cuda") for k, v in inputs.items()} #to.cuda
    
    outputs = model1.generate( #model2
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        top_p=0.9,
        temperature=0.9,
        max_new_tokens=max_new_tokens,
        stop_token_id=tokenizer1.convert_tokens_to_ids("</assistant>") #tokenizer2
    )
    print("*****************************")
    print(tokenizer1.decode(outputs[0], skip_special_tokens=True)) #tokenizer2
    return tokenizer1.decode(outputs[0], skip_special_tokens=True) #tokenizer2

def question_answer_main2(messages):
    input_ids = tokenizer1.apply_chat_template(messages, return_tensors="pt").to("cuda") #tokenizer2, #to.cuda
    outputs = model1.generate( #model2
        input_ids=input_ids,
        top_p=0.9,
        temperature=0.9,
        do_sample=False,
        max_new_tokens=500,
        return_dict_in_generate=True,
        pad_token_id=tokenizer1.eos_token_id #tokenizer2
    )
    output_text = tokenizer1.decode(outputs.sequences[0]) #tokenizer2
    print("Response: ", output_text)
    return output_text



def model_stream_generator(prompt: str, user_input: str):
    
    global model_response
    raw_text = question_answer(prompt)
    model_response = raw_text
    print("printing model response")
    print(model_response)
    #data_ = extract_text(model_response, user_input)
    data_ = extract_last_assistant_dict(model_response)

    print("printing json dict")
    json_dict = ast.literal_eval(data_)
    if isinstance(json_dict, dict):
        pass 
    else:
        json_dict = {
            "score": 0,
            "reason": "Please try again."
        }
    print(json_dict)
    words = json_dict["reason"].split()
    assembled = ""
    for w in words:
        assembled += w + " "
        yield w + " "
        time.sleep(0.05)

    # Mark end of stream
    yield "__STREAMLIT_STREAM_DONE__" + assembled

def model_stream_generator_general(prompt: list, user_input: str):
    
    global model_response
    raw_text = question_answer_main2(prompt)
    model_response = raw_text
    print("printing raw response")
    print(model_response)
    print("printing model response")
    text_ = extract_last_assistant_response(model_response)
    words = text_.split()
    assembled = ""
    for w in words:
        assembled += w + " "
        yield w + " "
        time.sleep(0.05)

    # Mark end of stream
    yield "__STREAMLIT_STREAM_DONE__" + assembled

def extract_last_assistant_response(text: str):
    user_pattern = re.escape("<|start_header_id|>user<|end_header_id|>")
    assistant_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(\{.*?\}|\S.*?)\s*<\|eot_id\|>"

    # Find the last occurrence of the user tag
    user_matches = list(re.finditer(user_pattern, text))
    if not user_matches:
        return "Can you please repeat your question?"
    
    last_user_match = user_matches[-1]
    start_index = last_user_match.end()  # Start searching after the last <user>

    # Find the first <assistant> after the last <user>
    assistant_match = re.search(assistant_pattern, text[start_index:], re.DOTALL)
    if not assistant_match:
        return "Can you please repeat your question?"
    
    return assistant_match.group(1).strip()

def extract_last_assistant_dict(text: str):
    assistant_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(\{.*?\})\s*<\|eot_id\|>"

    # Find all assistant responses containing JSON-like structures
    matches = re.findall(assistant_pattern, text, re.DOTALL)

    if not matches:
        return None  # No valid JSON response found

    return matches[-1]  

def extract_text(text: str, user_string: str):
    user_pattern = re.escape(f"<user>{user_string}</user>")
    assistant_pattern = r"<assistant>(.*?)</assistant>"
    
    # Find the last occurrence of the user_string
    user_matches = list(re.finditer(user_pattern, text))
    if not user_matches:
        return "Can you please repeat your question?"
    
    last_user_match = user_matches[-1]
    start_index = last_user_match.end()  # Start searching after last <user> match
    
    # Find the first <assistant> after the last <user>
    assistant_matches = list(re.finditer(assistant_pattern, text[start_index:]))
    if not assistant_matches:
        return "Can you please repeat your question?"
    
    return assistant_matches[0].group(1).strip()

def extract_json_dict(text: str):
    start_idx = text.rfind('<assistant>') + len('<assistant>')
    end_idx = text.rfind('</assistant>')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        text_class = text[start_idx:end_idx].strip()
        data = ast.literal_eval(text_class)
        if isinstance(data, dict):
            return data
        else:
            return {}
    return {}
def extract_entity_v3(text):
    start_tag = '<assistant>'
    end_tag = '</assistant>'
    
    start_idx = text.find(start_tag)
    while start_idx != -1:
        start_idx += len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        if end_idx != -1:
            entity = text[start_idx:end_idx].strip()
            # Validate that it matches the expected dictionary-like structure
            if entity.startswith("{") and entity.endswith("}"):
                return entity
        # Move to the next occurrence
        start_idx = text.find(start_tag, end_idx)
    
    return None

def build_prompt_main(user_input) -> str:
    prompt = f"""
        <system>
        You are a helpful assistant. You need to characterize the user question into one of the two categories.
        If the user asks about doing an exercise then return SPECIFIC, and if the user asks about something else like about themselves, their habits, reports, health etc then return GENERAL.
        JUST Return a VALID dictionary like below:
        {{ 
            "key": "SPECIFIC or GENERAL"
        }}
        </system>
        <user>{user_input}</user>\n<assistant>
        """
    # messages_list = []

    # for message in messages:
    #     if message["role"] == "user":
    #         messages_list.append(
    #             f"<user>{message['content']}</user>\n<assistant>"
    #         )
    #     elif message["role"] == "assistant":
    #         messages_list.append(
    #             f"<assistant>{message['content']}</assistant>"
    #         )

    # # Combine the messages into a single string for output
    # formatted_messages = "\n".join(messages_list)
    # prompt_main = prompt + formatted_messages
    # print(prompt_main)
    return prompt

def build_prompt_general(user_data: dict, messages: list) -> str:
    topK = 10
    ITEM_QUESTION_EMBEDDING_FIELD='item_question_vector'


    messages_list_inital = [{"role": "system", "content": prompt}]

    for message in messages:
        if message["role"] == "user":
            messages_list_inital.append({"role": "user", "content": message['content']})
        elif message["role"] == "assistant":
            messages_list_inital.append({"role": "assistant", "content": message['content']})

    query = next(
        (msg["content"] for msg in reversed(messages_list_inital) if msg["role"] == "user"),
        None 
        )
    # Vectorize the query
    query_vector = model_sentence_transformer.encode(query).astype(np.float32).tobytes()

    # Prepare the query
    q = Query(f'(@INDEX:{{KNEE}})=>[KNN {topK} @{ITEM_QUESTION_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0,topK).return_fields('vector_score','INDEX', 'Question', 'Answer').dialect(2)
    params_dict = {"vec_param": query_vector}

    # Execute the query
    results = redis_conn.ft('QA_index').search(q, query_params = params_dict)

    result_string = ""
    # Print similar products found
    for product in results.docs:
        print('***************Q/A found ************')
        result_string = result_string + "Question:\n"+product.Question+"\nAnswer:\n"+product.Answer+"\n\n"


    context = result_string
    
    prompt = f"""
        You are a helpful assistant. 

        You will be given user data including the Big 5 personality traits and their demographic and health information along with some context. Based on this, the user can ask questions regarding their data. JUST ANSWER THIER QUESTION. NO NEED TO THINK BY YOURSELF.

        Extroversion: {user_data.get('Extroversion', '')}/7 (General Norm: 4.44)
        Agreeableness: {user_data.get('Agreeableness', '')}/7 (General Norm: 5.23)
        Conscientiousness: {user_data.get('Conscientiousness', '')}/7 (General Norm: 5.4)
        Emotional Stability: {user_data.get('Emotional_Stability', '')}/7 (General Norm: 4.83)
        Openness: {user_data.get('Openness', '')}/7 (General Norm: 5.38)

        Age: {user_data.get('Age', '')}
        Gender: {user_data.get('Gender', '')}
        Employment status: {user_data.get('Employment_status', '')}
        Current emotional state: {user_data.get('Current_emotional_state', '')}
        Have any physical disabilities: {user_data.get('Have_any_physical_disabilities', '')}
        Type of physical activities: {user_data.get('Type_of_physical_activities', '')}
        How many days do you do exercise: {user_data.get('How_many_days_do_you_do_exercise', '')}
        Overall health status: {user_data.get('Overall_health_status', '')}
        Current mobility: {user_data.get('Current_mobility', '')}

        User Response to the question: {user_data.get('question_answer_string', '')}

        REMEMBER, ONLY ANSWER WHAT USER ASKED. NO NEED TO GENERATE STUFF BY YOURSELF.

        BELOW IS THE CONTEXT:\n\n

        {context}

        """
    messages_list = [{"role": "system", "content": prompt}]

    for message in messages:
        if message["role"] == "user":
            messages_list.append({"role": "user", "content": message['content']})
        elif message["role"] == "assistant":
            messages_list.append({"role": "assistant", "content": message['content']})

    
    return messages_list

def build_prompt(user_data: dict, messages: list) -> str:
    topK = 10
    ITEM_QUESTION_EMBEDDING_FIELD='item_question_vector'
    
    messages_list_inital = [{"role": "system", "content": prompt}]

    for message in messages:
        if message["role"] == "user":
            messages_list_inital.append({"role": "user", "content": message['content']})
        elif message["role"] == "assistant":
            messages_list_inital.append({"role": "assistant", "content": message['content']})

    query = next(
        (msg["content"] for msg in reversed(messages_list_inital) if msg["role"] == "user"),
        None 
        )
    # Vectorize the query
    query_vector = model_sentence_transformer.encode(query).astype(np.float32).tobytes()

    # Prepare the query
    q = Query(f'(@INDEX:{{KNEE}})=>[KNN {topK} @{ITEM_QUESTION_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0,topK).return_fields('vector_score','INDEX', 'Question', 'Answer').dialect(2)
    params_dict = {"vec_param": query_vector}

    # Execute the query
    results = redis_conn.ft('QA_index').search(q, query_params = params_dict)

    result_string = ""
    # Print similar products found
    for product in results.docs:
        print('***************Q/A found ************')
        result_string = result_string + "Question:\n"+product.Question+"\nAnswer:\n"+product.Answer+"\n\n"


    context = result_string


    prompt = f"""
        You are a helpful assistant. 

        You will be given user data including the Big 5 personality traits and their demographic and health information along with some context. Based on this, the user is asked about the difficulty of performing an exercise and provides their own rating. Your job is to reflect the user's score based on their data.
        
        ALWAYS ONLY RETURN A JSON IN BELOW FORMAT.

        Remember:
        - The user is always correct.
        - The question: "How difficult does it look to perform 10 squats?"
        - Rate difficulty from 1 to 5.
        - You have the following user data:

        Extroversion: {user_data.get('Extroversion', '')}/7 (General Norm: 4.44)
        Agreeableness: {user_data.get('Agreeableness', '')}/7 (General Norm: 5.23)
        Conscientiousness: {user_data.get('Conscientiousness', '')}/7 (General Norm: 5.4)
        Emotional Stability: {user_data.get('Emotional_Stability', '')}/7 (General Norm: 4.83)
        Openness: {user_data.get('Openness', '')}/7 (General Norm: 5.38)

        Age: {user_data.get('Age', '')}
        Gender: {user_data.get('Gender', '')}
        Employment status: {user_data.get('Employment_status', '')}
        Current emotional state: {user_data.get('Current_emotional_state', '')}
        Have any physical disabilities: {user_data.get('Have_any_physical_disabilities', '')}
        Type of physical activities: {user_data.get('Type_of_physical_activities', '')}
        How many days do you do exercise: {user_data.get('How_many_days_do_you_do_exercise', '')}
        Overall health status: {user_data.get('Overall_health_status', '')}
        Current mobility: {user_data.get('Current_mobility', '')}

        User Response to the question: {user_data.get('question_answer_string', '')}

        JUST Return a json like below. DO NOT RETURN ANY TEXT BEFORE OR AFTER THE JSON:
        {{ 
            "score": 1 to 5,
            "reason": THE REASON THE DOCTOR THINKS THE USER CAN OR CANNOT DO THE EXERCISE,
            "user_score": 1 to 5
        }}

        BELOW IS THE CONTEXT:\n\n

        {context}
        """
    messages_list = [{"role": "system", "content": prompt}]

    for message in messages:
        if message["role"] == "user":
            messages_list.append({"role": "user", "content": message['content']})
        elif message["role"] == "assistant":
            messages_list.append({"role": "assistant", "content": message['content']})

    return messages_list


def show_home():
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        st.image("assets/assets/fedwell_logo.png", use_container_width=False, width=200)
    
    # Middle column left empty (or you could put a title here)
    with col2:
        #st.title("MENTALYTICS")
        st.markdown("<h1 style='text-align: center;'>MENTALYTICS</h1>", unsafe_allow_html=True)
    
    with col3:
        st.image("assets/assets/dfki_logo2.png", use_container_width=False, width=100)

    #st.title("MENTALYTICS")
    st.subheader("Description")
    st.write(""" 
    Effective medical rehabilitation depends on good patient-therapist collaboration. But, cognitive impairments and psychological barriers affect nearly 30\% of rehabilitation patients, hindering clear communication and accurate patient assessments. Artificial Mental Models (AMMs) offer a solution by capturing patients' implicit expectations about therapy, aiding clearer communication and better treatment decisions. We demonstrate MENTALYTICS, a tool for knee rehabilitation, employing AMMs developed from fine-tuned large language models (LLaMA-2, LLaMA-3, GPT-4.o-mini). Trained via systematic data collection and an empirical user study (n=116), the proposed AMM  predicts patients' expected pain and effort during exercises. The optimized LLaMA-3 (8B) model outperformed larger models, highlighting issues of overfitting and generalization. Results show that LLMs can serve as effective baseline models for AMMs, though challenges remain in domain-specific fine-tuning.
             
    """)
    # Check if the image exists in the correct path
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        pass
    with col2:
        st.image("assets/assets/FedWell-Demo.png", use_container_width=True, width=1000)
    with col3:
        pass


    image_path = "assets/assets/csm_Project_1669_413615496a.png"
    
    st.subheader("Approach")
    st.write(""" 
    We propose AMM (Artificial Mental Model Framework) shown in figure, a computational system designed to capture patient beliefs, perspectives, and cognitive understanding to support personalized treatment planning. The AMM platform integrates a visualization tool that enables doctors to interpret patients' mental models, thereby improving the effectiveness of diagnosis and treatment strategies. Additionally, the framework includes a real-time conversational agent that assists patients in enhancing their medical knowledge and supports informed decision-making throughout their treatment journey. 
    """)


def get_exercise_emoji(exercise):
    emoji_map = {
        "squats": "🏋️‍♂️",
        "calf raises": "🦵",
        "standing toe touch": "🤸",
        "toe touch stretch": "🧘",
        "push-ups": "💪",
        "jumping jacks": "🤾",
        "plank": "🛠️",
        "lunges": "🚶",
        "running": "🏃",
        "cycling": "🚴",
        "swimming": "🏊",
        "yoga": "🧘‍♀️"
    }
    
    return emoji_map.get(exercise.lower(), "🏋️")

def get_exercise_icon(exercise):
    image_map = {
        "squats": "standing_toe_touch.png",
        "calf raises": "standing_toe_touch.png",
        "standing toe touch": "standing_toe_touch.png",
        "toe touch stretch": "standing_toe_touch.png"
    }
    return os.path.join("assets/assets", image_map.get(exercise, "standing_toe_touch.png"))  



def show_user_data():
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        st.image("assets/assets/fedwell_logo.png", use_container_width=False, width=200)
    
    # Middle column left empty (or you could put a title here)
    with col2:
        
        pass  # or st.write("Some title here")
    
    with col3:
        st.image("assets/assets/dfki_logo2.png", use_container_width=False, width=100)


    st.title("Provide Your Details")

    # st.session_state.user_data["Extroversion"] = st.number_input("Extroversion (1-7)", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
    # st.session_state.user_data["Agreeableness"] = st.number_input("Agreeableness (1-7)", min_value=1.0, max_value=7.0, value=5.0, step=0.1)
    # st.session_state.user_data["Conscientiousness"] = st.number_input("Conscientiousness (1-7)", min_value=1.0, max_value=7.0, value=5.0, step=0.1)
    # st.session_state.user_data["Emotional_Stability"] = st.number_input("Emotional Stability (1-7)", min_value=1.0, max_value=7.0, value=4.5, step=0.1)
    # st.session_state.user_data["Openness"] = st.number_input("Openness (1-7)", min_value=1.0, max_value=7.0, value=5.0, step=0.1)

    

    st.session_state.user_data["Age"] = st.number_input("Age", min_value=1, max_value=120, value=30)
    st.session_state.user_data["Gender"] = st.selectbox("Gender", ["Male", "Female"])
    st.session_state.user_data["Employment_status"] = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Student", "Unemployed", "Retired", "Other"])
    st.session_state.user_data["Current_emotional_state"] = st.text_input("Current emotional state", "Calm")
    st.session_state.user_data["Have_any_physical_disabilities"] = st.selectbox("Physical disabilities?", ["No", "Yes"])
    st.session_state.user_data["Type_of_physical_activities"] = st.text_input("Physical activities (e.g. Jogging)", "Jogging")
    st.session_state.user_data["How_many_days_do_you_do_exercise"] = st.slider("Days of exercise/week", 0, 7, 3)    
    st.session_state.user_data["Overall_health_status"] = st.selectbox("Overall health", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
    st.session_state.user_data["Current_mobility"] = st.selectbox("Current mobility", ["Poor", "Fair", "Good", "Very Good", "Excellent"])

    st.session_state.user_data["Q26"] = st.number_input("To what extent do you agree with the statement: 'I see myself as reserved and quiet.'?", min_value=1.0, max_value=7.0, value=4.0, step=1.0)
    st.session_state.user_data["Q21"] = st.number_input("To what extent do you agree with the statement: 'I see myself as extraverted and enthusiastic'?", min_value=1.0, max_value=7.0, value=4.0, step=1.0)
    st.session_state.user_data["Extroversion"] = (st.session_state.user_data["Q26"] + st.session_state.user_data["Q21"]) / 2

    st.session_state.user_data["Q22"] = st.number_input("To what extent do you agree with the statement: 'I see myself as critical and quarrelsome'?", min_value=1.0, max_value=7.0, value=4.0, step=1.0)
    st.session_state.user_data["Q27"] = st.number_input("To what extent do you agree with the statement: 'I see myself as sympathetic and warm'?", min_value=1.0, max_value=7.0, value=4.0, step=1.0)
    st.session_state.user_data["Agreeableness"] = (st.session_state.user_data["Q22"] + st.session_state.user_data["Q27"]) / 2

    st.session_state.user_data["Q28"] = st.number_input("To what extent do you agree with the statement: 'I see myself as disorganized and careless'?", min_value=1.0, max_value=7.0, value=4.0, step=1.0)
    st.session_state.user_data["Q23"] = st.number_input("To what extent do you agree with the statement: 'I see myself as dependable and self-disciplined'?", min_value=1.0, max_value=7.0, value=4.0, step=1.0)
    st.session_state.user_data["Conscientiousness"] = (st.session_state.user_data["Q28"] + st.session_state.user_data["Q23"]) / 2


    st.session_state.user_data["Q24"] = st.number_input("To what extent do you agree with the statement: 'I see myself as anxious and easily upset'?", min_value=1.0, max_value=7.0, value=4.0, step=1.0)
    st.session_state.user_data["Q29"] = st.number_input("To what extent do you agree with the statement: 'I see myself as calm and emotionally stable'?", min_value=1.0, max_value=7.0, value=4.0, step=1.0)
    st.session_state.user_data["Emotional_Stability"] = (st.session_state.user_data["Q24"] + st.session_state.user_data["Q29"]) / 2


    st.session_state.user_data["Q30"] = st.number_input("To what extent do you agree with the statement: 'I see myself as conventional, uncreative'?", min_value=1.0, max_value=7.0, value=4.0, step=1.0)
    st.session_state.user_data["Q25"] = st.number_input("To what extent do you agree with the statement: 'I see myself as open to new experiences and complex'?", min_value=1.0, max_value=7.0, value=4.0, step=1.0)
    st.session_state.user_data["Openness"] = (st.session_state.user_data["Q30"] + st.session_state.user_data["Q25"]) / 2

    st.session_state.user_data["question_answer_string"] = st.number_input("User's difficulty rating (1-5) for 10 squats", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
    st.session_state.user_data["question_answer_string2"] = st.number_input("User's difficulty rating (1-5) for 10 calf raises", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
    st.session_state.user_data["question_answer_string3"] = st.number_input("User's difficulty rating (1-5) for 10 standing toe touch", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
    st.session_state.user_data["question_answer_string4"] = st.number_input("User's difficulty rating (1-5) for 10 toe touch stretch", min_value=1.0, max_value=5.0, value=3.0, step=1.0)

import json

# Path where patient JSON records are stored
PATIENT_DATA_DIR = "./patient_records"
os.makedirs(PATIENT_DATA_DIR, exist_ok=True)


def load_patient_data(patient_id):
    filepath = os.path.join(PATIENT_DATA_DIR, f"{patient_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def save_patient_data(patient_id, data):
    filepath = os.path.join(PATIENT_DATA_DIR, f"{patient_id}.json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def show_user_data_2():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.image("assets/assets/fedwell_logo.png", width=200)
    with col2:
        st.markdown("<h1 style='text-align: center;'>Patient Profile</h1>", unsafe_allow_html=True)
    with col3:
        st.image("assets/assets/dfki_logo2.png", width=100)

    st.subheader("Enter Patient ID")
    patient_id = st.text_input("Patient ID")

    # Init empty
    default_data = {}

    # Try loading if patient ID entered
    if patient_id:
        if "loaded_patient_id" not in st.session_state or st.session_state.loaded_patient_id != patient_id:
            with st.spinner("Loading patient record..."):
                time.sleep(5)
                patient_data = load_patient_data(patient_id)
                st.session_state.loaded_patient_id = patient_id
                st.session_state.user_data = patient_data or {}
                st.success("Patient record loaded successfully!" if patient_data else "New patient. Fill in details and save.")

    st.markdown("---")

    # Set up default value loading
    data = st.session_state.get("user_data", {})

    with st.form("user_profile_form"):
        st.markdown("### Demographics")
        age = st.number_input("Age", 1, 120, value=data.get("Age", ))
        gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(data.get("Gender", "Male")))
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Student", "Unemployed", "Retired", "Other"], index=["Employed", "Self-Employed", "Student", "Unemployed", "Retired", "Other"].index(data.get("Employment_status", "Employed")))
        emotional_state = st.text_input("Current emotional state", value=data.get("Current_emotional_state", "Calm"))

        st.markdown("### Physical Health")
        has_disabilities = st.selectbox("Physical disabilities?", ["No", "Yes"], index=["No", "Yes"].index(data.get("Have_any_physical_disabilities", "No")))
        physical_activities = st.text_input("Physical activities (e.g. Jogging)", value=data.get("Type_of_physical_activities", "Jogging"))
        exercise_days = st.slider("Days of exercise/week", 0, 7, value=data.get("How_many_days_do_you_do_exercise", 3))
        health_status = st.selectbox("Overall health", ["Poor", "Fair", "Good", "Very Good", "Excellent"], index=["Poor", "Fair", "Good", "Very Good", "Excellent"].index(data.get("Overall_health_status", "Good")))
        mobility = st.selectbox("Current mobility", ["Poor", "Fair", "Good", "Very Good", "Excellent"], index=["Poor", "Fair", "Good", "Very Good", "Excellent"].index(data.get("Current_mobility", "Good")))

        st.markdown("### Personality (Big Five)")
        q26 = st.number_input("Q26: Reserved and quiet", 1.0, 7.0, value=data.get("Q26", 4.0), step=1.0)
        q21 = st.number_input("Q21: Extraverted and enthusiastic", 1.0, 7.0, value=data.get("Q21", 4.0),step=1.0)
        q22 = st.number_input("Q22: Critical and quarrelsome", 1.0, 7.0, value=data.get("Q22", 4.0),step=1.0)
        q27 = st.number_input("Q27: Sympathetic and warm", 1.0, 7.0, value=data.get("Q27", 4.0),step=1.0)
        q28 = st.number_input("Q28: Disorganized and careless", 1.0, 7.0, value=data.get("Q28", 4.0),step=1.0)
        q23 = st.number_input("Q23: Dependable and self-disciplined", 1.0, 7.0, value=data.get("Q23", 4.0),step=1.0)
        q24 = st.number_input("Q24: Anxious and easily upset", 1.0, 7.0, value=data.get("Q24", 4.0),step=1.0)
        q29 = st.number_input("Q29: Calm and emotionally stable", 1.0, 7.0, value=data.get("Q29", 4.0),step=1.0)
        q30 = st.number_input("Q30: Conventional and uncreative", 1.0, 7.0, value=data.get("Q30", 4.0),step=1.0)
        q25 = st.number_input("Q25: Open to new experiences and complex", 1.0, 7.0, value=data.get("Q25", 4.0),step=1.0)

        st.markdown("### Exercise Difficulty Ratings")
        rating1 = st.number_input("10 squats", 1.0, 5.0, value=data.get("question_answer_string", 3.0))
        rating2 = st.number_input("10 calf raises", 1.0, 5.0, value=data.get("question_answer_string2", 3.0))
        rating3 = st.number_input("10 standing toe touch", 1.0, 5.0, value=data.get("question_answer_string3", 3.0))
        rating4 = st.number_input("10 toe touch stretch", 1.0, 5.0, value=data.get("question_answer_string4", 3.0))

        submitted = st.form_submit_button("💾 Save Profile")

    if submitted:
        updated_data = {
            "Age": age,
            "Gender": gender,
            "Employment_status": employment_status,
            "Current_emotional_state": emotional_state,
            "Have_any_physical_disabilities": has_disabilities,
            "Type_of_physical_activities": physical_activities,
            "How_many_days_do_you_do_exercise": exercise_days,
            "Overall_health_status": health_status,
            "Current_mobility": mobility,
            "Q26": q26, "Q21": q21, "Q22": q22, "Q27": q27,
            "Q28": q28, "Q23": q23, "Q24": q24, "Q29": q29,
            "Q30": q30, "Q25": q25,
            "Extroversion": (q26 + q21) / 2,
            "Agreeableness": (q22 + q27) / 2,
            "Conscientiousness": (q28 + q23) / 2,
            "Emotional_Stability": (q24 + q29) / 2,
            "Openness": (q30 + q25) / 2,
            "question_answer_string": rating1,
            "question_answer_string2": rating2,
            "question_answer_string3": rating3,
            "question_answer_string4": rating4
        }

        st.session_state.user_data = updated_data
        save_patient_data(patient_id, updated_data)
        st.success("✅ Profile saved for patient ID: {}".format(patient_id))

    


    # st.write("Health Profile saved. Switch to **Rehab Corner** in the sidebar to continue.")
    # Last four exercises with emojis
    # Last four exercises with emojis at the far right
    # exercises = [
    #     "squats",
    #     "calf raises",
    #     "standing toe touch",
    #     "toe touch stretch"
    # ]
    
    # for i, exercise in enumerate(exercises, start=1):
    #     col_input, col_emoji = st.columns([9, 1])  # Adjust width for better alignment
    #     with col_input:
    #         rating = st.number_input(f"User's difficulty rating (1-5) for 10 {exercise}", min_value=1.0, max_value=5.0, value=3.0, step=1.0, key=f"rating_{i}")
    #         st.session_state.user_data[f"question_answer_string{i}"] = rating
    #     with col_emoji:
    #         st.markdown(f"<p style='font-size:30px; text-align:right; margin-top:8px;'>{get_exercise_emoji(exercise)}</p>", unsafe_allow_html=True)

    # st.write("Health Profile saved. Switch to **Rehab Corner** in the sidebar to continue.")
    # exercises = [
    #     "squats",
    #     "calf raises",
    #     "standing toe touch",
    #     "toe touch stretch"
    # ]
    
    # for i, exercise in enumerate(exercises, start=1):
    #     col_input, col_emoji = st.columns([10, 1])  # Keep icon aligned
    #     with col_input:
    #         rating = st.number_input(f"User's difficulty rating (1-5) for 10 {exercise}", min_value=1.0, max_value=5.0, value=3.0, step=1.0, key=f"rating_{i}")
    #         st.session_state.user_data[f"question_answer_string{i}"] = rating
    #     with col_emoji:
    #         image_path = get_exercise_icon(exercise)
    #         st.image(image_path, width=35)  # Resize image to match emoji size

    # st.write("Health Profile saved. Switch to **Rehab Corner** in the sidebar to continue.")



# def show_chat():
#     st.title("What can I help you with?")

#     # 1) Display existing messages
#     for msg in st.session_state.messages:
#         if msg["role"] == "user":
#             with st.chat_message("user"):
#                 st.markdown(msg["content"])
#         else:
#             with st.chat_message("assistant"):
#                 st.markdown(msg["content"])  # only 'reason' is stored in content
#                 if "chart_df" in msg and msg["chart_df"] is not None:
#                     st.bar_chart(msg["chart_df"])

#     # 2) Accept new user input at bottom
#     user_input = st.chat_input("Ask a question or say hello...")
#     if user_input:
#         # (a) Show user message
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)

#         prompt_text_main = build_prompt_main(user_input)
#         raw_text_main = question_answer_main(prompt_text_main, max_new_tokens=20)

#         json_dict_main = extract_json_dict(raw_text_main)

#         if json_dict_main["key"] == "SPECIFIC":
#             prompt_text = build_prompt(st.session_state.user_data, st.session_state.messages)

#             # (c) Stream the assistant's reply
#             with st.chat_message("assistant"):
#                 def stream_and_capture():
#                     for chunk in model_stream_generator(prompt_text, user_input):
#                         if chunk.startswith("__STREAMLIT_STREAM_DONE__"):
#                             # final chunk => parse the text
#                             full_text = chunk.replace("__STREAMLIT_STREAM_DONE__", "")
#                             return full_text
#                         else:
#                             yield chunk

#                 # We'll get the final text after streaming
#                 final_output = st.write_stream(stream_and_capture())
                
#                 # (d) Extract JSON, but only display 'reason'
#                 parsed_json = extract_last_assistant_dict(model_response)
#                 parsed_json = ast.literal_eval(parsed_json)

#                 if isinstance(parsed_json, dict):
#                     pass 
#                 else:
#                     parsed_json = {
#                         "score": 0,
#                         "reason": "Please try again.",
#                         "user_score": 0
#                     }
#                 reason_text = parsed_json.get("reason", model_response)  # fallback to entire text if no 'reason'

#                 # (e) Build chart if we have score + user_score
#                 chart_df = None
#                 if "score" in parsed_json and "user_score" in parsed_json:
#                     s = parsed_json["score"]
#                     us = parsed_json["user_score"]
#                     score2 = st.session_state.user_data.get('question_answer_string2', '')
#                     score3 = st.session_state.user_data.get('question_answer_string3', '')
#                     score4 = st.session_state.user_data.get('question_answer_string4', '')
#                     score_map = {
#                         1: "1 - Not difficult",
#                         2: "2 - Slightly difficult",
#                         3: "3 - Moderately difficult",
#                         4: "4 - Very difficult",
#                         5: "5 - Extremely difficult"
#                     }
#                     chart_df = pd.DataFrame({
#                         "Exercise": [
#                             "Model Score",
#                             "Squats",
#                             "Calf Raises",
#                             "Toe Touch",
#                             "Toe Touch Stretch"
#                         ],
#                         "NumericScore": [s, us, score2, score3, score4]
#                     })
#                     chart_df["ScoreLabel"] = chart_df["NumericScore"].map(score_map)

#                 # (f) Save only the 'reason' to session messages
#                 st.session_state.messages.append({
#                     "role": "assistant",
#                     "content": reason_text,  # Only reason
#                     "chart_df": chart_df
#                 })

#                 # (g) Display chart immediately (on first run)
#                 if chart_df is not None:
#                     #st.bar_chart(chart_df)
#                     chart = (
#                         alt.Chart(chart_df)
#                         .mark_bar(size=30)
#                         .encode(
#                             x=alt.X(
#                                 "Exercise:O",
#                                 sort=None,
#                                 axis=alt.Axis(
#                                     labelAngle=0,      # Keep labels horizontal (try 45 if too crowded)
#                                     labelOverlap=False,
#                                     labelPadding=10    # Extra space between label and tick
#                                 )
#                             ),
#                             y=alt.Y("NumericScore:Q", title="Difficulty Score"),
#                             color=alt.Color(
#                                 "ScoreLabel:N",
#                                 legend=alt.Legend(title="Difficulty Level"),
#                                 sort=[
#                                     "1 - Not difficult",
#                                     "2 - Slightly difficult",
#                                     "3 - Moderately difficult",
#                                     "4 - Very difficult",
#                                     "5 - Extremely difficult"
#                                 ]
#                             )
#                         )
#                         # Make the chart wide enough so labels have room
#                         .properties(width=100)
#                     )

#                     st.altair_chart(chart, use_container_width=True)
#         else:
#             prompt_text_general = build_prompt_general(st.session_state.user_data, st.session_state.messages)

#             # (c) Stream the assistant's reply
#             with st.chat_message("assistant"):
#                 def stream_and_capture():
#                     for chunk in model_stream_generator_general(prompt_text_general, user_input):
#                         if chunk.startswith("__STREAMLIT_STREAM_DONE__"):
#                             # final chunk => parse the text
#                             full_text = chunk.replace("__STREAMLIT_STREAM_DONE__", "")
#                             return full_text
#                         else:
#                             yield chunk

#                 # We'll get the final text after streaming
#                 final_output = st.write_stream(stream_and_capture())
                

                

#                 st.session_state.messages.append({
#                     "role": "assistant",
#                     "content": final_output
#                 })



def show_chat():
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        st.image("assets/assets/fedwell_logo.png", use_container_width=False, width=200)
    
    # Middle column left empty (or you could put a title here)
    with col2:
        #st.title("MENTALYTICS")
        # st.markdown("<h1 style='text-align: center;'>Patient Corner</h1>", unsafe_allow_html=True)
        pass  # or st.write("Some title here")
    
    with col3:
        st.image("assets/assets/dfki_logo2.png", use_container_width=False, width=100)
    st.title("What can I help you with?")

    # 1) Display existing messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:  # assistant
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if "chart_df" in msg and msg["chart_df"] is not None:
                    # Build Altair chart from saved DataFrame
                    chart = (
                        alt.Chart(msg["chart_df"])
                        .mark_bar(size=30)
                        .encode(
                            x=alt.X(
                                "Exercise:O",
                                sort=None,
                                axis=alt.Axis(labelAngle=0, labelOverlap=False)
                            ),
                            y=alt.Y("NumericScore:Q", title="Difficulty Score"),
                            color=alt.Color(
                                "ScoreLabel:N",
                                legend=alt.Legend(title="Difficulty Level"),
                                sort=[
                                    "1 - Not difficult",
                                    "2 - Slightly difficult",
                                    "3 - Moderately difficult",
                                    "4 - Very difficult",
                                    "5 - Extremely difficult"
                                ]
                            )
                        )
                        .properties(width=100)
                    )
                    st.altair_chart(chart, use_container_width=True)

    # 2) Accept new user input at bottom
    user_input = st.chat_input("Ask a question or say hello...")
    if user_input:
        # (a) Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # (b) Decide which model: SPECIFIC or GENERAL
        prompt_text_main = build_prompt_main(user_input)
        raw_text_main = question_answer_main(prompt_text_main, max_new_tokens=20)
        json_dict_main = extract_json_dict(raw_text_main)

        # (c) If SPECIFIC, use the finetuned model
        if json_dict_main["key"] == "SPECIFIC":
            prompt_text = build_prompt(st.session_state.user_data, st.session_state.messages)

            with st.chat_message("assistant"):
                def stream_and_capture():
                    for chunk in model_stream_generator(prompt_text, user_input):
                        if chunk.startswith("__STREAMLIT_STREAM_DONE__"):
                            # final chunk => parse the text
                            full_text = chunk.replace("__STREAMLIT_STREAM_DONE__", "")
                            return full_text
                        else:
                            yield chunk

                final_output = st.write_stream(stream_and_capture())
                
                # Parse JSON from model output
                parsed_json = extract_last_assistant_dict(model_response)
                if not parsed_json:
                    parsed_json = '{"score": 0, "reason": "Please try again.", "user_score": 0}'
                parsed_json = ast.literal_eval(parsed_json)

                # Fallback if it's not a dict
                if not isinstance(parsed_json, dict):
                    parsed_json = {
                        "score": 0,
                        "reason": "Please try again.",
                        "user_score": 0
                    }

                reason_text = parsed_json.get("reason", "No reason provided")

                # (d) Build a DataFrame for the chart
                s = parsed_json.get("score", 0)
                us = parsed_json.get("user_score", 0)
                score2 = st.session_state.user_data.get("question_answer_string2", 0)
                score3 = st.session_state.user_data.get("question_answer_string3", 0)
                score4 = st.session_state.user_data.get("question_answer_string4", 0)

                score_map = {
                    1: "1 - Not difficult",
                    2: "2 - Slightly difficult",
                    3: "3 - Moderately difficult",
                    4: "4 - Very difficult",
                    5: "5 - Extremely difficult"
                }

                chart_df = pd.DataFrame({
                    "Exercise": [
                        #"Model Score",
                        "Squats",
                        "Calf Raises",
                        "Toe Touch",
                        "Toe Touch Stretch"
                    ],
                    #"NumericScore": [s, us, score2, score3, score4]
                    "NumericScore": [us, score2, score3, score4]
                })
                chart_df["ScoreLabel"] = chart_df["NumericScore"].map(score_map)

                # (e) Save the assistant reply & chart
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reason_text, 
                    "chart_df": chart_df
                })

                # (f) Immediately render the chart for this response
                chart = (
                    alt.Chart(chart_df)
                    .mark_bar(size=30)
                    .encode(
                        x=alt.X(
                            "Exercise:O",
                            sort=None,
                            axis=alt.Axis(labelAngle=0, labelOverlap=False)
                        ),
                        y=alt.Y("NumericScore:Q", title="Difficulty Score"),
                        color=alt.Color(
                            "ScoreLabel:N",
                            legend=alt.Legend(title="Difficulty Level"),
                            sort=[
                                "1 - Not difficult",
                                "2 - Slightly difficult",
                                "3 - Moderately difficult",
                                "4 - Very difficult",
                                "5 - Extremely difficult"
                            ]
                        )
                    )
                    .properties(width=100)
                )

                st.altair_chart(chart, use_container_width=True)

        # (g) Otherwise, use GENERAL model
        else:
            prompt_text_general = build_prompt_general(st.session_state.user_data, st.session_state.messages)

            with st.chat_message("assistant"):
                def stream_and_capture():
                    for chunk in model_stream_generator_general(prompt_text_general, user_input):
                        if chunk.startswith("__STREAMLIT_STREAM_DONE__"):
                            full_text = chunk.replace("__STREAMLIT_STREAM_DONE__", "")
                            return full_text
                        else:
                            yield chunk

                final_output = st.write_stream(stream_and_capture())

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_output
                })

def show_physio_corner():
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        st.image("assets/assets/fedwell_logo.png", use_container_width=False, width=200)
    
    with col2:
        st.markdown("<h1 style='text-align: center;'>Physio Corner</h1>", unsafe_allow_html=True)
        #pass
    
    with col3:
        st.image("assets/assets/dfki_logo2.png", use_container_width=False, width=100)

    #st.title("Physio Corner")

    st.subheader("Patient Information & Overall Health Status")
    # Get user data from session state
    user_data = st.session_state.get("user_data", {})

    # Create a 3-column layout for the patient details
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Age**: {user_data.get('Age', 'N/A')}")
        st.write(f"**Gender**: {user_data.get('Gender', 'N/A')}")
        st.write(f"**Employment Status**: {user_data.get('Employment_status', 'N/A')}")

    with col2:
        st.write(f"**Physical Disabilities**: {user_data.get('Have_any_physical_disabilities', 'N/A')}")
        st.write(f"**Physical Activities**: {user_data.get('Type_of_physical_activities', 'N/A')}")
        st.write(f"**Days of Exercise/Week**: {user_data.get('How_many_days_do_you_do_exercise', 'N/A')}")

    with col3:
        st.write(f"**Overall Health Status**: {user_data.get('Overall_health_status', 'N/A')}")
        st.write(f"**Current Mobility**: {user_data.get('Current_mobility', 'N/A')}")
        st.write(f"**Current Emotional State**: {user_data.get('Current_emotional_state', 'N/A')}")

    st.title(" ")

    # Get scores from session state (user_data)
    user_data = st.session_state.get("user_data", {})
    model_score = user_data.get("question_answer_string", 0)  # You can customize or set to 0
    score2 = user_data.get("question_answer_string2", 0)
    score3 = user_data.get("question_answer_string3", 0)
    score4 = user_data.get("question_answer_string4", 0)

    score_map = {
        1: "1 - Not difficult",
        2: "2 - Slightly difficult",
        3: "3 - Moderately difficult",
        4: "4 - Very difficult",
        5: "5 - Extremely difficult"
    }

    chart_df = pd.DataFrame({
        "Exercise": [
            "Squats",
            "Calf Raises",
            "Toe Touch",
            "Toe Touch Stretch"
        ],
        "NumericScore": [model_score, score2, score3, score4]
    })
    chart_df["ScoreLabel"] = chart_df["NumericScore"].map(score_map)

    

    chart = (
        alt.Chart(chart_df)
        .mark_bar(size=30)
        .encode(
            x=alt.X("Exercise:O", sort=None, axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("NumericScore:Q", title="Numeric Rating Scale (NRS) Scale"),
            color=alt.Color(
                "ScoreLabel:N",
                legend=alt.Legend(title="Difficulty Level"),
                sort=[
                    "1 - Not difficult",
                    "2 - Slightly difficult",
                    "3 - Moderately difficult",
                    "4 - Very difficult",
                    "5 - Extremely difficult"
                ]
            )
        )
        .properties(width=500)
    )

    

    def render_personality_graph(user_data):
    # Trait names and general norms
        traits = [
            "Extroversion", 
            "Agreeableness", 
            "Conscientiousness", 
            "Emotional Stability", 
            "Openness"
        ]
        general_norms = {
            "Extroversion": 4.44,
            "Agreeableness": 5.23,
            "Conscientiousness": 5.4,
            "Emotional Stability": 4.83,
            "Openness": 5.38
        }

        # Get user scores
        user_scores = [user_data.get(trait.replace(" ", "_"), 0) for trait in traits]

        # Build dataframe
        data = []
        for trait, score in zip(traits, user_scores):
            data.append({"Trait": trait, "Group": "User", "Score": score})
        for trait in traits:
            data.append({"Trait": trait, "Group": "General Norm", "Score": general_norms[trait]})

        df = pd.DataFrame(data)

        

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Trait:N",
                    title=None,
                    axis=alt.Axis(labelAngle=0),
                    scale=alt.Scale(paddingInner=0.2, paddingOuter=0.1)
                ),
                y=alt.Y(
                    "Score:Q",
                    title="Score (out of 7)",
                    scale=alt.Scale(domain=[0, 7])
                ),
                # Sort ensures User bars appear first (on the left), then General Norm
                xOffset=alt.XOffset("Group:N", sort=["User", "General Norm"]),
                color=alt.Color(
                    "Group:N",
                    # The domain order also ensures User is first, General Norm is second
                    scale=alt.Scale(
                        domain=["User", "General Norm"],
                        range=["#81D4FA", "#FFB74D"]
                    ),
                    legend=alt.Legend(title="Legend")
                ),
                tooltip=["Trait", "Group", "Score"]
            )
            # Step sizing to control overall width
            .properties(width=alt.Step(60), height=300)
        )

        st.altair_chart(chart, use_container_width=True)
    
    # Create a 3-column layout for the patient details
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Anticipated Pain/Difficulty")
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.subheader("Patient Personality Traits & Insights")
        render_personality_graph(st.session_state.user_data)


    

    def render_personality_text(user_data):
        # Format personality summary
        summary = f"""
    #### What each trait means:
    - **Extroversion**: Sociability and enthusiasm.
    - **Agreeableness**: Cooperation and compassion.
    - **Conscientiousness**: Reliability and self-discipline.
    - **Emotional Stability**: Composure and resilience.
    - **Openness**: Curiosity and willingness to explore.

    ---
    #### Patient Insights:
    **Socially balanced**, **moderately cooperative**, **reasonably disciplined**, **emotionally steady**, and **open to trying new exercises**, though need occasional motivation and structured guidance to stay consistent with the rehabilitation plan.

        """
        st.markdown(summary)

    def render_Exercise_text(user_data):
        # Format personality summary
        summary = f"""
    #### Model Reasoning:
    The user is young, active, and in good health with no physical disabilities. They engage in frequent exercise, including walking, gym workouts, cycling, swimming, and team sports 5-6 days a week. Performing 10 squats is easy for their fitness level. While they perceive their toe-touch ability as low (2), their actual capability is likely higher. However, due to their medical history, they may experience difficulty with the toe-touch stretch.

        """
        st.markdown(summary)



    # Then call them in Physio Corner:
    #st.subheader("Personality Profile (Big Five)")
    col1, col2 = st.columns(2)

    with col1:
        render_Exercise_text(st.session_state.user_data)

    with col2:
        render_personality_text(st.session_state.user_data)

    
    

def main():
    global tokenizer1, model1, tokenizer2, model2 #remove later

    #st.sidebar.title("Navigation")
    pages = ["Mentalytics Overview", "Patient Profiles","Physio Corner", "AI Assistant"]
    choice = st.sidebar.radio("", pages)

    if choice == "Mentalytics Overview":
        show_home()
    elif choice == "Patient Profiles":
        show_user_data_2()
    elif choice == "AI Assistant":
        # Only load models here
        tokenizer1, model1 = load_model() #remove later 
        show_chat()
        #show_chat()
    elif choice == "Physio Corner":
        show_physio_corner()




if __name__ == "__main__":
    main()
