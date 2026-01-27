from typing import TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from IPython.display import Image, display
from TTS.api import TTS
from simpleaudio import WaveObject
from dotenv import load_dotenv
from os import sync
import json
import random

QUESTION_BANKS_FILEPATH = "questions.json"
SPEAKER_FILEPATH = "output.wav"

ASK_QUESTIONS: str = (
    "Identity: You are an interviewer named Lucas for the US citizenship interview at the USCIS.\n\n"
    "Role:\n"
    "Conduct the interview.\n\n"
    "You will be given:\n"
    "- Interviewee's name\n"
    "- Current question's index\n"
    "- Current question\n"
    "- Previous question could be None\n"
    "- Interviewee's answer to previous question (could be None)\n"
    "- Correct answer to previous question (could be None)\n"
    "- Did they get previous answer right? (could be None)\n\n"
    "Rules:\n"
    "- If this is the first question (question's index = 0), then\n"
    "       - Greet the interviewee, introduce yourself.\n" 
    "       - Ask the question.\n"
    "- If this is the second question onward (question's index > 0), then\n"
    "       - Continue the interview without any greeting, welcome"
    "       - Give short feedback on the previous question, guide:\n"
    "           - If they got it right, say that their answer is correct.\n"
    "           - Otherwise, compare their question with correct one (using previous question if you want) in ONE or TWO sentences.\n"
    "       - Move on and ask the question.\n"
    "- You can paraphrase the question for slight challenge, without changing meaning.\n"
    "- Be professional, and human natural.\n"
    "- Speak directly to the interviewee.\n"
    "- Do NOT add any information beyond what is required.\n"
    "- Do NOT add any hint.\n"
)


CONCLUDE_INTERVIEW = (
    "Identity: You are an interviewer named Lucas for the US citizenship interview at the USCIS.\n\n"
    "Role:\n"
    "Announce the interview result and conclude.\n\n"
    "You will be given:\n"
    "- Interviewee's name\n"
    "- Do they pass interview?\n"
    "- Number of correct answers\n\n"
    "Rules:\n"
    "- Tell the interviewee how many questions they answered correctly and if they passed the interview or not.\n"
    "- If the interviewee passed, then\n"
    "       - Congratulate them on passing.\n"
    "       - Tell them to wait for schedule of ceremony.\n"
    "- If the interviewee failed, then\n"
    "       - Encourage them to do better next time.\n"
    "       - Be motivating, and empathetic.\n"
    "- End with a short farewell.\n"
    "- Be concise and professional but human natural.\n"
    "- Speak directly to the interviewee.\n"
    "- Do NOT ask any questions.\n"
)


EVALUATE_ANSWER = (
    "Identity: You are a strict grader for a U.S. citizenship interview.\n\n"
    "You will be given:\n"
    "- Question"
    "- Interviewee's answer"
    "- Correct answer\n\n"
    "Rules:\n"
    "- Compare the interviewee's answer to the correct answer.\n"
    "- Accept paraphrases and synonyms.\n"
    "- Reject answers that are factually incorrect or incomplete.\n"
    "- Do NOT explain your reasoning.\n"
    "- Do NOT add any extra text.\n\n"
    "Output format:\n"
    "Respond with exactly one word: YES or NO\n"
)


# DEFINE THE GRAPH
class Question(TypedDict):
    content: str
    answer: str

class InterviewState(TypedDict):
    """State of the interview during the process"""
    name: str
    # Initialized during setup
    questions: Optional[list[Question]]
    cur_index: Optional[int]
    num_correct: Optional[int]

    # Update during interview
    prev_answer: Optional[str]
    prev_correct_answer: Optional[str]
    prev_answer_eval: Optional[bool]
    pass_interview: Optional[bool]


load_dotenv()
# Initialize the LLM and TTS model
print("Init the LLM and TTS...")
gemini_flash_lite = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite", 
    temperature=0.5, 
    max_retries=2
)
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True, 
    gpu=False
)
print("Init the LLM and TTS successfully!")

# TODO: Fix the segmentation fault that happens
def speak(content: str) -> None:
    """Agent speaks the content"""
    # Convert the content to the speech file
    try:
        # Print for testing
        print(f"Interviewer says: {content}")
        tts.tts_to_file(
            text=content,
            speaker=tts.speakers[0],
            language="en",
            file_path=SPEAKER_FILEPATH
        )
        
        sync()

        # Execute speech file
        wave_obj = WaveObject.from_wave_file(SPEAKER_FILEPATH)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(e)
        raise RuntimeError("Lucas has a problem speaking")
    

def convert_to_text() -> str:
    return ""


# DEFINE NODES
def setup_interview(state: InterviewState) -> InterviewState:
    """Setup the interview"""
    # Load memory from file
    state["questions"] = []
    try:
        with open(QUESTION_BANKS_FILEPATH, "r") as f:
            question_banks: list[dict[str, str]] = json.load(f)
            indices: list[int] = random.sample(
                range(0, len(question_banks)), 
                10
            )

            for index in indices:
                state["questions"].append(question_banks[index])
    except FileNotFoundError:
        raise RuntimeError("Questions' source doesn't exist")
    except json.JSONDecodeError:
        raise RuntimeError("Questions are stored in invalid format")

    # Initialize default values for the state
    state["cur_index"] = 0
    state["num_correct"] = 0
    state["prev_answer"] = None
    state["prev_answer_eval"] = None
    state["prev_correct_answer"] = None

    return state


def ask_questions(state: InterviewState) -> InterviewState:
    """LLM to conduct interview and ask next question"""
    print(json.dumps(state, indent=2))

    system_prompt = SystemMessage(
        content=ASK_QUESTIONS
    )
    index = state["cur_index"]
    prev_q = (
        state["questions"][index-1]["content"] if index > 0
        else None
    )
    state_prompt = HumanMessage(
        content=(
            f"""Interviewee's name: {state["name"]},"""
            f"""Current question's index: {index},"""
            f"""Current question: {state["questions"][index]["content"]},"""
            f"""Previous question: {prev_q},"""
            f"""Interviewee's answer to previous question: {state["prev_answer"]},"""
            f"""Correct answer to previous question: {state["prev_correct_answer"]},"""
            f"""Did they get previous answer right?: {state["prev_answer_eval"]}."""
        )
    )
    response = gemini_flash_lite.invoke([system_prompt, state_prompt])
    print(response.content)
    return state


def wait_user_response(state: InterviewState) -> InterviewState:
    """
    Wait for the user's response.
    """
    # The answer of current question is saved to prev_answer
    state["prev_answer"] = input("Enter your answer here: ")
    return state


def evaluate_answers(state: InterviewState) -> InterviewState:
    """Evaluate the user's answers"""
    system_prompt = SystemMessage(
        content=EVALUATE_ANSWER
    )
    question = state["questions"][state["cur_index"]]
    state_prompt = HumanMessage(
        content=(
            f"""Question: {question["content"]},"""
            f"""Interviewee's answer: {state["prev_answer"]},"""
            f"""Correct answer: {question["answer"]}."""
        )
    )
    response = gemini_flash_lite.invoke([system_prompt, state_prompt])

    state["prev_correct_answer"] = question["answer"]
    if response.content == "YES":
        state["prev_answer_eval"] = True
        state["num_correct"] += 1
    else:
        state["prev_answer_eval"] = False

    # Move to next question
    state["cur_index"] += 1
    if state["num_correct"] >= 6:
        # Answering at least 6 questions correctly means passing
        state["pass_interview"] = True

    return state


def should_continue_interview(state: InterviewState) -> InterviewState:
    """
    Conditional node:
    Determine if agent continues interview after evaluation.
    """
    if state["num_correct"] < 6:
        num_q_left: int = len(state["questions"]) - state["cur_index"] 
        if state["num_correct"] + num_q_left >= 6:
            # There are enough questions left, user still has a chance to pass
            return "continue_interview"
        else:
            # User is cooked
            state["pass_interview"] = False
            return "end_interview"

    return "end_interview"


def conclude_interview(state: InterviewState) -> InterviewState:
    """Conclude the interview with results reporting"""
    system_prompt = SystemMessage(
        content=CONCLUDE_INTERVIEW
    )
    state_prompt = HumanMessage(
        content=(
            f"""Interviewee's name: {state["name"]}"""
            f"""Do they pass interview?: {state["pass_interview"]},"""
            f"""Number of correct answers: {state["num_correct"]}."""
        )
    )
    response = gemini_flash_lite.invoke([system_prompt, state_prompt])
    print(response.content)

    return state


# CONSTRUCT THE AGENT'S GRAPH (FLOW OF THE INTERVIEW)
def construct_graph():
    builder = StateGraph(InterviewState)

    builder.add_node("setup_interview", setup_interview)
    builder.add_node("ask_question", ask_questions)
    builder.add_node("wait_response", wait_user_response)
    builder.add_node("judge_answer", evaluate_answers)
    builder.add_node("conclude_interview", conclude_interview)

    builder.set_entry_point("setup_interview")
    builder.add_edge("setup_interview", "ask_question")
    builder.add_edge("ask_question", "wait_response")
    builder.add_edge("wait_response", "judge_answer")

    # Interview loop
    builder.add_conditional_edges(
        "judge_answer",
        should_continue_interview,
        {
            "continue_interview": "ask_question",
            "end_interview": "conclude_interview"
        }
    )
    builder.set_finish_point("conclude_interview")

    return builder.compile()


def visualize_graph(compiled_graph) -> None:
    """Draw the interview's workflow represented by graph"""
    display(Image(compiled_graph.get_graph().draw_mermaid_png()))


def conduct_interview(compiled_graph, interviewee_name: str) -> None:
    """Run the interview"""
    compiled_graph.invoke({
        "name": interviewee_name
    })

if __name__ == "__main__":
    try:
        compiled_graph = construct_graph()
        # visualize_graph(compiled_graph)

        # Start running the interview
        conduct_interview(
            compiled_graph, interviewee_name=input("Enter you name: ")
        )
    except Exception as e:
        print("INTERVIEW_INTERRUPTION!")
        print(e)
