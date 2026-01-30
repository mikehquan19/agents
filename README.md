# 2 BOTS: FLIGHT AND HOTEL BOOKING BOT, AND CITIZENSHIP INTERVIEW BOT

## CITIZENSHIP INTERVIEW BOT
### Design
State:
```
class AgentState(TypedDict):
    questions: list[dict[str, str]] # [{content: "", answer: ""}]
    next_index: int # Index of the next question to ask
    num_correct: int # Number of correct questions user got right
    pass_interview: Optional[bool] # If user passed the interview
```

- Here we have 2 LLM nodes:
    - Node ```conduct_interview```, including:
        - Welcome the user, introduce itself, and give first question
        - Give feedback after every question and ask the next one.

    - Node ```conclude_interview``` by telling if user passed the question or not, how many question the user has answered correctly. Congratulate if they passed and encourage them if they failed.

    - Node ```evaluate_answers```: Evaluate the the user's answer against correct answer in state using just another llm.
    Use the conditional node:
    And update states (number of correct answers and passing status)
    - If the user has passed or failed, move to LLM that concludes
    - If interview is still going on, go back to LLM that asks questions

- Node ```setup_interview```: Initialize some of the default value for the state. Query from the database (file for prototype) the list of 10 random questions and their correct answers.

- Node ```wait_user_response```: Record (or prompt input of user for prototype) user's response and save it to the state's ```last_answer```.


## FLIGHT AND HOTEL BOOKING BOT
- Search for flights and hotels given location and time from user
- Search for any advisory information attached to selected flight
- Recommend user on they
- Book the flight or hotel for user