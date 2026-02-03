import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
import plaid
from plaid.api.plaid_api import PlaidApi
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.transactions_get_request import TransactionsGetRequest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
TOKEN_FILEPATH = "credentials.json"

class PlaidService:
    """Service that processes and stores the user id and list of access tokens"""
    def __init__(self, user_id: str) -> None:
        configuration = plaid.Configuration(
            host=plaid.Environment.Sandbox,
            api_key={
                "clientId": os.getenv("CLIENT_ID"),
                "secret": os.getenv("CLIENT_SECRET"),
            }
        )
        api_client = plaid.ApiClient(configuration=configuration)
        
        self.client = PlaidApi(api_client)
        self.user_id = user_id
        self.access_tokens = []

        if os.path.exists(TOKEN_FILEPATH) and os.path.getsize(TOKEN_FILEPATH) > 0:
            with open(TOKEN_FILEPATH, "r") as f:
                credentials: dict[str, list[str]] = json.load(f)
                if self.user_id in credentials:
                    self.access_tokens = credentials[self.user_id]

    # METHODS FOR AUTH 

    def create_link_token(self) -> str:
        """Generate the link token from Plaid client to pass to client"""
        link_request = LinkTokenCreateRequest(
            products=[
                Products('auth'),
                Products('transactions'),
            ],
            client_name="Personal Finance Agent",
            country_codes=[
                CountryCode('US')
            ],
            language='en',
            user=LinkTokenCreateRequestUser(client_user_id=self.user_id),
        )

        link_response = self.client.link_token_create(link_request)
        print(link_response)
        return link_response['link_token']

    def exchange_for_access_token(self, public_token: str) -> None:
        """Generate the access token from public token"""
        exchange_request = ItemPublicTokenExchangeRequest(
            public_token=public_token
        )
        exchange_response = self.client.item_public_token_exchange(exchange_request)
        self.access_tokens.append(exchange_response["access_token"])

        with open(TOKEN_FILEPATH, "w") as f:
            json.dump({self.user_id: self.access_tokens}, f)

        print(exchange_response)

    # METHODS FOR GETTING THE FINANCIAL DATA

    def get_accounts(self) -> list:
        """Get the list of accounts of the user's institution"""
        accounts = []
        for acc_tok in self.access_tokens:
            # Get the list of account of each of the user's login institution
            acc_request = AccountsGetRequest(
                access_token=acc_tok
            )
            response = self.client.accounts_get(acc_request)
            # Convert the Plaid's base obj into a dict to be serialized
            accounts.extend(response.to_dict()["accounts"])

        return accounts
    
    def get_transactions_between(self, start_date: str, end_date: str) -> list:
        "Get list of latest transactions of the user"
        transactions = []

        start=datetime.strptime(start_date, '%Y-%m-%d').date()
        end=datetime.strptime(end_date, '%Y-%m-%d').date()

        for acc_tok in self.access_tokens:
            request = TransactionsGetRequest(
                access_token=acc_tok,
                start_date=start, end_date=end
            )

            response = self.client.transactions_get(request).to_dict()
            transactions += response["transactions"]

            # Since transactions are paginated, call more until everything is added
            while len(transactions) < response["total_transactions"]:
                request = TransactionsGetRequest(
                    access_token=acc_tok,
                    start_date=start, end_date=end,
                    options=TransactionsGetRequestOptions(
                        offset=len(transactions)
                    )
                )

                response = self.client.transactions_get(request).to_dict()
                transactions += response["transactions"]

        return transactions

app = FastAPI()
plaid_service = PlaidService("1000")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def check_health():
    return {"message": "ok"}


@app.post("/link_token")
def generate_link_token():
    try:
        link_token = plaid_service.create_link_token()
    except Exception as e:
        print(f"Unable to create the link token\n{e}")
        return {
            "status": "ERROR",
            "message": "Unable to create the link token"
        }
    
    return {
        "status": "OK",
        "link_token": link_token
    }


class ExchangeObj(BaseModel):
    public_token: str


@app.post("/exchange")
def exchange_public_for_access(exchange: ExchangeObj):
    try:
        plaid_service.exchange_for_access_token(exchange.public_token)
    except Exception as e:
        print(f"Unable to exchange public for access token\n{e}")
        return {
            "status": "ERROR",
            "message": "Unable to exchange public for access token"
        }
    
    return {
        "status": "OK",
        "message": "Access token generated!"
    }

@app.get("/accounts")
def get_accounts():
    try:
        accounts = plaid_service.get_accounts()
    except Exception as e:
        print(f"Unable to get the list of accounts\n{e}")
        return {
            "status": "ERROR",
            "message": "Unable get accounts"
        }
    
    return {
        "status": "OK",
        "data": accounts
    }

@app.get("/transactions")
def get_transactions():
    try:
        transactions = plaid_service.get_transactions_between(
            "2026-01-01", "2026-02-02"
        )
    except Exception as e:
        print(f"Unable to get the list of transactions\n{e}")
        return {
            "status": "ERROR",
            "message": "Unable get transactions"
        }
    
    return {
        "status": "OK",
        "data": transactions
    }
