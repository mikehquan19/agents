import os
import json
import plaid
from plaid.api.plaid_api import PlaidApi
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from fastapi import FastAPI
from pydantic import BaseModel


TOKEN_FILEPATH = "credentials.json"

class PlaidService:
    """
    The services that processes and stores the user id and list of access tokens.
    For prototype purposes, this only works for one user and is not scalable.
    """
    def __init__(self, env_type: str, user_id: str) -> None:
        configuration = plaid.Configuration(
            host=(
                plaid.Environment.Sandbox if env_type == "sandbox"
                else plaid.Environment.Production
            ),
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

    def create_link_token(self) -> str:
        """Generate the link token from Plaid client to pass to frontend"""
        request = LinkTokenCreateRequest(
            products=[
                Products('auth'),
                Products('transactions'),
            ],
            client_name="Personal Finance Agent",
            country_codes=[
                CountryCode('US')
            ],
            language='en',
            user=LinkTokenCreateRequestUser(
                client_user_id=self.user_id
            ),
        )

        response = self.client.link_token_create(request)
        print(response)
        return response['link_token']

    def exchange_for_access_token(self, public_token: str) -> None:
        """Generate the access token from public token"""
        request = ItemPublicTokenExchangeRequest(
            public_token=public_token
        )

        response = self.client.item_public_token_exchange(request)
        print(response)
        self.access_tokens.append(response["access_token"])

        # For now, just save to file.
        # TODO: Find more secure and scalable way later
        with open(TOKEN_FILEPATH, "w") as f:
            json.dump(
                {self.user_id: self.access_tokens}, 
                f
            )


app = FastAPI()
plaid_service = PlaidService("sandbox", "1000")

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