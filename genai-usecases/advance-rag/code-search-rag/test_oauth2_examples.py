"""
OAuth2 Authentication Examples
================================
This module provides comprehensive OAuth2 authentication implementations
for various use cases including web applications, mobile apps, and APIs.

Author: Test Module
License: MIT
"""

import requests
import base64
import hashlib
import secrets
from urllib.parse import urlencode, parse_qs
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import json


# =============================================================================
# 1. BASIC OAUTH2 AUTHORIZATION CODE FLOW
# =============================================================================

class OAuth2Client:
    """
    OAuth2 client implementing the Authorization Code flow.

    This is the most common OAuth2 flow used by web applications.
    """

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str,
                 auth_url: str, token_url: str):
        """
        Initialize OAuth2 client.

        Args:
            client_id: OAuth2 client identifier
            client_secret: OAuth2 client secret
            redirect_uri: URL where user will be redirected after authorization
            auth_url: Authorization endpoint URL
            token_url: Token endpoint URL
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_url = auth_url
        self.token_url = token_url
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None

    def get_authorization_url(self, scope: str = "read write",
                             state: Optional[str] = None) -> str:
        """
        Generate authorization URL for redirecting user.

        Args:
            scope: Space-separated list of requested permissions
            state: Random string for CSRF protection

        Returns:
            Authorization URL
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': scope,
            'state': state
        }

        return f"{self.auth_url}?{urlencode(params)}"

    def exchange_code_for_token(self, authorization_code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.

        Args:
            authorization_code: Code received from authorization server

        Returns:
            Token response containing access_token, refresh_token, etc.

        Raises:
            requests.HTTPError: If token exchange fails
        """
        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        response = requests.post(self.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()

        # Store tokens
        self.access_token = token_data['access_token']
        self.refresh_token = token_data.get('refresh_token')

        # Calculate expiry time
        if 'expires_in' in token_data:
            self.token_expiry = datetime.now() + timedelta(seconds=token_data['expires_in'])

        return token_data

    def refresh_access_token(self) -> Dict[str, Any]:
        """
        Refresh the access token using refresh token.

        Returns:
            New token response

        Raises:
            ValueError: If refresh token is not available
            requests.HTTPError: If token refresh fails
        """
        if not self.refresh_token:
            raise ValueError("No refresh token available")

        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        response = requests.post(self.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()

        # Update tokens
        self.access_token = token_data['access_token']
        if 'refresh_token' in token_data:
            self.refresh_token = token_data['refresh_token']

        if 'expires_in' in token_data:
            self.token_expiry = datetime.now() + timedelta(seconds=token_data['expires_in'])

        return token_data

    def is_token_expired(self) -> bool:
        """Check if access token is expired."""
        if not self.token_expiry:
            return True
        return datetime.now() >= self.token_expiry

    def make_authenticated_request(self, url: str, method: str = "GET",
                                   **kwargs) -> requests.Response:
        """
        Make an authenticated API request.

        Args:
            url: API endpoint URL
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional arguments for requests

        Returns:
            Response object
        """
        # Refresh token if expired
        if self.is_token_expired() and self.refresh_token:
            self.refresh_access_token()

        headers = kwargs.pop('headers', {})
        headers['Authorization'] = f'Bearer {self.access_token}'

        return requests.request(method, url, headers=headers, **kwargs)


# =============================================================================
# 2. PKCE (Proof Key for Code Exchange) - FOR MOBILE/SPA
# =============================================================================

class OAuth2PKCEClient:
    """
    OAuth2 client with PKCE extension for public clients.

    PKCE is recommended for mobile apps and single-page applications
    where client secret cannot be securely stored.
    """

    def __init__(self, client_id: str, redirect_uri: str,
                 auth_url: str, token_url: str):
        """
        Initialize OAuth2 PKCE client.

        Args:
            client_id: OAuth2 client identifier
            redirect_uri: Redirect URI after authorization
            auth_url: Authorization endpoint URL
            token_url: Token endpoint URL
        """
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.auth_url = auth_url
        self.token_url = token_url
        self.code_verifier = None
        self.code_challenge = None

    def generate_pkce_pair(self) -> Tuple[str, str]:
        """
        Generate PKCE code verifier and challenge.

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        # Generate code verifier (random string)
        self.code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')

        # Generate code challenge (SHA256 hash of verifier)
        challenge_bytes = hashlib.sha256(self.code_verifier.encode('utf-8')).digest()
        self.code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')

        return self.code_verifier, self.code_challenge

    def get_authorization_url(self, scope: str = "read write") -> str:
        """
        Generate authorization URL with PKCE parameters.

        Args:
            scope: Requested permissions

        Returns:
            Authorization URL with PKCE challenge
        """
        if not self.code_challenge:
            self.generate_pkce_pair()

        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': scope,
            'code_challenge': self.code_challenge,
            'code_challenge_method': 'S256'
        }

        return f"{self.auth_url}?{urlencode(params)}"

    def exchange_code_for_token(self, authorization_code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for token using PKCE verifier.

        Args:
            authorization_code: Authorization code from callback

        Returns:
            Token response
        """
        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.client_id,
            'code_verifier': self.code_verifier
        }

        response = requests.post(self.token_url, data=data)
        response.raise_for_status()

        return response.json()


# =============================================================================
# 3. CLIENT CREDENTIALS FLOW - FOR SERVICE-TO-SERVICE
# =============================================================================

def authenticate_client_credentials(client_id: str, client_secret: str,
                                    token_url: str, scope: str = "") -> Dict[str, Any]:
    """
    Authenticate using OAuth2 Client Credentials flow.

    This flow is used for server-to-server authentication where
    no user interaction is required.

    Args:
        client_id: OAuth2 client identifier
        client_secret: OAuth2 client secret
        token_url: Token endpoint URL
        scope: Requested permissions (optional)

    Returns:
        Token response with access_token

    Example:
        >>> token_data = authenticate_client_credentials(
        ...     client_id="my-service",
        ...     client_secret="secret-key",
        ...     token_url="https://auth.example.com/oauth/token",
        ...     scope="api.read api.write"
        ... )
        >>> access_token = token_data['access_token']
    """
    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }

    if scope:
        data['scope'] = scope

    response = requests.post(token_url, data=data)
    response.raise_for_status()

    return response.json()


# =============================================================================
# 4. RESOURCE OWNER PASSWORD FLOW (Legacy - Not Recommended)
# =============================================================================

def authenticate_with_password(username: str, password: str,
                               client_id: str, client_secret: str,
                               token_url: str, scope: str = "") -> Dict[str, Any]:
    """
    Authenticate using Resource Owner Password Credentials flow.

    WARNING: This flow is deprecated and should only be used for
    legacy applications. Use Authorization Code flow instead.

    Args:
        username: User's username
        password: User's password
        client_id: OAuth2 client identifier
        client_secret: OAuth2 client secret
        token_url: Token endpoint URL
        scope: Requested permissions

    Returns:
        Token response
    """
    data = {
        'grant_type': 'password',
        'username': username,
        'password': password,
        'client_id': client_id,
        'client_secret': client_secret
    }

    if scope:
        data['scope'] = scope

    response = requests.post(token_url, data=data)
    response.raise_for_status()

    return response.json()


# =============================================================================
# 5. OAUTH2 TOKEN VALIDATOR
# =============================================================================

class OAuth2TokenValidator:
    """
    Validate and decode OAuth2 access tokens.

    Supports both opaque tokens (via introspection) and JWT tokens.
    """

    def __init__(self, introspection_url: str, client_id: str, client_secret: str):
        """
        Initialize token validator.

        Args:
            introspection_url: Token introspection endpoint
            client_id: Client identifier
            client_secret: Client secret
        """
        self.introspection_url = introspection_url
        self.client_id = client_id
        self.client_secret = client_secret

    def validate_token(self, access_token: str) -> Dict[str, Any]:
        """
        Validate an access token via introspection endpoint.

        Args:
            access_token: Token to validate

        Returns:
            Token metadata including active status, scopes, expiry, etc.

        Example:
            >>> validator = OAuth2TokenValidator(
            ...     introspection_url="https://auth.example.com/oauth/introspect",
            ...     client_id="client-id",
            ...     client_secret="client-secret"
            ... )
            >>> result = validator.validate_token("access-token-here")
            >>> if result['active']:
            ...     print(f"Token is valid with scopes: {result['scope']}")
        """
        # Create Basic Auth header
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = base64.b64encode(auth_string.encode('utf-8'))
        auth_header = f"Basic {auth_bytes.decode('utf-8')}"

        headers = {
            'Authorization': auth_header,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        data = {'token': access_token}

        response = requests.post(self.introspection_url, headers=headers, data=data)
        response.raise_for_status()

        return response.json()


# =============================================================================
# 6. OAUTH2 MIDDLEWARE FOR WEB FRAMEWORKS
# =============================================================================

class OAuth2Middleware:
    """
    Middleware for protecting web application routes with OAuth2.

    Compatible with Flask, FastAPI, and other WSGI/ASGI frameworks.
    """

    def __init__(self, introspection_url: str, client_id: str,
                 client_secret: str, required_scopes: list = None):
        """
        Initialize OAuth2 middleware.

        Args:
            introspection_url: Token validation endpoint
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            required_scopes: List of required scopes for access
        """
        self.validator = OAuth2TokenValidator(introspection_url, client_id, client_secret)
        self.required_scopes = required_scopes or []

    def validate_request(self, authorization_header: str) -> Dict[str, Any]:
        """
        Validate authorization header and return token info.

        Args:
            authorization_header: Authorization header value (e.g., "Bearer token")

        Returns:
            Token information if valid

        Raises:
            ValueError: If authorization header is invalid
            PermissionError: If token is invalid or lacks required scopes
        """
        if not authorization_header or not authorization_header.startswith('Bearer '):
            raise ValueError("Missing or invalid Authorization header")

        # Extract token
        token = authorization_header[7:]  # Remove "Bearer " prefix

        # Validate token
        token_info = self.validator.validate_token(token)

        if not token_info.get('active'):
            raise PermissionError("Token is not active")

        # Check required scopes
        token_scopes = set(token_info.get('scope', '').split())
        required_scopes_set = set(self.required_scopes)

        if not required_scopes_set.issubset(token_scopes):
            missing_scopes = required_scopes_set - token_scopes
            raise PermissionError(f"Missing required scopes: {missing_scopes}")

        return token_info


# =============================================================================
# 7. EXAMPLE USAGE AND UTILITIES
# =============================================================================

def revoke_token(token: str, token_type: str, revocation_url: str,
                client_id: str, client_secret: str) -> bool:
    """
    Revoke an access or refresh token.

    Args:
        token: Token to revoke
        token_type: Either 'access_token' or 'refresh_token'
        revocation_url: Token revocation endpoint
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret

    Returns:
        True if revocation successful

    Example:
        >>> revoke_token(
        ...     token="my-access-token",
        ...     token_type="access_token",
        ...     revocation_url="https://auth.example.com/oauth/revoke",
        ...     client_id="client-id",
        ...     client_secret="client-secret"
        ... )
        True
    """
    data = {
        'token': token,
        'token_type_hint': token_type,
        'client_id': client_id,
        'client_secret': client_secret
    }

    response = requests.post(revocation_url, data=data)
    return response.status_code == 200


def example_web_app_flow():
    """
    Example: Complete OAuth2 flow for a web application.
    """
    # Initialize client
    client = OAuth2Client(
        client_id="my-web-app",
        client_secret="super-secret-key",
        redirect_uri="https://myapp.com/callback",
        auth_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token"
    )

    # Step 1: Get authorization URL
    auth_url = client.get_authorization_url(scope="profile email")
    print(f"Redirect user to: {auth_url}")

    # Step 2: User authorizes and is redirected back with code
    authorization_code = "received-from-callback"

    # Step 3: Exchange code for token
    token_data = client.exchange_code_for_token(authorization_code)
    print(f"Access token: {token_data['access_token']}")

    # Step 4: Make authenticated requests
    response = client.make_authenticated_request("https://api.provider.com/user/profile")
    user_data = response.json()
    print(f"User profile: {user_data}")

    # Step 5: Refresh token when expired
    if client.is_token_expired():
        client.refresh_access_token()


def example_mobile_app_flow():
    """
    Example: OAuth2 with PKCE for mobile apps.
    """
    # Initialize PKCE client
    client = OAuth2PKCEClient(
        client_id="my-mobile-app",
        redirect_uri="myapp://callback",
        auth_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token"
    )

    # Generate PKCE pair
    verifier, challenge = client.generate_pkce_pair()

    # Get authorization URL with PKCE
    auth_url = client.get_authorization_url(scope="profile")
    print(f"Open in browser: {auth_url}")

    # Exchange code with PKCE verifier
    authorization_code = "received-from-callback"
    token_data = client.exchange_code_for_token(authorization_code)
    print(f"Token received: {token_data}")


if __name__ == "__main__":
    print("OAuth2 Authentication Examples Module")
    print("=" * 50)
    print("\nThis module contains examples for:")
    print("1. Authorization Code Flow (Web Apps)")
    print("2. PKCE Flow (Mobile/SPA)")
    print("3. Client Credentials Flow (Service-to-Service)")
    print("4. Token Validation and Introspection")
    print("5. OAuth2 Middleware for Web Frameworks")
    print("\nUse these examples to implement OAuth2 in your application!")
