---
aliases:
  - OAuth 2.0
  - OAuth2
---
A standard for *delegated* [[Authorization]].

It lets one application access resources on another system without giving that application the user's password.
> "Can this app access this API/resource?"

It doesn't fully answer "Who is this user?" That identity layer is what [[OpenID Connect]] adds, which implements login/[[Authentication]] on top of OAuth 2.0.

## Example
> "Allow CalendarBot to read your Google Calendar?"
- If you approve, Google gives CalendarBot an access token.
- CalendarBot can then call Google Calendar APIs with limited permission.

## The main pieces:
- Resource Owner: The user
- Client: The app requesting access
- Authorization Server: The system issuing tokens
- Resource Server: The API being accessed
- Access Token: Credential used to call the API
- Scope: The permission boundary, like `calendar.read`

## Typical OAuth 2.0 flow:
1. App redirects user to the authorization server (the system issuing tokens)
2. User logs in and approves access, given description of what the App wants to do
3. Authorization server returns an an authorization code
4. App exchanges the code for an access token
5. App uses the access token to call an API on behalf of the user


![[Pasted image 20260611132321.png]]
You're getting back an Access Token for the Google Drive API, which you might think is an Authentication method, but the access token just proves that the app CAN access the resources, but it does NOT tell the app who you are; it just proves that the app *can* access certain resources from your Google Drive.

