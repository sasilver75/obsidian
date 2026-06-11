A piece of data that represents some security-relevant fact or authority.
- Credential: Thing used to prove access
- Token: A common kind of credential or flow artifact
- Principal: The actor that a token resolves to
- Claims/Scopes/Roles: Facts or permissions associated with it


Various types exist:
- [[Session]] Token: "This browser is logged into session X."
- [[Access Token]]: "Bearer may call API Y with scopes Z."
- [[Refresh Token]]: "Bearer may obtain new access tokens."
- [[ID Token]]: "This user was authenticated; here are signed identity claims."
- [[CSRF Token]]: "This form/request came from a page this site generated."
- [[API Key Authentication|API Key]]: "Bearer is this app/project/integration."

A token can be (not always exclusive):
- [[Opaque Token|Opaque]]: Random-looking; server must look it up.
- Structured (containing readable *claims*, as in a [[JSON Web Token|JWT]]).
- [[Cryptographic Signature|Signed]], so that servers can detect tampering.
- [[Encryption|Encrypted]], so that contents are hidden from the holder.
- [[Bearer Token|Bearer]]: Possession of the token is enough to use it.



