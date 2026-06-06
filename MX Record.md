An MX (Mail Exchange) record is a type of [[Domain Name Service|DNS]] record that directs email messages to your mail server.

When someone sends an email to your domain (`you@yourcompany.com`), the MX record tells the sender's server exactly which mail server should receive it.

Consists of three main parts:
- Host/Name: Usually your root domain (often represented by @) or a specific subdomain.
- Points to (Value/Destination): The domain name of the mail server handling the emails (e.g., smtp.google.com or mail.example.com).
- Priority: A number that indicates the order in which mail servers should be used.

