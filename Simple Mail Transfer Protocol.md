---
aliases:
  - SMTP
---
The standard internet protocol used for sending and relaying email across the internet..

When you hit "send" on your email client, SMTP acts as the digital post service, routing your message to the recipient's mail service.

```
Sender's mail app 
↓ 
MUA → MSA → MTA → MTA → MDA → recipient mailbox 
									↓ 
									IMAP/POP/webmail
```

Actors involved:
- Mail User Agent (MUA): The email interface a person uses, e.g. Gmail, Outlook
- Mail Submission Agent (MSA): Accepts outgoing mail from a user's mail app, requires login/authn.
- Mail Transfer Agent (MTA): Moves email between mail servers. Looks up recipient's domain's [[MX Record]] in [[Domain Name Service|DNS]] to find where to send the message. Examples include Postfix, Exim, Sendmail.
- Mail Delivery Agent (MDA): Performs final delivery into the recipient's mailbox or mail store. May apply filters, spam rules, aliases, forwarding, or mailbox placement. Examples include Dovecot LDA/LMTP, Procmail, and mail server-specific delivery systems.

IMAP/POP3 are for reading mail, not sending it.
- IMAP: Keeps mail on the server and syncs folders across devices
- POP3: Usually downloads mail to a device

In short:
- SMTP sends mail
- MTAs relay it
- MDAs deliver it
- IMAP/POP/webmail lets users read it







