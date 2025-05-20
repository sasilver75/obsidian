All reads reflect the most recent write. All readers have the same view of the system. 

This is the most expensive consistency model in terms of performance, but is necessary for systems that require absolute accuracy like bank account balances, or systems like TicketMaster (e.g. reserving a ticket, or buying a pair of a limited sneakers).

