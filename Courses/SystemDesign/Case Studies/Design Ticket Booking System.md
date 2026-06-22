
e.g. Ticketmaster

> Ticketmaster is an online platform that allows users to purchase tickets for concerts, sports events, theater, and other live entertainment.


# Functional Requirements
We're looking for "Users should be able to...." statements here, prioritizing the top ~3 or so functional requirements. For other requirements, you can/should note them, but they should be clearly stated to be "below the line" or "out of scope" so that the interviewer knows you won't be including them in the design.

1. Users should be able to view Events
2. Users should be able to search for Events
3. Users should be able to book tickets to Events

Out of scope: Viewing booked tickets, admins/event coordinators adding events, popular events should have dynamic pricing
- "Should I be prioritizing any sort of flows in this application? For instance, maybe I should be looking at the flows from a ticker-buying user's perspective, rather than the experience from an even-creator's perspective?"
- "Is this a system where we need to be able to manage the booking of specific seats at an event, or are they generic tickets to an event?"
	- Yes, specific seats.

# Non-Functional Requirements
1. The system should prioritize availability for searching and viewing events, but should prioritize consistency for booking events (no double booking)
2. the system should be scalable and be able to handle high throughput in the form of popular events (10 million users for one event)
3. The system should have low latency search (<500ms)
4. The system is read heavy, and thus needs to be able to support high read throughput.

Out of scope: GDPR, fault tolerance, secure transactions for purchases, ease of deploying to CI/CD, having regular backups.

![[Pasted image 20260621222822.png]]


# Core Entities
- ==User==: The individual interacting with the system
- ==Event==: The essential and central information about an event, including date/description/type/performer/venue.
- ==Performer==: Represents the individual/group performing in the event.
- ==Venue==: Represents the physical location where an event takes place.
- ==Ticket==: Represents something like a specific seat in a venue, along with the seat detail (section, row ,seat number), pricing, status. Stored as part of the Venue entity
- ==Booking==: Records the details of a user's ticket purchase. Includes the userID, a list of ticketIDs being booked, total price, booking status.

> You could arguably fold booking data into the Ticket entity itself, but a separate Booking entity is useful  when a user purchases multiple tickets in one transaction, since it groups them under a single order with a shared payment status and total price.


# API
Let's create APIs that satisfy the functional requirements (Users should be able to view Events, Users should be able to search for Events, Users should be able to book tickets to Events).

View Event
```
GET /events/{event_id} -> Event
```

Search Events
```
GET /events?keyword={}&start={}&end={}&limit={}&offset={} -> Event[]
```

Book Ticket
```
POST bookings/{event_id} -> bookingId
{
	"ticketsId": string[],
	"paymentDetails": ...
}
```
((A question here: I imagine that we're going to want to have strong consistency here for the actual booking of these specific finite resources/seats. If ONE out of 12 of our seats isn't bookable... is it reasonable to fail the request? In this case, I think the answer is yes, because if you can't sit with your friend, you'll find somewhere else.))

# High-Level Design
It's useful to look at our API routes and build something to support them.

1. User should be able to view events
![[Pasted image 20260621231018.png]]
- When a user navigates to `www.myticketmaster.com/event/{event_id}`, they should see details about that event. Crucially, this should include a seatmap showing seat availability. The page will also display the event's name, along with a description. Key information such as the location (including venue details), event dates, and facts about the performers or teams involved could be outlined.
- API Gateway serves as an entry point for clients to access different microservices of the system, responsible for routing but also cross-cutting concerns like authn, ratel limiting, logging.
- Event Service: Our first service
- Events DB: Stores tables for events, performers, venues
- Flow
	- Client makes a `GET` request with the eventId
	- API Gateway forwards to EventService
	- EventService then queries the EventsDB for the event, venue, and performer information and returns it to the client.


FR #2: Users should be able to search for events
	- Now, how are users supposed to find events int he first place?
	- The most basic thing you can do is create a simple service which accepts search queries... this service will connect your DB and query it by filtering for the fields in the API request. This has issues, but it's a good starting point. We will dig into better options in the deep dives below.
![[Pasted image 20260621231436.png]]
- ((I'm pretty annoyed by this, honestly... I don't feel like we yet have a strong reason to break this into a separate service, especially when it shares the same fucking database. Stupid as hell.))

When a user makes a search request, it's straightforward:
- The client makes a GET request with the search parameters
- Our LB accepts the request and routes it to eh API gateway with the fewest current connections ((We haven't even mentioned LBs yet, wtf))
- The API Gateway handles basic authn and rate limiting, and forwards the request onto the search service
- The search service then queries the events DB for the events matching the search parameters and returns them to the client.
Done!

FR #3 : Users should be able to book tickets to events
- The main thing that we're trying to avoid here is double-booking the same ticket.
- To handle this, we can select a database that supports transactions. This will allow us to ensure that only one user can book a ticket a time.
- While anything from MySQL to DynamoDB would be fine choices (just needs ACID properties), we'll opt for PostgreSQL.

![[Pasted image 20260621232313.png]]
1. New Tables in Events DB: We added a Bookings and Tickets table to our database.
	- ((I'm curious about why we're including status/userId in ticket; isn't it basically implied by the presence of a Booking, which itself also has a userId?))
2. Booking Service
	- This actually does the core functionality of the ticket booking process, interacting with databases that store data on bookings and tickets.
	- It also interfaces with a payment processor for transactions. Once a payment is confirmed, the booking service updates the ticket status to "sold".
	- Communicates with the Bookings/Tickets tables to fetch/update/store relevant data.
3. Payment Processor
	- Responsible for handling payment transactions; once a payment is processed, it notifies the booking service of the transaction status.

When a user goes to book a ticket:
1. User is redirected to a booking page where they can provide their payment details and confirm the booking.
2. Upon confirmation, a POST request is sent to the `/bookings` endpoint with the selected ticket IDs
3. The booking server initiates a trx to: Check availability of tickets, update the ticket statuses to book, and create a new booking record
	- ((You know how I feel about this...))
4. If the transaction is successful, the booking server returns a success response to the client. Otherwise, if the trx failed because another user booked the ticket in the meantime, a server returns a failure response and we pass this information back to the client.

Note: This means that when a new event is created, we need to create a new ticket for each seat in the venue. Each of which will be available for purchase until it is booked.

Note: Yes, here we're sharing a database across services... many of the world's largest companies share databases across services when it makes sense. Here, a shared database is the right call because the data is tightly coupled (we want ACID transactions for booking), and splitting databases would add complexity for no real benefit. ==Verbally weigh the tradeoffs and make a decision, instead of parroting architectural dogma.==


You might have noticed that there is a fundamental issue:
- Users can get to the booking page, type in their payment details, then find that the ticket they want is no longer available! We'll talk about this later in our deep dive.


# Deep Dives


DD #1: How do we improve the booking service by reserving tickets?
- Right now, sometime might spend 5 minutes filling out a payment form only to find out that the tickets they wanted are no longer available, because someone else typed their credit card information faster.
- We need to ensure that the ticket is locked for the user while they are checking out. We also need to ensure that if the user abandons the checkout process, the ticket is released for other users to purchase.

We can accomplish this using a distributed locking system like [[Redis]] with a [[Time to Live|TTL]].
- We want a *temporary reservation that automatically expires*. PostgreSQL doesn't natively support row-level expiration logic. 
- Flow:
	1. When a user selects a ticket, acquire a lock in Redis using a unique identifier (e.g `tickets:ticketid:lock`) with a predefined TTL. This acts as an automatic expiration time for the lock.
	2. If a user completes a purchase, the ticket's status is updated to "Booked", and the lock in Redis is manually released by the application code before the TTL expires.
	3. If the TTL expires, Redis automatically releases the lock. This ensures that the ticket becomes available for booking by other users without any additional intervention.
- ((I'm sort of thinking about a problem though where a process *thinks* that it has the lock, but it actually garbage collected for 15 seconds. What's to stop it from acting badly? This is the [[Stale Lease]] problem; do we use a [[Fencing Token]]?))

Now our ticket has two states: `available` and `booked`. Locking of reserved tickets is handled entirely by Redis, using `SET key value NX EX seconds`, which is atomic so that only one client will successfully set the key. If any lock fails, release the ones you already acquired.
- Redis lets you use Lua scripts to make multi-lock acquisition atomic, as long as the tickets hash to the same Redis node  (maybe we use a {section_id} on the key so that nearby seats are in the same redis node?)

> TTL expiration during payment: What if the lock TTL expires while payment is being processed? If User A's lock expires at minute 10 but their payment completes at minute 11, User B could have grabbed the lock in between. In this rare scenario, the database transaction in step 7 will fail for one of them (OCC ensures only one write succeeds), and we issue an automatic refund via Stripe for the failed booking. Set the TTL generously to minimize this, and, even better, consider extending the lock when payment is initiated.

![[Pasted image 20260621234840.png]]

When a user wants to book a ticket(s):
1. A user selects a seat from an interactive seat map on an event detail page. This triggers a POST `/bookings` with the ticketId associated with that seat
2. The request is forwarded from our API gateway onto the Booking Service
3. The booking Service will lock that ticket by adding it to our Redis Distributed Lock with a TTL of 10 minutes (that's how long we hold a ticket for)
4. The booking service will also write a new booing entry into the DB with a status of in-progress
5. We then respond to the user with their newly created `bookingId` and route the client to the payment page.
6. The user fills out payment information and Stripe processes the payment and notifies our system via webhook that the payment was successful.
7. Upon successful payment confirmation, our system's webhook retrieves the `bookingId` embedded within the Stripe metadata.
8. Our system's webhook retrieves the bookId embedded within the Stripe metadata. With this bookingId, the webhook initiates a database transaction to concurrently update the Ticket and Booking tables. Specifically, the status of the ticket linked to the booking is changed to "sold" in the Ticket table. The webhook handler should be marked as 









