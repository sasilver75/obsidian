c.f. [[Representational State Transfer|REST]]



________ 

There are some instances where we might want to use a different protocol for our application layer...

In GraphQL, we acknowledge some problems that exist in [[Representational State Transfer|REST]]ful APIs, and propose some solutions.


Say that we're populating a view that has some profile information, some status information, and a list of groups that the user is associated with.
- I might need to make 5+ requests to a server
- My page/screen is only going to be fully loaded after *all* of these return (see: [[Tail Latency]])
- All of the overhead of serializing/deserializing the information happens in each request. This is ==underfetching==, where we don't get all the information we need in our response, and so have to send many requests.
![[Pasted image 20260605163125.png]]


So we might create an "everything you need" endpoint:
![[Pasted image 20260605163257.png]]
- This request is going to be very expensive to return, and it's goign to take a long time.

GraphQL is going to solve this problem... where our requirements are constantly changing: We might want to add a widget here, or change fields ... and we don't necessarily have to want to change our APIs on the backend when that happens.

The way that GraphQL works:
- Provides a way for the frontend to describe a shape of the data that it needs to retrieve
- The backend is then able to figure out where to get that data so that it can return it in the way that the frontend needs.

![[Pasted image 20260605163411.png]]
Here's a GraphQL query... We're fetching a user, and their profile, and their groups, with their categories, etc.
The GraphQL response returns all this data, and nothing more.


Useful in situations where:
- Frontend requirements are changing all the time
- You have lots of different teams consuming data from your backend, for instance.

Both of these situations ==do not happen often in system design interviews, where you're often the sole author, and requirements are fixed.==
- It's useful only in situations where your interviewer isn't giving you all the reuqireements or is implying that they're going to be changing system, or when you want to design a system to support arbitrary query patterns.
- Sometimes used as the endpoint for newsfeed-style applications, where it's not obvious all the different views that a client might request up front.




