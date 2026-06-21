The RED method is a popular observability framework designed for monitoring microservices and request-driven applications by focusing on three key metrics: 
- ==**R**ate==: How many requests is the service handling?
- ==**E**rrors==: How many requests are failing?
- ==**D**uration==: How long are requests taking?

A complement to [[USE Method]] (Utilization/Saturation/Errors)

The problem it solves:
> In a distributed system, a service can be “up” but still unhealthy. An API can be responding, but latency is high, or the process is running but each request returns `500`.

The RED Method focuses on what matters from the perspective of the caller.
> Is the service receiving work, succeeding at that work, and completing that work quickly enough?


