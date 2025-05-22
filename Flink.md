SDIAH Link: https://www.hellointerview.com/learn/system-design/deep-dives/flink

Many System Design problems will require **[[Stream]] Processing**, where you have a continuous flow of data and you want to process, transform, or analyze it in real time.
- **Stream Processing** actually hard and expensive to get right! Many problems that *seem* like stream processing problems can actually be reduced to **Batch Processing** problems where you'd use something like [[Spark]] (or Haddoop, if you're old).
- Before embarking on a Stream processing solution, ==ask yourself: "**Do I really need real-time latencies?**"== ***For many problems, the answer is No***, and the Engineer after you will thank you for saving them the ops headache.

The most basic example of stream processing might be a service reading clicks from a Kafka topic, doing a trivial transformation (maybe reformatting the data for ingestion), and writing to a database.
![[Pasted image 20250522094641.png]]
Things can get substantially more complex from here!
- ==Imagine if we want to keep track of the count of clicks per user in the last 5 minutes!==
- Because of this 5 minute window, we've now introduced ==**state**== to our problem!
	- ==**Each message cannot be processed independently, because we need to remember the count from previous messages!**==
	- At first glance, we can do this in our Transform Service (consumer) by holding the count in memory, but this introduced a bunch of **==problems==**:
		- If our transform service **crashes**, it will **lose all of its state**! Our service could hypothetically recover from this by re-reading all of the messages from the [[Kafka]] topic, but this is slow and expensive.
		- Re: **Scaling** - if we add another service instance because we want to handle more clicks, we need to figure out **how to ==redistribute this state==** from existing instances to new ones! This is complicated with many failure scenarios!
		- What if events come in **==out-of-order==** or **==come in late==**? This is likely to happen and will impact the accuracy of our counts.

And things only get *harder* from here as we add complexity and more statefulness! Luckily, we have powerful systems/abstractions to help us with this, like Apache [[Flink]].

--------

**Flink** is a **==framework for building stream processing applications that solves some of the tricky problems like those we've discussed above and more.==**
- We'll talk about how Flink is used; there's a good chance you'll encounter a stream-oriented problem in your interview, and Flink is a powerful, flexible tool for the job when it applies.
- We'll see how Flink works at a high level under the hood. It's important to understand this for deep-dive questions.

# Basic Flink Concepts
- Flink is a ==**dataflow engine**==, built around the concept of a **==dataflow graph,==** which is a directed graph of nodes and edges describing a computation.
	- The nodes are the operations being performed, and the edges are the streams of data being passed between the operations.
![[Pasted image 20250522095426.png]]
- In Flink, the nodes are called **operators** and the edges are called **streams**. We give special names to the nodes at the beginning and end of the graph: **==sources==** and **==sinks==**.
- As a developer, your task is to define this graph, and Flink takes on the work of arranging the resources to execute the computation.
> ((Again, the point of Flink, a dataflow engine, is to define a dataflow graph of nodes and edges, where nodes are operations and edges are streams of data. The first node is a source and the last node is a sink.))

### Sources/Sinks
- **==sources==** and **==sinks==** are the entry and exit points for data in your Flink application.
- Sources read data from external systems and convert them into Flink streams. Common sources include:
	- **Kafka**, for message queues. ==The vast majority of sources we see in SD interviews start from Kafka==.
		- This is convenient because Kafka already forces you to think about how your data is arranged into topics and partitions, which will be relevant for reasoning about your Flink application.
	- **Kinesis**: For AWS streaming data
	- **File systems**: For batch processing
	- **Custom sources**: For specialized integrations
- Sinks write data from Flink streams to external systems. Common sinks include:
	- **Databases**: MySQL, PostgreSQL, MongoDB, et.c
	- **Data warehouses**: Snowflake, BigQuery, Redshift
	- **Message queues**: Kafka, RabbitMQ
	- **File systems**: HDFS, S3, local files


### Streams
- If sources, sinks, and operators are the nodes in the graph, then Streams are the edges of your dataflow graph.
- A stream is an unbounded sequence of data elements flowing through the system; think of it like an infinite array of events, which might look like:
```json
// Example event in a stream
{
  "user_id": "123",
  "action": "click",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "page": "/products/xyz"
}
```
Flink gives us tools to slice, transform, aggregate, recombine, and otherwise process streams.
- **==WARN==**: Streams in Flink are not necessarily append-only logs like they are with [[Kafka]]! **==There are no offsets or expectations of persistence in the stream abstraction.==** 
	- In Flink, durability is managed by **==Checkpoints==** which the system periodically creates. We'll get more into the detail on this later.


### Operators
- An **Operator** is a ==potentially stateful transformation== that ==processes one or more input streams and produces one or more output streams==.
- Operators are the building blocks for your stream processing application; Common operators include:
	- **==Map==**: Transform each element individually.
	- ==**Filter**==: Remove elements that don't match a condition.,
	- **==Reduce==**: Combine elements within a key.
	- **==Window==**: Group elements by time or count.
	- **==Join==**: Combine elements from two streams.
	- **==FlatMap==**: Transform each element into 0 or more elements.
	- **==Aggregate==**: Compute aggregates over windows or keys.
- Aside: Special note here for those familiar with map/reduce is that Flink operators can serve similar purposes to both mappers and reducers in [[MapReduce]], though the execution model is quite different; Flink processes records one at a time in a streaming fashion, rather than in batches like MapReduce.

A simple example of Operators in action:
```java
DataStream<ClickEvent> clicks = // input stream
clicks
  .keyBy(event -> event.getAdId())
  .window(TumblingEventTimeWindows.of(Time.minutes(5)))
  .reduce((a, b) -> new ClickEvent(a.getAdId(), a.getCount() + b.getCount()))
```
- This takes the input stream of clicks and partitions them by adId using the keyBy operator, created a KeyedStream.
- Applies a [[Tumbling Window]] of 5 minutes to the KeyedStream, which groups elements with the same key that fall within the same 5-minute time period.
- Applies a reduce function to each window. This function combines pairs of ClickEvents by creating a new ClickEvent that keeps the adId and adds the count values together. ((We're basically summing click counts in the window into a new simple ClickEvent.))

==The result is a stream that emits aggregated click counts per advertisement at 5-minute intervals.==


### State
Flink operators are **==stateful==**, meaning they can maintain internal state across multiple events! 
- If you want to count how many times a user has clicked in the last 5 minutes ,we need to maintain state about previous clicks (how many clicks have occurred, and when).
- This State needs to be managed internally by Flink for teh framework to give us scaling and fault tolerance guarantees. When a node crashes, Flink can restore the state from a checkpoint and resume processing from there.

Flink provides **==several types of state:==**
- ==Value State==: A single value per key
- ==List State==: A list of values per key
- ==Map State==: A map of values per key
- ==Aggregating State==: State for incremental aggregations
- ==Reducing State==: State for incremental reductions

Here's a simple example of using state to count clicks:
```java
// Create a ClickCounter class that extends KeyedProcessFunction
public class ClickCounter extends KeyedProcessFunction<String, ClickEvent, ClickCount> { // Processes clicks keyed by a String(userId), takes ClickEvent inputs, and produces ClickEvent outputs.
    private ValueState<Long> countState; // To store the count of clicks for each user


	// Initializes the state with a descriptor that names the state "count" and specifies its type as a Long
    @Override
    public void open(Configuration config) {
	    // Names the state as a Long named "count" 
        ValueStateDescriptor<Long> descriptor = 
            new ValueStateDescriptor<>("count", Long.class);
        // Initializes this state
        countState = getRuntimeContext().getState(descriptor);
    }

	// processElement is called for each input event
    @Override
    public void processElement(ClickEvent event, Context ctx, Collector<ClickCount> out) 
        throws Exception {
	        // Retrieves the current count from state (or initializes to a Long of 0, if null)
	        Long count = countState.value();
	        if (count == null) {
	            count = 0L;
        }
        // Increments the count
        count++;
        // Updates the state with the new count
        countState.update(count);
        // Outputs a new ClickCount object with the userId and updated count
        out.collect(new ClickCount(event.getUserId(), count));
    }
}
```

The end result of this state-based operator is that it ==maintains an ongoing count of clicks for each user and emits an updated count every time a new click arrives.==
- The important concept is that we need to make sure that the Flink framework knows about our state so that it can **Checkpoint** and restore in the event of a failure

### Watermarks
- In distributed stream processing systems, one of the biggest challenges is handling **==out-of-order events!==**
- Events can arrive late for various reasons:
	- Network delays between event sources
	- Different processing speeds across partitions
	- Source system delays or failures
	- Varying latencies in different parts of the system
- [[[Watermark]]s are Flink's solution to the problem:
	- A water mark is essentially a timestamp that flows through the system alongside streaming data, and declares: =="all events with timestamps before this watermark have arrived."==
	- As an example, you might receive the the watermark that lets you know 5PM has passed at 5:01:15PM.
	- This ensures that we have sufficient time to process all data that might have been created at 4:59PM but processed late.
	- ==By processing watermarks *alongside* the rest of the streaming data, we can ==:
		- Make decisions about when to trigger window computations.
		- Handle late-arriving events gracefully
		- Maintain consistent event-time processing across the distributed system.

Watermarks are configured on the **Source** of the stream. This watermark strategy tells Flink how long to wait for late events. Flink supports a number of watermark strategies, but you'll typically see two:
- **==Bounded Out-of-Orderness==**: Tells Flink to wait for events that arrive up to a certain time *after* the event timestamp.
- **==No Watermarks==**: Tells Flink to not wait for any late events, and processes events as they arrive.

> ==Interviewers like to see you thinking carefully about the implications of late and out-of-order events==. While Bounded OOO is common, most mission-critical systems will **augment** this with an offline true-up process to ensure that even very late data is eventually processed.


### Windows
- The final concept we'll cover are **Windows**. A window is a way to group elements in a stream by time or count.
- This is essential for aggregating data in a streaming context! Flink supports several typse of windows:
	- **==[[Tumbling Window]]s==**: Fixed size, non-overlapping windows
	- **==[[Sliding Window]]s==**: Fixed-size, overlapping windows
	- **==[[Session Window]]s==**: Dynamic-size windows based on activity
	- **==[[Global Window]]s==**: Custom windowing logic

![[Pasted image 20250522103106.png]]

Based on the window type, **==Flink will emit a new value for the window when the window ends!==**
- If it's a tumbling window of 5 minute durations and my input is clicks, Flink will emit a new value which contains all clicks that occurred in the last 5 minutes every 5 minutes.
	- ((It seems like the ==default behavior is that no result is emitted for empty windows==, which is an optimization so that you don't have a lot of "zero count" events.))

Windows can be applied to both keyed and non-keyed streams, though they're most commonly used with Keyed Streams.
- ==When applied to a Keyed Stream, windows are maintained independently for each key.==
	- This allows you to look at the window of data for a specific user, account, or other key.
	- **==Window choice can dramatically impact the accuracy and performance of your streaming app!==**
		- A tumbling window of 5 minute duration will emit once every 5 minutes
		- A sliding window of 5 minute duration with a 1 minute interval will emit every minute
		- It's worth reasoning backwards from problem requirements to determine the **least expensive window type that will give you the accuracy you need!**
	- Windows work closely with Watermarks to determine when to trigger computations and how to handle late events.
	- **You can also configure windows with allowed lateness to process events that arrive after the window has closed, but before a grace period has ended.**

----------

# Basic Use of Flink
- Let's set up a simple Flink app to process a stream of user clicks!

### Defining a Job
- A Flink job starts with a StreamExecutionEnvironment and typically involves defining your source, your transformations, and your sink.

```java
// We create a StreamExecutionEnvironment
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

// Define source (e.g., Kafka). Here, our Flink Source is a consumer of a Kafka stream. Some referenced code here (ClickEventSchema, properties) is out of view.
DataStream<ClickEvent> clicks = env
    .addSource(new FlinkKafkaConsumer<>("clicks", new ClickEventSchema(), properties));

// Define transformations (our series of Operations). Some refereneced code here is out of view (ClickAggregator).
// We're taking our clicks DataStream and creating a new DataStream by keying the click events by user ID, applying a 5 minute tumbling window, and somehow aggregating them.
DataStream<WindowedClicks> windowedClicks = clicks
    .keyBy(event -> event.getUserId())
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new ClickAggregator());

// Define sink (e.g., Elasticsearch) for our Kafka job
// See that we add a sink directly to the windowClicks DataStream, rather than create a new... stream...
windowedClicks
    .addSink(new ElasticsearchSink.Builder<>(elasticsearchConfig).build());

// Execute: This submits the job to the Flink cluster to run.
env.execute("Click Processing Job");
```
Above: We're defining a source, a series (here, just one) of transformations, and a sink
- Here, our source is a Kafka "clicks" topic, the transformation is some sort of aggregation of per-user clicks in a 5 minute tumbling window, and the sink is ElasticSearch.
- The env.execute submits the Flink job to the cluster to run.
### Submitting a Job
- The next step is to submit this job to the Flink cluster to run. When we cal env.execute on our ==StreamExecutionEnvironment==, Flink will:
	- Generate a ==JobGraph==: The Flink compiler transforms your logical data flow (==DataStream== operations) into an **optimized execution plan**.
	- Submit to ==JobManager==: The JobGraph is submitted to the JobManager, which serves as the **coordinator** for your Flink cluster.
	- Distribute Tasks: The JobManager breaks down the JobGraph into **tasks** and distributes them to ==TaskManagers==
	- Execute: The TaskManagers execute the tasks, with each task processing a portion of the data.

### Sample Job
- The nice thing about Flink is that both simple and extremely-sophisticated flows can be modeled with the same primitives.
- We can describe the entirety of our logic within a job.

#### Example: Basic Dashboard using Redis
- Here's a simple example of a dashboard using Redis to store the state of the counts.
```java
// We create a source for our StreamExecutionEnvironment, which is a Kafka Consumer pulling from teh clicks topic.
// (We're also passing some information about the schema of the events, and some Kafka cluster information assumedly to connect)
// We do this by just adding a source to our StreamExecutionEnvironent
DataStream<ClickEvent> clickstream = env
    .addSource(new FlinkKafkaConsumer<>("clicks", new JSONDeserializationSchema<>(ClickEvent.class), kafkaProps));
    
// Calculate metrics with 1-minute windows
// We add an Operator by keying our previous ClickEvent DataStream "clickstream" by pageId, and counting the number of events per page-id in 1 minute tumbling windows.
DataStream<PageViewCount> pageViews = clickstream
    .keyBy(click -> click.getPageId())
    .window(TumblingProcessingTimeWindows.of(Time.minutes(1)))
    .aggregate(new CountAggregator());
    
// Write to Redis for dashboard consumption
// We do this by just adding a sink to our last DataStream.
pageViews.addSink(new RedisSink<>(redisConfig, new PageViewCountMapper()));
```

#### Example: Fraud Detection System
- This is a slightly more sophisticated example of a Fraud detection system using Flink to detect fraudulent transactions
```java
 // As usual, we start by creating a Source (by adding it to our StreamExecutionEnvironment). Here, this source is a Kafka Consumer 
 // We're also assigning Timestamps and Watermarks, using a "BoundedOutOfOrderness" strategy (wait for events that arrive up to a certain time *after* the event timestamp)
DataStream<Transaction> transactions = env
    .addSource(new FlinkKafkaConsumer<>("transactions", 
                new KafkaAvroDeserializationSchema<>(Transaction.class), kafkaProps))
    .assignTimestampsAndWatermarks(
        WatermarkStrategy.<Transaction>forBoundedOutOfOrderness(Duration.ofSeconds(10))
            .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
    );
    
// Enrich transactions with account information
// Here, we're creatinga new DataSteram by keying the initial transactions datastream by accountId, connecting ((joining??)) the stream with an accountInfoStream, and creating a new Account-Enriched Transaction DataStream of EnrichedTransactions.
DataStream<EnrichedTransaction> enrichedTransactions = 
    transactions.keyBy(t -> t.getAccountId())
                .connect(accountInfoStream.keyBy(a -> a.getAccountId()))
                .process(new AccountEnrichmentFunction());

// Calculate velocity metrics (multiple transactions in short time)
// We're taking our EnrichedTransaction DataStream, keying it by AcountId, and applying a VelocityDetector, which alerts if there are >3 transactions over $1000 in any 30 minute window, sliding at 5 minute increments!
DataStream<VelocityAlert> velocityAlerts = enrichedTransactions
    .keyBy(t -> t.getAccountId())
    .window(SlidingEventTimeWindows.of(Time.minutes(30), Time.minutes(5)))
    .process(new VelocityDetector(3, 1000.0)); // Alert on 3+ transactions over $1000 in 30 min
    
// Pattern detection with CEP for suspicious sequences
// Defining a pattern... where first a fraudster makes a small purchase (to test the credit card), then makes a large purchase (to use the stolen credit card) within some short time period
Pattern<EnrichedTransaction, ?> fraudPattern = Pattern.<EnrichedTransaction>begin("small-tx")
    .where(tx -> tx.getAmount() < 10.0)
    .next("large-tx")
    .where(tx -> tx.getAmount() > 1000.0)
    .within(Time.minutes(5));

// We take that Pattern and crate a new DataStream of PaternAllerts by applying that CEP Pattern to the same EnrichedTransactoin datastream that our VelocityAlert operator is taking as input!
// We key our enriched transactions by CardId ... and apply the fraudpattern, then select some information (unspecified)
DataStream<PatternAlert> patternAlerts = CEP.pattern(
    enrichedTransactions.keyBy(t -> t.getCardId()), fraudPattern)
    .select(new PatternAlertSelector());
    
// Union all alerts and deduplicate
// Given that we've created our VelocityAlert and PatternAlert Datastream, this allAlerts stream just seems to... union them together, window them, and deduplicate them?
DataStream<Alert> allAlerts = velocityAlerts.union(patternAlerts)
    .keyBy(Alert::getAlertId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new AlertDeduplicator());
    
// Output to Kafka and Elasticsearch
// As usual, we just add a sink to our last DataStream, here connecting it to a FlinkProducer which will stick the Alert events in teh allAerts datastream into a KAfka "alerts" topic (with some event info, kafka cluster info)
// Oops, we actually chose to add TWO SINKS! So we're both outputting that information to a Kafka topic (for later processing by downstream systems) as well as inserting that information into ElasticSearch, perhaps for use in a UI/Dashboard.
allAlerts.addSink(new FlinkKafkaProducer<>("alerts", new AlertSerializer(), kafkaProps));
allAlerts.addSink(ElasticsearchSink.builder(elasticsearchConfig).build()); 
```
- Above: See that we defined a pattern for [[Complex Event Processing]] (CEP), a technique that analyzes real-time data streams to detect patterns/correlations/complex events, triggering actions based on these inputs.
- Above: See that we actually define TWO DIFFERENT SINKS!
- ==The net result above is a whole system design in one Flink job! :)==


# How Flink Works
- Let's now dive under the hood.
- Flink's architecture is designed to ==provide [[Exactly Once]] processing guarantees==, even in the face of failures, while maintaining high throughput and low latency.

### Cluster Architecture

##### Job Managers and Task Managers
- Flink runs as a distributed system with two main types of processes:
	- **Job Manager**: The ==coordinator== of the Flink cluster, responsible for ==scheduling tasks==, ==coordinating checkpoints==, and ==handling failures==. Think of it as the "supervisor" for the operation.
	- **Task Manager**: Workers that ==execute the actual data processing==. Each Task Manager provides a certain number of ==processing slots== to the cluster.

![[Pasted image 20250522113704.png]]

==Job Managers are leader-based, meaning there's a single job manager that's responsible for coordinating the work in the cluster==.
- High availability is achieved by deploying multiple Job Managers together and using a [[Quorum]]-based mechanism (e.g. [[ZooKeeper]]) to elect a leader.

- When you submit a job to Flink:
	1. The Job Manager receives the application and constructs the **execution graph**
	2. The Job Manager allocates tasks in that graph to available slots in Task Managers
	3. Task Managers start executing their assigned tasks
	4. The Job Manager monitors execution and handles failures.

==**Note:**== Unless you're interviewing for a data-engineering heavy role, most interviewers aren't going to ask you about Flink cluster administration; it's enough for non-specialized roles to know that there a re **Job Managers** who receive your job and coordinate the work in the cluster, and **Task Managers** who execute the actual data processing.


#### Task Slots and Parallelism
- Each **Task Manager** has one or more ==**Task Slots**, which are the basic unit of resource scheduling in Flink.==
- A task slot is a **unit of parallelism**, and by ==default the number of task slots is equal to the number of cores on the machine== (and this can be overridden to, for instance, use slots to represent chunks of memory of GPUs!)
	- Slots reserve capacity on a machine for jobs, and are frequently shared between operators of the same job! 

![[Pasted image 20250522114149.png]]

Slots serve several purposes:
1. They isolate memory between tasks
2. They control the number of parallel task instances
3. They enable resource sharing between different **Tasks** of the same **Job**.

The net result is that each **Task Manager** has a granular set of **Task Slots**, which are atomic resources that can be distributed between **Jobs** and **Operators**.


### State Management
- Okay, but what about durability guarantees? One of the biggest problems with stateful stream processing systems like Flink is to ensure how we can recover from failures without losing data!
- This is accomplished via Flink's **state management system**

#### State Backends
- Flink offers developers an abstraction for managing state which gives each **Job** a way to store state alongside each **operator** either for the entire job, or for each key.
	- The state itself is stored in some **Backend**, which is a component that manages the storage and retrieval of state.
- Flink offers many different state backend for different use cases:
	- ==Memory State Backend==: Stores state in JVM heap.
		- **Most of the time you'll prefer using memory state backend due to its performance,** but if you're running an operator that needs to store *more state than you have available memory,* you have the two below options.
	- ==FS State Backend==: Stores state in filesystem
	- ==[[RocksDB]] State Backend==: Stores state in RocksDB (supports a state larger than memory)

**Note**: All of these backends can be configured to store state in remote storage (e.g. S3, GCS, etc.) if you're running Flink in a cloud environment.
==**NOTE**==: The choice of state backend is crucial for production systems; memory state backend is fast but limited by RAM, while RocksDB can handle terabytes of state, but with higher latency.

#### Checkpointing and Exactly-Once Processing
- State is awesome, except when we need t orecover from failure!
- This 














































