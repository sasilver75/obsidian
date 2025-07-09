---
aliases:
  - Airflow
---
An open-source platform for: Developing...Scheduling... and Monitoring... **Batch-Oriented Workflows**
- Airflow's extensible Python framework lets us build workflows connecting with virtually any technology, in Python.
- A web-based UI helps us visualize, manage, and debug our worfklows.
- We can run Airflow in a variety of configurations, from a single process on our laptop to a distributed system capable of handling massive workloads.

**Workflows are defined entirely in code (Python)**, which bring advantages:
- Enabling dynamic DAG generation and parametrization
- Includes a wide range of built-in ==operators== and can be extend to fit your needs.
- Leverages the Jinja templating engine, allowing rich customizations.

==Airflow DAGs== are models that encapsulate everything needed to execute a workflow. Some attributes include the following:
- ==Schedule==: When the workflow should run
- ==Tasks==: The discrete units of work, run on ==Workers==.
	- (It seems that these are also called Operators sometime)
- ==Task Dependencies==: The order and conditions under which tasks execute.
- ==Callbacks==: Actions to take when the entire workflow completes.
- Addditional Parameters: Many other operational details

Simple DAG:
```python
from datetime import datetime

from airflow.sdk import DAG, task
from airflow.providers.standard.operators.bash import BashOperator

# A DAG represents a workflow, a collection of tasks
with DAG(dag_id="demo", start_date=datetime(2022, 1, 1), schedule="0 0 * * *") as dag:
    # Tasks are represented as operators
    hello = BashOperator(task_id="hello", bash_command="echo hello")

    @task()
    def airflow():
        print("airflow")

    # Set dependencies between tasks
    hello >> airflow()
```
Above:
- A DAG named "demo" is scheduled to run daily starting on Jan 1st, 2022. This DAG is how Airflow represents a workflow.
- One using BashOperator to run a shell script, another using the @task decorator to define a Python function.
- The >> operator defines a dependency between our two tasks, executing them in the defined order.

Airflow tasks can be can be virtually any code! A task might be:
- Running a Spark job
- Moving files between storage buckets
- Sending a notification email

![[Pasted image 20250618151303.png]]
Each column in the chart on the left is a single run of the DAG (of all tasks in the DAG).

![[Pasted image 20250618151325.png]]
This ==DAG Overview== mode can be used to monitor and troubleshoot workflows.

--------------

==Flash AppBuilder (FAB)== Auth Manager:
- FAB Auth Mnager defines the user authentication and authorization in Airflow. The backend used to store all entities by the FAB auth manager is the Airflow database.
![[Pasted image 20250707172020.png]]
See more here: https://airflow.apache.org/docs/apache-airflow-providers-fab/stable/auth-manager/index.html

---------

Airflow 101: Building our First DAG:
- Remember that a ==DAG== is a collection of ==tasks== organized in away that reflects their relationships and dependenices. It's a roadmap for our workflow, showing how each task connects to the others.

Example Pipeline definition:

```python

import textwrap
from datetime import datetime, timedelta

# Operators; we need this to operate!
from airflow.providers.standard.operators.bash import BashOperator

# The DAG object; we'll need this to instantiate a DAG
from airflow.sdk import DAG

# See that we do a context manager of "with DAG(...)", passing some DAG instantiation information and default arguments (overridable) for operators that are going to be in the dag.
with DAG(
    "tutorial",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function, # or list of functions
        # 'on_success_callback': some_other_function, # or list of functions
        # 'on_retry_callback': another_function, # or list of functions
        # 'sla_miss_callback': yet_another_function, # or list of functions
        # 'on_skipped_callback': another_function, #or list of functions
        # 'trigger_rule': 'all_success'
    },
    description="A simple tutorial DAG",
    schedule=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:

    # t1, t2 and t3 are examples of tasks created by instantiating operators
    # Note that tasks and dags are pretty much the same
    t1 = BashOperator(
        task_id="print_date",
        bash_command="date",
    )

	# A second task
    t2 = BashOperator(
        task_id="sleep",
        depends_on_past=False,
        bash_command="sleep 5",
        retries=3,
    )
    t1.doc_md = textwrap.dedent(
        """\
    #### Task Documentation
    You can document your task using the attributes `doc_md` (markdown),
    `doc` (plain text), `doc_rst`, `doc_json`, `doc_yaml` which gets
    rendered in the UI's Task Instance Details page.
    ![img](https://imgs.xkcd.com/comics/fixing_problems.png)
    **Image Credit:** Randall Munroe, [XKCD](https://xkcd.com/license.html)
    """
    ) # See that we're adding soem documentation to the first task, it seems? Don't know why it's down here, after the second task.

	# You can also add documentation to the DAG itself, it seems. Again, I don't think that this should be down here. It should be up at the top of the context manager.
    dag.doc_md = __doc__  # providing that you have a docstring at the beginning of the DAG; OR
    dag.doc_md = """
    This is a documentation placed anywhere
    """  # otherwise, type it like this
    templated_command = textwrap.dedent(
        """
    {% for i in range(5) %}
        echo "{{ ds }}"
        echo "{{ macros.ds_add(ds, 7)}}"
    {% endfor %}
    """
    )

	# Here's a third operator. Note that we're giving each of these a task_id. I'm not sure yet waht depdns_on_past means.
    t3 = BashOperator(
        task_id="templated",
        depends_on_past=False,
        bash_command=templated_command,
    )

	# Interesting. We're using the bitshift operator to say that t1 must complete before t2 and t3 can start. Implicitly, it's saying that t2 and t3 can run in parallel, since they both only depend on t1. This is just some syntactic sugar; the arrows show the direction of dependency.
    t1 >> [t2, t3]
```
Above:
- When creating a DAG and its task, you can either pass arguments directly to each task, or define a set of default parameters in a dictionary (more efficient, cleaner).
- See that we created a ==DAG== that's going to run every day (schedule=timedelta(days=1)).
- An ==operator== represents a unit of work in Airflow. 
	- These are the building blocks of your workflows letting you define what tasks will be executed. While we can use operators for many tasks, Airflow also offers the ==Taskflow API== for a more Pythonic way to define workflows.
	- All operators derive from the ==BaseOperator==, which includes the essential arguments needed to run tasks in airflow. Popular operators include PythonOperator, BashOperator, and KubernetesPodOperator.
- To use an operator, we need to ***instantiate it as a*** ==task==. Tasks dictate the the operator will perform its work within the DAG's context. The ==task_id== serves as a unique operator for each of the two (e.g.) BashOperators we instantiate.
	- We mix operator-specific arguments (e.g. "bash_command") with operator-common arguments, like "retries," which are inherited from BAseOperator. This simplifies our code!
	- Ever task must include or inherit the arguments ==task_id== and ==owner==, or Airflow will raise an error.

Airflow harnesses the power of ==Jinja Templating==, giving us access to built-in parameters and macros to enhnace our workflow.
```python
templated_command = textwrap.dedent(
    """
{% for i in range(5) %}
    echo "{{ ds }}"
    echo "{{ macros.ds_add(ds, 7)}}"
{% endfor %}
"""
)

t3 = BashOperator(
    task_id="templated",
    depends_on_past=False,
    bash_command=templated_command,
)
```
Above:
- Notice that the templated_command includes logic in some `{% %}` blocks and references parameters like `{{ ds }}`. You can also pass files to the bash command, like bash_command=templated_command.sh, allowing for better organization of our code. We can even define user_defined_macros and user_defined_filters to create our own variables and filters for use in templates.

We can add documentation to both our DAG and our individual tasks. For DAGs, we have to use Markdown. For Tasks, it has to be plain text, markdown, reStructuredText, JSON, or YAML. It's a good practice to include documentation at the start of your DAG file.

```python
t1.doc_md = textwrap.dedent(
    """\
#### Task Documentation
You can document your task using the attributes `doc_md` (markdown),
`doc` (plain text), `doc_rst`, `doc_json`, `doc_yaml` which gets
rendered in the UI's Task Instance Details page.
![img](https://imgs.xkcd.com/comics/fixing_problems.png)
**Image Credit:** Randall Munroe, [XKCD](https://xkcd.com/license.html)
"""
)

dag.doc_md = __doc__  # providing that you have a docstring at the beginning of the DAG; OR
dag.doc_md = """
This is a documentation placed anywhere
"""  # otherwise, type it like this
```

This appears as:
![[Pasted image 20250707174023.png]]

Setting up Dependenies
- In Airflows, tasks can depend on one another. For instance, if you have tasks t1, t2, and t3, you can define their ==dependencies== in several ways:

```python
t1.set_downstream(t2)

# This means that t2 will depend on t1
# running successfully to run.
# It is equivalent to:
t2.set_upstream(t1)

# The bit shift operator can also be
# used to chain operations:
t1 >> t2

# And the upstream dependency with the
# bit shift operator:
t2 << t1

# Chaining multiple dependencies becomes
# concise with the bit shift operator:
t1 >> t2 >> t3

# A list of tasks can also be set as
# dependencies. These operations
# all have the same effect:
t1.set_downstream([t2, t3])
t1 >> [t2, t3]
[t2, t3] << t1
```

Once you've written a DAG in a file like above, we can just run `python <filename>` to see if there are any problems. If it runs correctly, you're good**!

`airflow db migrate`
- This performs database migrations using Alembic against the SQLite database located at `/Users/sam/airflow/airflow.db`. This creates tables for storing:
	- DAG defintiions
	- Task instances
	- Connections to external systems
	- Variable and configuration
	- Users and roles
It includes my new `tutorial` DAG that I made above.

Then we can:
`airflow dags list`
To print the list of active dags
- Includes dag_id, file location, owners, is_paused, bundle-name, and bundle_version

And we can do 
`airflow tasks list <dagname>` , using our `tutorial` DAG, to see the id of all of the tasks in the DAG.

	
You can test specific task instances for a designated _logical date_. This simulates the scheduler running your task for a particular date and time.
`airflow tasks test tutorial print_date 2015-06-01`
- Where print_date is the task_id of one of the tasks in our tutorial dag.

Keep in mind that the `airflow tasks test` command runs task instances ==locally==, outputs their logs to stdout, and doesn’t track state in the database. This is a handy way to test individual task instances.

Similarly, `airflow dags test` runs a single DAG run ==without registering any state in the database==, which is useful for testing your entire DAG locally.

-----

## Python DAGs with the TaskFlow API
- You've built your first Airflow DAG using ==traditional Operators== like PythonOperator. Let's look at a ==more modern and Pythonic way to write workflows== using the ==**TaskFlow API**==.
- Let's make a simple ETL pipeline that Extracts, Transforms, and Loads using the TaskFlow API!

Let's see a TaskFlow pipeline:

```python

import json

import pendulum

from airflow.sdk import dag, task

# Okay, see that instead of using a context amnager with... we instead use this @dag decorator.
@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),  # I don't believe it's guaranteed that it will execute at this time. Just... after this time? I'm not sure.
    catchup=False,
    tags=["example"],
)
def tutorial_taskflow_api():
    """
    ### TaskFlow API Tutorial Documentation
    This is a simple data pipeline example which demonstrates the use of
    the TaskFlow API using three simple tasks for Extract, Transform, and Load.
    Documentation that goes along with the Airflow TaskFlow API tutorial is
    located
    [here](https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html)
    """

	# See that we use the @task decorator to create a task.
    @task()
    def extract():  # Interesting that this one doesn't seem that take any arguments.
        """
        #### Extract task
        A simple Extract task to get data ready for the rest of the data
        pipeline. In this case, getting data is simulated by reading from a
        hardcoded JSON string.
        """
        # This is simulating us reading from a databse, or something.
        data_string = '{"1001": 301.27, "1002": 433.21, "1003": 502.22}'
        order_data_dict = json.loads(data_string)

		# And returning that data.
        return order_data_dict

	# See that task can take arguments, here. Multiple Outputs here tells Airflow that the funciton returns a dictionary of values, which should be split into individual "XComs". Each key in the returned dictionary becomes its own XCom entry, making it easy to reference specific values in downstream tasks. If you omit this (as in the previous task), the entire dictionary returend is stored as a single XCom instead, and must be accessed as a whole.
    @task(multiple_outputs=True)
    def transform(order_data_dict: dict):  # See that this seems to take as input the output from the previous task? I don't see where we define that this task depends on that one, though.
        """
        #### Transform task
        A simple Transform task which takes in the collection of order data and
        computes the total order value.
        """
        total_order_value = 0

        for value in order_data_dict.values():
            total_order_value += value

        return {"total_order_value": total_order_value}

	# Another task! 
    @task()
    def load(total_order_value: float): # See here that while the last task returned only a dictionary (which had a key total_order_value), it seems that we're able to select only that key here in the parameters of our next task. OOPS! Actually, read on to the "AHHH" part!
        """
        #### Load task
        A simple Load task which takes in the result of the Transform task and
        instead of saving it to end user review, just prints it out.
        """

        print(f"Total order value is: {total_order_value:.2f}")


	# AHHH, so this is where we actually instantiate these tasks.
	# See that we stuff the output form 1 into 2, and pluck the value from 2's output and pass into 3.
    order_data = extract()
    order_summary = transform(order_data) 
    load(order_summary["total_order_value"])


tutorial_taskflow_api()

```
Above:
- To make our DAG discoverable, we call the Python FUNCTION that we've decorated with `@dag`
- We write our tasks as regular Python functions, using the `@task` decorator to turn it into a task that Airflow can schedule and run.
	- The function's return value is passed to the next task -- no manual use of ==XComs== is required.
	- Under the hood, TaskFlow uses XComs to manage data passing automatically, abstracting away the complexity of manual XCom management from previous methods.
	- The use of `@task(multiple_outputs=True)` tells Airflow that a function that returns a dictionary of values should be split into individual XComs. Each key becomes its own XCom entry, making it easy to reference specific values in downstream tasks. If you omit this, the entire dictionary is stored as a single XCom instead, and must be accessed as a whole.
- Once we've defined our Tasks, we can build the pipeline by simply calling them like Python functions -- Airflow uses this invocation to set task ==dependencies== and manage data passing!

==Running Your DAG:== to enable and trigger your DAG, navigate to the Airflow UI, find your DAG in the list and click the toggle to enable it. You can trigger it manually by clicking the Trigger DAG button, or wait for it to run on its schedule.

*The "Old Way" (1.0?): Manual Wiring and XComs:* ((Note: We're now in Airflow 3.X))
- You had to use Operators like PythonOperator and pass data manually between tasks using XComs.
- Here's what the same DAG might have looked like using the traditional approach:

```python
# NOTE: This is all 1.X code, we're now in 3.X!
import json
import pendulum
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


def extract():
    # Old way: simulate extracting data from a JSON string
    data_string = '{"1001": 301.27, "1002": 433.21, "1003": 502.22}'
    return json.loads(data_string)


def transform(ti):
    # Old way: manually pull from XCom
    order_data_dict = ti.xcom_pull(task_ids="extract")
    total_order_value = sum(order_data_dict.values())
    return {"total_order_value": total_order_value}


def load(ti):
    # Old way: manually pull from XCom
    total = ti.xcom_pull(task_ids="transform")["total_order_value"]
    print(f"Total order value is: {total:.2f}")


with DAG(
    dag_id="legacy_etl_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
) as dag:
    extract_task = PythonOperator(task_id="extract", python_callable=extract)
    transform_task = PythonOperator(task_id="transform", python_callable=transform)
    load_task = PythonOperator(task_id="load", python_callable=load)

    extract_task >> transform_task >> load_task
```


> What's an ==XCom==?
> XComs are short for "==Cross Communications==", and are Airflows mechanism for passing small amounts of data between tasks in a DAG. They enable tasks to share information and coordinate their operations.
> XComs are stored in Airflow's metadata database and are identified by: DAG ID and Run ID (execution instance), Task ID (source task), Key (identifier for the data).
> XComs are meant for small data (eg <48KB), for data that must be JSON-serializable by default. XComs are limited to the same DAG run. Good for passing configuration values, sharing small results (IDs, counts, status), and coordinating between tasks... Alternatively, you can use some sort of external storage.
> The TaskFlow API (introduced in 2.0, refiend in 3.0) uses decorators to make XComs usage **completely transparent**.
> 	The downside is that TaskFlow API can seem "magical" without understanding XComs first.
> 	So how do they actually work? XComs are stored in AIrflow's metadata database as databse records. Fori nstance they might be stored in a `xcom` SQLlite table with fields like (id, dag_id, run_id, task_id, map_index, key, value, timestamp, dag_run_id). By default, JSON serialization is used.
> Note that while we're using SQLite here (`/Users/sam/airflow/airflow.db`), we may use PostgreSQL in production.


### Some advanced Taskflow patterns

Reusing Decorated Tasks
- You can reuse decorated tasks across multiple DAGs or DAG runs.
## Task Parameterization

Reusing Decorated Tasks
![[Pasted image 20250707201024.png]]

Handling Conflicting Dependencies
![[Pasted image 20250707201042.png]]






