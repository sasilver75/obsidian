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


#### Why Use Airflow?
- A platform for or

