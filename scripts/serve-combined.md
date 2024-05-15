(serve-key-concepts)=

# Key Concepts

(serve-key-concepts-deployment)=

## Deployment

Deployments are the central concept in Ray Serve.
A deployment contains business logic or an ML model to handle incoming requests and can be scaled up to run across a Ray cluster.
At runtime, a deployment consists of a number of *replicas*, which are individual copies of the class or function that are started in separate Ray Actors (processes).
The number of replicas can be scaled up or down (or even autoscaled) to match the incoming request load.

To define a deployment, use the {mod}`@serve.deployment <ray.serve.deployment>` decorator on a Python class (or function for simple use cases).
Then, `bind` the deployment with optional arguments to the constructor to define an [application](serve-key-concepts-application).
Finally, deploy the resulting application using `serve.run` (or the equivalent `serve run` CLI command, see [Development Workflow](serve-dev-workflow) for details).

```{literalinclude} ../serve/doc_code/key_concepts.py
:start-after: __start_my_first_deployment__
:end-before: __end_my_first_deployment__
:language: python
```

(serve-key-concepts-application)=

## Application

An application is the unit of upgrade in a Ray Serve cluster. An application consists of one or more deployments. One of these deployments is considered the [“ingress” deployment](serve-key-concepts-ingress-deployment), which handles all inbound traffic.

Applications can be called via HTTP at the specified `route_prefix` or in Python using a `DeploymentHandle`.
 
(serve-key-concepts-deployment-handle)=

## DeploymentHandle (composing deployments)

Ray Serve enables flexible model composition and scaling by allowing multiple independent deployments to call into each other.
When binding a deployment, you can include references to _other bound deployments_.
Then, at runtime each of these arguments is converted to a {mod}`DeploymentHandle <ray.serve.handle.DeploymentHandle>` that can be used to query the deployment using a Python-native API.
Below is a basic example where the `Ingress` deployment can call into two downstream models.
For a more comprehensive guide, see the [model composition guide](serve-model-composition).

```{literalinclude} ../serve/doc_code/key_concepts.py
:start-after: __start_deployment_handle__
:end-before: __end_deployment_handle__
:language: python
```

(serve-key-concepts-ingress-deployment)=

## Ingress deployment (HTTP handling)

A Serve application can consist of multiple deployments that can be combined to perform model composition or complex business logic.
However, one deployment is always the "top-level" one that is passed to `serve.run` to deploy the application.
This deployment is called the "ingress deployment" because it serves as the entrypoint for all traffic to the application.
Often, it then routes to other deployments or calls into them using the `DeploymentHandle` API, and composes the results before returning to the user.

The ingress deployment defines the HTTP handling logic for the application.
By default, the `__call__` method of the class is called and passed in a `Starlette` request object.
The response will be serialized as JSON, but other `Starlette` response objects can also be returned directly.
Here's an example:

```{literalinclude} ../serve/doc_code/key_concepts.py
:start-after: __start_basic_ingress__
:end-before: __end_basic_ingress__
:language: python
```

After binding the deployment and running `serve.run()`, it is now exposed by the HTTP server and handles requests using the specified class.
We can query the model using `requests` to verify that it's working.

For more expressive HTTP handling, Serve also comes with a built-in integration with `FastAPI`.
This allows you to use the full expressiveness of FastAPI to define more complex APIs:

```{literalinclude} ../serve/doc_code/key_concepts.py
:start-after: __start_fastapi_ingress__
:end-before: __end_fastapi_ingress__
:language: python
```

## What's next?
Now that you have learned the key concepts, you can dive into these guides:
- [Resource allocation](serve-resource-allocation)
- [Autoscaling guide](serve-autoscaling)
- [Configuring HTTP logic and integrating with FastAPI](http-guide)
- [Development workflow for Serve applications](serve-dev-workflow)
- [Composing deployments to perform model composition](serve-model-composition)


(serve-multi-application)=
# Deploy Multiple Applications

Serve supports deploying multiple independent Serve applications. This user guide walks through how to generate a multi-application config file and deploy it using the Serve CLI, and monitor your applications using the CLI and the Ray Serve dashboard.

## Context
### Background 
With the introduction of multi-application Serve, we walk you through the new concept of applications and when you should choose to deploy a single application versus multiple applications per cluster. 

An application consists of one or more deployments. The deployments in an application are tied into a direct acyclic graph through [model composition](serve-model-composition). An application can be called via HTTP at the specified route prefix, and the ingress deployment handles all such inbound traffic. Due to the dependence between deployments in an application, one application is a unit of upgrade. 

### When to use multiple applications
You can solve many use cases by using either model composition or multi-application. However, both have their own individual benefits and can be used together.

Suppose you have multiple models and/or business logic that all need to be executed for a single request. If they are living in one repository, then you most likely upgrade them as a unit, so we recommend having all those deployments in one application.

On the other hand, if these models or business logic have logical groups, for example, groups of models that communicate with each other but live in different repositories, we recommend separating the models into applications. Another common use-case for multiple applications is separate groups of models that may not communicate with each other, but you want to co-host them to increase hardware utilization. Because one application is a unit of upgrade, having multiple applications allows you to deploy many independent models (or groups of models) each behind different endpoints. You can then easily add or delete applications from the cluster as well as upgrade applications independently of each other.

## Getting started

Define a Serve application:
```{literalinclude} doc_code/image_classifier_example.py
:language: python
:start-after: __serve_example_begin__
:end-before: __serve_example_end__
```

Copy this code to a file named `image_classifier.py`.

Define a second Serve application:
```{literalinclude} doc_code/translator_example.py
:language: python
:start-after: __serve_example_begin__
:end-before: __serve_example_end__
```
Copy this code to a file named `text_translator.py`.

Generate a multi-application config file that contains both of these two applications and save it to `config.yaml`.

```
serve build image_classifier:app text_translator:app -o config.yaml
```

This generates the following config:
```yaml
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

grpc_options:
  port: 9000
  grpc_servicer_functions: []

logging_config:
  encoding: JSON
  log_level: INFO
  logs_dir: null
  enable_access_log: true

applications:
  - name: app1
    route_prefix: /classify
    import_path: image_classifier:app
    runtime_env: {}
    deployments:
      - name: downloader
      - name: ImageClassifier

  - name: app2
    route_prefix: /translate
    import_path: text_translator:app
    runtime_env: {}
    deployments:
      - name: Translator
```

:::{note} 
The names for each application are auto-generated as `app1`, `app2`, etc. To give custom names to the applications, modify the config file before moving on to the next step.
:::

### Deploy the applications
To deploy the applications, be sure to start a Ray cluster first.

```console
$ ray start --head

$ serve deploy config.yaml
> Sent deploy request successfully!
```

Query the applications at their respective endpoints, `/classify` and `/translate`.
```{literalinclude} doc_code/image_classifier_example.py
:language: python
:start-after: __request_begin__
:end-before: __request_end__
```
```{literalinclude} doc_code/translator_example.py
:language: python
:start-after: __request_begin__
:end-before: __request_end__
```

#### Development workflow with `serve run`
You can also use the CLI command `serve run` to run and test your application easily, either locally or on a remote cluster. 
```console
$ serve run config.yaml
> 2023-04-04 11:00:05,901 INFO scripts.py:327 -- Deploying from config file: "config.yaml".
> 2023-04-04 11:00:07,505 INFO worker.py:1613 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
> 2023-04-04 11:00:09,012 SUCC scripts.py:393 -- Submitted deploy config successfully.
```

The `serve run` command blocks the terminal, which allows logs from Serve to stream to the console. This helps you test and debug your applications easily. If you want to change your code, you can hit Ctrl-C to interrupt the command and shutdown Serve and all its applications, then rerun `serve run`.

:::{note}
`serve run` only supports running multi-application config files. If you want to run applications by directly passing in an import path, `serve run` can only run one application import path at a time.
:::

### Check status
Check the status of the applications by running `serve status`.

```console
$ serve status
proxies:
  2e02a03ad64b3f3810b0dd6c3265c8a00ac36c13b2b0937cbf1ef153: HEALTHY
applications:
  app1:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1693267064.0735464
    deployments:
      downloader:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
      ImageClassifier:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
  app2:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1693267064.0735464
    deployments:
      Translator:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
```

### Send requests between applications
You can also make calls between applications without going through HTTP by using the Serve API `serve.get_app_handle` to get a handle to any live Serve application on the cluster. This handle can be used to directly execute a request on an application. Take the classifier and translator app above as an example. You can modify the `__call__` method of the `ImageClassifier` to check for another parameter in the HTTP request, and send requests to the translator application.

```{literalinclude} doc_code/image_classifier_example.py
:language: python
:start-after: __serve_example_modified_begin__
:end-before: __serve_example_modified_end__
```

Then, send requests to the classifier application with the `should_translate` flag set to True:
```{literalinclude} doc_code/image_classifier_example.py
:language: python
:start-after: __second_request_begin__
:end-before: __second_request_end__
```


### Inspect deeper

For more visibility into the applications running on the cluster, go to the Ray Serve dashboard at [`http://localhost:8265/#/serve`](http://localhost:8265/#/serve).

You can see all applications that are deployed on the Ray cluster:

![applications](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/applications-dashboard.png)

The list of deployments under each application:

![deployments](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/deployments-dashboard.png)

As well as the list of replicas for each deployment:

![replicas](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/replica-dashboard.png)

For more details on the Ray Serve dashboard, see the [Serve dashboard documentation](dash-serve-view).


## Add, delete, and update applications
You can add, remove or update entries under the `applications` field to add, remove or update applications in the cluster. This doesn't affect other applications on the cluster. To update an application, modify the config options in the corresponding entry under the `applications` field.

:::{note}
The in-place update behavior for an application when you resubmit a config is the same as the single-application behavior. For how an application reacts to different config changes, see [Updating a Serve Application](serve-inplace-updates).
:::



(serve-config-migration)=
### Migrating from a single-application config

Migrating the single-application config `ServeApplicationSchema` to the multi-application config format `ServeDeploySchema` is straightforward. Each entry under the  `applications` field matches the old, single-application config format. To convert a single-application config to the multi-application config format:
* Copy the entire old config to an entry under the `applications` field.
* Remove `host` and `port` from the entry and move them under the `http_options` field.
* Name the application.
* If you haven't already, set the application-level `route_prefix` to the route prefix of the ingress deployment in the application. In a multi-application config, you should set route prefixes at the application level instead of for the ingress deployment in each application.
* When needed, add more applications.

For more details on the multi-application config format, see the documentation for [`ServeDeploySchema`](serve-rest-api-config-schema).

:::{note} 
You must remove `host` and `port` from the application entry. In a multi-application config, specifying cluster-level options within an individual application isn't applicable, and is not supported.
:::


(serve-monitoring)=

# Monitor Your Application

This section helps you debug and monitor your Serve applications by:

* viewing the Ray dashboard
* viewing the `serve status` output
* using Ray logging and Loki
* inspecting built-in Ray Serve metrics
* exporting metrics into Arize platform

## Ray Dashboard

You can use the Ray dashboard to get a high-level overview of your Ray cluster and Ray Serve application's states.
This includes details such as:
* the number of deployment replicas currently running
* logs for your Serve controller, deployment replicas, and proxies
* the Ray nodes (i.e. machines) running in your Ray cluster.

You can access the Ray dashboard at port 8265 at your cluster's URI.
For example, if you're running Ray Serve locally, you can access the dashboard by going to `http://localhost:8265` in your browser.

View important information about your application by accessing the [Serve page](dash-serve-view).

```{image} https://raw.githubusercontent.com/ray-project/Images/master/docs/new-dashboard-v2/serve.png
:align: center
```

This example has a single-node cluster running a deployment named `Translator`. This deployment has 2 replicas.

View details of these replicas by browsing the Serve page. On the details page of each replica. From there, you can view metadata about the replica and the logs of the replicas, including the `logging` and `print` statements generated by the replica process.


Another useful view is the [Actors view](dash-actors-view). This example Serve application uses four [Ray actors](actor-guide):

- 1 Serve controller
- 1 HTTP proxy
- 2 `Translator` deployment replicas

You can see the details of these entities throughout the Serve page and in the actor's page.
This page includes additional useful information like each actor's process ID (PID) and a link to each actor's logs. You can also see whether any particular actor is alive or dead to help you debug potential cluster failures.

:::{tip}
To learn more about the Serve controller actor, the HTTP proxy actor(s), the deployment replicas, and how they all work together, check out the [Serve Architecture](serve-architecture) documentation.
:::

For a detailed overview of the Ray dashboard, see the [dashboard documentation](observability-getting-started).

(serve-in-production-inspecting)=

## Inspect applications with the Serve CLI

Two Serve CLI commands help you inspect a Serve application in production: `serve config` and `serve status`.
If you have a remote cluster, `serve config` and `serve status` also has an `--address/-a` argument to access the cluster. See [VM deployment](serve-in-production-remote-cluster) for more information on this argument.

`serve config` gets the latest config file that the Ray Cluster received. This config file represents the Serve application's goal state. The Ray Cluster constantly strives to reach and maintain this state by deploying deployments, and recovering failed replicas, and performing other relevant actions.

Using the `serve_config.yaml` example from [the production guide](production-config-yaml):

```console
$ ray start --head
$ serve deploy serve_config.yaml
...

$ serve config
name: default
route_prefix: /
import_path: text_ml:app
runtime_env:
  pip:
    - torch
    - transformers
deployments:
- name: Translator
  num_replicas: 1
  user_config:
    language: french
- name: Summarizer
  num_replicas: 1
```

`serve status` gets your Serve application's current status. This command reports the status of the `proxies` and the `applications` running on the Ray cluster.

`proxies` lists each proxy's status. Each proxy is identified by the node ID of the node that it runs on. A proxy has three possible statuses:
* `STARTING`: The proxy is starting up and is not yet ready to serve requests.
* `HEALTHY`: The proxy is capable of serving requests. It is behaving normally.
* `UNHEALTHY`: The proxy has failed its health-checks. It will be killed, and a new proxy will be started on that node.
* `DRAINING`: The proxy is healthy but is closed to new requests. It may contain pending requests that are still being processed.
* `DRAINED`: The proxy is closed to new requests. There are no pending requests.

`applications` contains a list of applications, their overall statuses, and their deployments' statuses. Each entry in `applications` maps an application's name to four fields:
* `status`: A Serve application has four possible overall statuses:
    * `"NOT_STARTED"`: No application has been deployed on this cluster.
    * `"DEPLOYING"`: The application is currently carrying out a `serve deploy` request. It is deploying new deployments or updating existing ones.
    * `"RUNNING"`: The application is at steady-state. It has finished executing any previous `serve deploy` requests, and is attempting to maintain the goal state set by the latest `serve deploy` request.
    * `"DEPLOY_FAILED"`: The latest `serve deploy` request has failed.
* `message`: Provides context on the current status.
* `deployment_timestamp`: A UNIX timestamp of when Serve received the last `serve deploy` request. The timestamp is calculated using the `ServeController`'s local clock.
* `deployments`: A list of entries representing each deployment's status. Each entry maps a deployment's name to three fields:
    * `status`: A Serve deployment has three possible statuses:
        * `"UPDATING"`: The deployment is updating to meet the goal state set by a previous `deploy` request.
        * `"HEALTHY"`: The deployment achieved the latest requests goal state.
        * `"UNHEALTHY"`: The deployment has either failed to update, or has updated and has become unhealthy afterwards. This condition may be due to an error in the deployment's constructor, a crashed replica, or a general system or machine error.
    * `replica_states`: A list of the replicas' states and the number of replicas in that state. Each replica has five possible states:
        * `STARTING`: The replica is starting and not yet ready to serve requests.
        * `UPDATING`: The replica is undergoing a `reconfigure` update.
        * `RECOVERING`: The replica is recovering its state.
        * `RUNNING`: The replica is running normally and able to serve requests.
        * `STOPPING`: The replica is being stopped.
    * `message`: Provides context on the current status.

Use the `serve status` command to inspect your deployments after they are deployed and throughout their lifetime.

Using the `serve_config.yaml` example from [an earlier section](production-config-yaml):

```console
$ ray start --head
$ serve deploy serve_config.yaml
...

$ serve status
proxies:
  cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec: HEALTHY
applications:
  default:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1694041157.2211847
    deployments:
      Translator:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
      Summarizer:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
```

For Kubernetes deployments with KubeRay, tighter integrations of `serve status` with Kubernetes are available. See [Getting the status of Serve applications in Kubernetes](serve-getting-status-kubernetes).

## Get application details in Python

Call the `serve.status()` API to get Serve application details in Python. `serve.status()` returns the same information as the `serve status` CLI command inside a `dataclass`. Use this method inside a deployment or a Ray driver script to obtain live information about the Serve applications on the Ray cluster. For example, this `monitoring_app` reports all the `RUNNING` Serve applications on the cluster:

```{literalinclude} doc_code/monitoring/monitor_deployment.py
:start-after: __monitor_start__
:end-before: __monitor_end__
:language: python
```

(serve-logging)=
## Ray logging

To understand system-level behavior and to surface application-level details during runtime, you can leverage Ray logging.

Ray Serve uses Python's standard `logging` module with a logger named `"ray.serve"`.
By default, logs are emitted from actors both to `stderr` and on disk on each node at `/tmp/ray/session_latest/logs/serve/`.
This includes both system-level logs from the Serve controller and proxy as well as access logs and custom user logs produced from within deployment replicas.

In development, logs are streamed to the driver Ray program (the Python script that calls `serve.run()` or the `serve run` CLI command), so it's convenient to keep the driver running while debugging.

For example, let's run a basic Serve application and view the logs that it emits.

First, let's create a simple deployment that logs a custom log message when it's queried:

```{literalinclude} doc_code/monitoring/monitoring.py
:start-after: __start__
:end-before: __end__
:language: python
```

Run this deployment using the `serve run` CLI command:

```console
$ serve run monitoring:say_hello

2023-04-10 15:57:32,100	INFO scripts.py:380 -- Deploying from import path: "monitoring:say_hello".
[2023-04-10 15:57:33]  INFO ray._private.worker::Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
(ServeController pid=63503) INFO 2023-04-10 15:57:35,822 controller 63503 deployment_state.py:1168 - Deploying new version of deployment SayHello.
(ProxyActor pid=63513) INFO:     Started server process [63513]
(ServeController pid=63503) INFO 2023-04-10 15:57:35,882 controller 63503 deployment_state.py:1386 - Adding 1 replica to deployment SayHello.
2023-04-10 15:57:36,840	SUCC scripts.py:398 -- Deployed Serve app successfully.
```

`serve run` prints a few log messages immediately. Note that a few of these messages start with identifiers such as

```
(ServeController pid=63881)
```

These messages are logs from Ray Serve [actors](actor-guide). They describe which actor (Serve controller, proxy, or deployment replica) created the log and what its process ID is (which is useful when distinguishing between different deployment replicas or proxies). The rest of these log messages are the actual log statements generated by the actor.

While `serve run` is running, we can query the deployment in a separate terminal window:

```
curl -X GET http://localhost:8000/
```

This causes the HTTP proxy and deployment replica to print log statements to the terminal running `serve run`:

```console
(ServeReplica:SayHello pid=63520) INFO 2023-04-10 15:59:45,403 SayHello SayHello#kTBlTj HzIYOzaEgN / monitoring.py:16 - Hello world!
(ServeReplica:SayHello pid=63520) INFO 2023-04-10 15:59:45,403 SayHello SayHello#kTBlTj HzIYOzaEgN / replica.py:527 - __CALL__ OK 0.5ms
```

:::{note}
Log messages include the logging level, timestamp, deployment name, replica tag, request ID, route, file name, and line number.
:::

Find a copy of these logs at `/tmp/ray/session_latest/logs/serve/`. You can parse these stored logs with a logging stack such as ELK or [Loki](serve-logging-loki) to be able to search by deployment or replica.

Serve supports [Log Rotation](log-rotation) of these logs through setting the environment variables `RAY_ROTATION_MAX_BYTES` and `RAY_ROTATION_BACKUP_COUNT`.

To silence the replica-level logs or otherwise configure logging, configure the `"ray.serve"` logger **inside the deployment constructor**:

```python
import logging

logger = logging.getLogger("ray.serve")

@serve.deployment
class Silenced:
    def __init__(self):
        logger.setLevel(logging.ERROR)
```

This controls which logs are written to STDOUT or files on disk.
In addition to the standard Python logger, Serve supports custom logging. Custom logging lets you control what messages are written to STDOUT/STDERR, files on disk, or both.

For a detailed overview of logging in Ray, see [Ray Logging](configure-logging).

### Configure Serve logging
From ray 2.9, the logging_config API configures logging for Ray Serve. You can configure
logging for Ray Serve. Pass a dictionary or object of [LoggingConfig](../serve/api/doc/ray.serve.schema.LoggingConfig.rst)
to the `logging_config` argument of `serve.run` or `@serve.deployment`.

#### Configure logging format
You can configure the JSON logging format by passing `encoding=JSON` to `logging_config`
argument in `serve.run` or `@serve.deployment`

::::{tab-set}

:::{tab-item} serve.run
```{literalinclude} doc_code/monitoring/logging_config.py
:start-after: __serve_run_json_start__
:end-before: __serve_run_json_end__
:language: python
```
:::

:::{tab-item} @serve.deployment
```{literalinclude} doc_code/monitoring/logging_config.py
:start-after: __deployment_json_start__
:end-before: __deployment_json_end__
:language: python
```
:::

::::

In the replica `Model` log file, you should see the following:

```
# cat `ls /tmp/ray/session_latest/logs/serve/replica_default_Model_*`

{"levelname": "INFO", "asctime": "2024-02-27 10:36:08,908", "deployment": "default_Model", "replica": "rdofcrh4", "message": "replica.py:855 - Started initializing replica."}
{"levelname": "INFO", "asctime": "2024-02-27 10:36:08,908", "deployment": "default_Model", "replica": "rdofcrh4", "message": "replica.py:877 - Finished initializing replica."}
{"levelname": "INFO", "asctime": "2024-02-27 10:36:10,127", "deployment": "default_Model", "replica": "rdofcrh4", "request_id": "f4f4b3c0-1cca-4424-9002-c887d7858525", "route": "/", "application": "default", "message": "replica.py:1068 - Started executing request to method '__call__'."}
{"levelname": "INFO", "asctime": "2024-02-27 10:36:10,127", "deployment": "default_Model", "replica": "rdofcrh4", "request_id": "f4f4b3c0-1cca-4424-9002-c887d7858525", "route": "/", "application": "default", "message": "replica.py:373 - __CALL__ OK 0.6ms"}
```

:::{note}
The `RAY_SERVE_ENABLE_JSON_LOGGING=1` environment variable is getting deprecated in the
next release. To enable JSON logging globally, use `RAY_SERVE_LOG_ENCODING=JSON`.
:::

#### Disable access log

:::{note}
Access log is Ray Serve traffic log, it is printed to proxy log files and replica log files per request. Sometimes it is useful for debugging, but it can also be noisy.
:::

You can also disable the access log by passing `disable_access_log=True` to `logging_config` argument of `@serve.deployment`. For example:

```{literalinclude} doc_code/monitoring/logging_config.py
:start-after: __enable_access_log_start__
:end-before:  __enable_access_log_end__
:language: python
```

The `Model` replica log file doesn't include the Serve traffic log, you should only see the application log in the log file.

```
# cat `ls /tmp/ray/session_latest/logs/serve/replica_default_Model_*`

INFO 2024-02-27 15:43:12,983 default_Model 4guj63jr replica.py:855 - Started initializing replica.
INFO 2024-02-27 15:43:12,984 default_Model 4guj63jr replica.py:877 - Finished initializing replica.
INFO 2024-02-27 15:43:13,492 default_Model 4guj63jr 2246c4bb-73dc-4524-bf37-c7746a6b3bba / <stdin>:5 - hello world
```

#### Configure logging in different deployments and applications
You can also configure logging at the application level by passing `logging_config` to `serve.run`. For example:

```{literalinclude} doc_code/monitoring/logging_config.py
:start-after: __application_and_deployment_start__
:end-before:  __application_and_deployment_end__
:language: python
```

In the Router log file, you should see the following:

```
# cat `ls /tmp/ray/session_latest/logs/serve/replica_default_Router_*`

INFO 2024-02-27 16:05:10,738 default_Router cwnihe65 replica.py:855 - Started initializing replica.
INFO 2024-02-27 16:05:10,739 default_Router cwnihe65 replica.py:877 - Finished initializing replica.
INFO 2024-02-27 16:05:11,233 default_Router cwnihe65 4db9445d-fc9e-490b-8bad-0a5e6bf30899 / replica.py:1068 - Started executing request to method '__call__'.
DEBUG 2024-02-27 16:05:11,234 default_Router cwnihe65 4db9445d-fc9e-490b-8bad-0a5e6bf30899 / <stdin>:7 - This debug message is from the router.
INFO 2024-02-27 16:05:11,238 default_Router cwnihe65 4db9445d-fc9e-490b-8bad-0a5e6bf30899 / router.py:308 - Using router <class 'ray.serve._private.replica_scheduler.pow_2_scheduler.PowerOfTwoChoicesReplicaScheduler'>.
DEBUG 2024-02-27 16:05:11,240 default_Router cwnihe65 long_poll.py:157 - LongPollClient <ray.serve._private.long_poll.LongPollClient object at 0x10daa5a80> received updates for keys: [(LongPollNamespace.DEPLOYMENT_CONFIG, DeploymentID(name='Model', app='default')), (LongPollNamespace.RUNNING_REPLICAS, DeploymentID(name='Model', app='default'))].
INFO 2024-02-27 16:05:11,241 default_Router cwnihe65 pow_2_scheduler.py:255 - Got updated replicas for deployment 'Model' in application 'default': {'default#Model#256v3hq4'}.
DEBUG 2024-02-27 16:05:11,241 default_Router cwnihe65 long_poll.py:157 - LongPollClient <ray.serve._private.long_poll.LongPollClient object at 0x10daa5900> received updates for keys: [(LongPollNamespace.DEPLOYMENT_CONFIG, DeploymentID(name='Model', app='default')), (LongPollNamespace.RUNNING_REPLICAS, DeploymentID(name='Model', app='default'))].
INFO 2024-02-27 16:05:11,245 default_Router cwnihe65 4db9445d-fc9e-490b-8bad-0a5e6bf30899 / replica.py:373 - __CALL__ OK 12.2ms
```

In the Model log file, you should see the following:

```
# cat `ls /tmp/ray/session_latest/logs/serve/replica_default_Model_*`

INFO 2024-02-27 16:05:10,735 default_Model 256v3hq4 replica.py:855 - Started initializing replica.
INFO 2024-02-27 16:05:10,735 default_Model 256v3hq4 replica.py:877 - Finished initializing replica.
INFO 2024-02-27 16:05:11,244 default_Model 256v3hq4 4db9445d-fc9e-490b-8bad-0a5e6bf30899 / replica.py:1068 - Started executing request to method '__call__'.
INFO 2024-02-27 16:05:11,244 default_Model 256v3hq4 4db9445d-fc9e-490b-8bad-0a5e6bf30899 / replica.py:373 - __CALL__ OK 0.6ms
```

When you set `logging_config` at the application level, Ray Serve applies to all deployments in the application. When you set `logging_config` at the deployment level at the same time, the deployment level configuration will overrides the application level configuration.

#### Configure logging for serve components
You can also update logging configuration similar above to the Serve controller and proxies by passing `logging_config` to `serve.start`.

```{literalinclude} doc_code/monitoring/logging_config.py
:start-after: __configure_serve_component_start__
:end-before:  __configure_serve_component_end__
:language: python
```

### Set Request ID
You can set a custom request ID for each HTTP request by including `X-Request-ID` in the request header and retrieve request ID from response. For example

```{literalinclude} doc_code/monitoring/request_id.py
:language: python
```
The custom request ID `123-234` can be seen in the access logs that are printed to the HTTP Proxy log files and deployment log files.

HTTP proxy log file:
```
INFO 2023-07-20 13:47:54,221 http_proxy 127.0.0.1 123-234 / default http_proxy.py:538 - GET 200 8.9ms
```

Deployment log file:
```
(ServeReplica:default_Model pid=84006) INFO 2023-07-20 13:47:54,218 default_Model default_Model#yptKoo 123-234 / default replica.py:691 - __CALL__ OK 0.2ms
```

(serve-logging-loki)=
### Filtering logs with Loki

You can explore and filter your logs using [Loki](https://grafana.com/oss/loki/).
Setup and configuration are straightforward on Kubernetes, but as a tutorial, let's set up Loki manually.

For this walkthrough, you need both Loki and Promtail, which are both supported by [Grafana Labs](https://grafana.com). Follow the installation instructions at Grafana's website to get executables for [Loki](https://grafana.com/docs/loki/latest/installation/) and [Promtail](https://grafana.com/docs/loki/latest/clients/promtail/).
For convenience, save the Loki and Promtail executables in the same directory, and then navigate to this directory in your terminal.

Now let's get your logs into Loki using Promtail.

Save the following file as `promtail-local-config.yaml`:

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://localhost:3100/loki/api/v1/push

scrape_configs:
  - job_name: ray
    static_configs:
      - labels:
        job: ray
        __path__: /tmp/ray/session_latest/logs/serve/*.*
```

The relevant part for Ray Serve is the `static_configs` field, where we have indicated the location of our log files with `__path__`.
The expression `*.*` will match all files, but it won't match directories since they cause an error with Promtail.

We'll run Loki locally.  Grab the default config file for Loki with the following command in your terminal:

```shell
wget https://raw.githubusercontent.com/grafana/loki/v2.1.0/cmd/loki/loki-local-config.yaml
```

Now start Loki:

```shell
./loki-darwin-amd64 -config.file=loki-local-config.yaml
```

Here you may need to replace `./loki-darwin-amd64` with the path to your Loki executable file, which may have a different name depending on your operating system.

Start Promtail and pass in the path to the config file we saved earlier:

```shell
./promtail-darwin-amd64 -config.file=promtail-local-config.yaml
```

Once again, you may need to replace `./promtail-darwin-amd64` with your Promtail executable.

Run the following Python script to deploy a basic Serve deployment with a Serve deployment logger and to make some requests:

```{literalinclude} doc_code/monitoring/deployment_logger.py
:start-after: __start__
:end-before: __end__
:language: python
```

Now [install and run Grafana](https://grafana.com/docs/grafana/latest/installation/) and navigate to `http://localhost:3000`, where you can log in with default credentials:

* Username: admin
* Password: admin

On the welcome page, click "Add your first data source" and click "Loki" to add Loki as a data source.

Now click "Explore" in the left-side panel.  You are ready to run some queries!

To filter all these Ray logs for the ones relevant to our deployment, use the following [LogQL](https://grafana.com/docs/loki/latest/logql/) query:

```shell
{job="ray"} |= "Counter"
```

You should see something similar to the following:

```{image} https://raw.githubusercontent.com/ray-project/Images/master/docs/serve/loki-serve.png
:align: center
```

You can use Loki to filter your Ray Serve logs and gather insights quicker.

(serve-production-monitoring-metrics)=

## Built-in Ray Serve metrics

You can leverage built-in Ray Serve metrics to get a closer look at your application's performance.

Ray Serve exposes important system metrics like the number of successful and
failed requests through the [Ray metrics monitoring infrastructure](dash-metrics-view). By default, the metrics are exposed in Prometheus format on each node.

:::{note}
Different metrics are collected when Deployments are called
via Python `DeploymentHandle` and when they are called via HTTP.

See the list of metrics below marked for each.
:::

The following metrics are exposed by Ray Serve:

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Name
     - Fields
     - Description
   * - ``ray_serve_deployment_request_counter_total`` [**]
     - * deployment
       * replica
       * route
       * application
     - The number of queries that have been processed in this replica.
   * - ``ray_serve_deployment_error_counter_total`` [**]
     - * deployment
       * replica
       * route
       * application
     - The number of exceptions that have occurred in the deployment.
   * - ``ray_serve_deployment_replica_starts_total`` [**]
     - * deployment
       * replica
       * application
     - The number of times this replica has been restarted due to failure.
   * - ``ray_serve_deployment_replica_healthy``
     - * deployment
       * replica
       * application
     - Whether this deployment replica is healthy. 1 means healthy, 0 unhealthy.
   * - ``ray_serve_deployment_processing_latency_ms`` [**]
     - * deployment
       * replica
       * route
       * application
     - The latency for queries to be processed.
   * - ``ray_serve_replica_processing_queries`` [**]
     - * deployment
       * replica
       * application
     - The current number of queries being processed.
   * - ``ray_serve_num_http_requests_total`` [*]
     - * route
       * method
       * application
       * status_code
     - The number of HTTP requests processed.
   * - ``ray_serve_num_grpc_requests_total`` [*]
     - * route
       * method
       * application
       * status_code
     - The number of gRPC requests processed.
   * - ``ray_serve_num_http_error_requests_total`` [*]
     - * route
       * error_code
       * method
       * application
     - The number of non-200 HTTP responses.
   * - ``ray_serve_num_grpc_error_requests_total`` [*]
     - * route
       * error_code
       * method
     - The number of non-OK gRPC responses.
   * - ``ray_serve_num_ongoing_http_requests`` [*]
     - * node_id
       * node_ip_address
     - The number of ongoing requests in the HTTP Proxy.
   * - ``ray_serve_num_ongoing_grpc_requests`` [*]
     - * node_id
       * node_ip_address
     - The number of ongoing requests in the gRPC Proxy.
   * - ``ray_serve_num_router_requests_total`` [*]
     - * deployment
       * route
       * application
       * handle
       * actor_id
     - The number of requests processed by the router.
   * - ``ray_serve_num_scheduling_tasks`` [*][†]
     - * deployment
       * actor_id
     - The number of request scheduling tasks in the router.
   * - ``ray_serve_num_scheduling_tasks_in_backoff`` [*][†]
     - * deployment
       * actor_id
     - The number of request scheduling tasks in the router that are undergoing backoff.
   * - ``ray_serve_handle_request_counter_total`` [**]
     - * handle
       * deployment
       * route
       * application
     - The number of requests processed by this DeploymentHandle.
   * - ``ray_serve_deployment_queued_queries`` [*]
     - * deployment
       * application
       * handle
       * actor_id
     - The current number of requests to this deployment that have been submitted to a replica.
   * - ``ray_serve_num_ongoing_requests_at_replicas`` [*]
     - * deployment
       * application
       * handle
       * actor_id
     - The current number of requests to this deployment that's been assigned and sent to execute on a replica.
   * - ``ray_serve_num_deployment_http_error_requests_total`` [*]
     - * deployment
       * error_code
       * method
       * route
       * application
     - The number of non-200 HTTP responses returned by each deployment.
   * - ``ray_serve_num_deployment_grpc_error_requests_total`` [*]
     - * deployment
       * error_code
       * method
       * route
       * application
     - The number of non-OK gRPC responses returned by each deployment.
   * - ``ray_serve_http_request_latency_ms`` [*]
     - * method
       * route
       * application
       * status_code
     - The end-to-end latency of HTTP requests (measured from the Serve HTTP proxy).
   * - ``ray_serve_grpc_request_latency_ms`` [*]
     - * method
       * route
       * application
       * status_code
     - The end-to-end latency of gRPC requests (measured from the Serve gRPC proxy).
   * - ``ray_serve_multiplexed_model_load_latency_ms``
     - * deployment
       * replica
       * application
     - The time it takes to load a model.
   * - ``ray_serve_multiplexed_model_unload_latency_ms``
     - * deployment
       * replica
       * application
     - The time it takes to unload a model.
   * - ``ray_serve_num_multiplexed_models``
     - * deployment
       * replica
       * application
     - The number of models loaded on the current replica.
   * - ``ray_serve_multiplexed_models_unload_counter_total``
     - * deployment
       * replica
       * application
     - The number of times models unloaded on the current replica.
   * - ``ray_serve_multiplexed_models_load_counter_total``
     - * deployment
       * replica
       * application
     - The number of times models loaded on the current replica.
   * - ``ray_serve_registered_multiplexed_model_id``
     - * deployment
       * replica
       * application
       * model_id
     - The mutliplexed model ID registered on the current replica.
   * - ``ray_serve_multiplexed_get_model_requests_counter_total``
     - * deployment
       * replica
       * application
     - The number of calls to get a multiplexed model.
```

[*] - only available when using proxy calls</br>
[**] - only available when using Python `DeploymentHandle` calls</br>
[†] - developer metrics for advanced usage; may change in future releases

To see this in action, first run the following command to start Ray and set up the metrics export port:

```bash
ray start --head --metrics-export-port=8080
```

Then run the following script:

```{literalinclude} doc_code/monitoring/metrics_snippet.py
:start-after: __start__
:end-before: __end__
:language: python
```

The requests loop until canceled with `Control-C`.

While this script is running, go to `localhost:8080` in your web browser.
In the output there, you can search for `serve_` to locate the metrics above.
The metrics are updated once every ten seconds, so you need to refresh the page to see new values.

For example, after running the script for some time and refreshing `localhost:8080` you should find metrics similar to the following:

```
ray_serve_deployment_processing_latency_ms_count{..., replica="sleeper#jtzqhX"} 48.0
ray_serve_deployment_processing_latency_ms_sum{..., replica="sleeper#jtzqhX"} 48160.6719493866
```

which indicates that the average processing latency is just over one second, as expected.

You can even define a [custom metric](application-level-metrics) for your deployment and tag it with deployment or replica metadata.
Here's an example:

```{literalinclude} doc_code/monitoring/custom_metric_snippet.py
:start-after: __start__
:end-before: __end__
```

The emitted logs include:

```
# HELP ray_my_counter_total The number of odd-numbered requests to this deployment.
# TYPE ray_my_counter_total counter
ray_my_counter_total{..., deployment="MyDeployment",model="123",replica="MyDeployment#rUVqKh"} 5.0
```

See the [Ray Metrics documentation](collect-metrics) for more details, including instructions for scraping these metrics using Prometheus.

## Profiling memory

Ray provides two useful metrics to track memory usage: `ray_component_rss_mb` (resident set size) and `ray_component_mem_shared_bytes` (shared memory). Approximate a Serve actor's memory usage by subtracting its shared memory from its resident set size (i.e. `ray_component_rss_mb` - `ray_component_mem_shared_bytes`).

If you notice a memory leak on a Serve actor, use `memray` to debug (`pip install memray`). Set the env var `RAY_SERVE_ENABLE_MEMORY_PROFILING=1`, and run your Serve application. All the Serve actors will run a `memray` tracker that logs their memory usage to `bin` files in the `/tmp/ray/session_latest/logs/serve/` directory. Run the `memray flamegraph [bin file]` command to generate a flamegraph of the memory usage. See the [memray docs](https://bloomberg.github.io/memray/overview.html) for more info.

## Exporting metrics into Arize
Besides using Prometheus to check out Ray metrics, Ray Serve also has the flexibility to export the metrics into other observability platforms.

[Arize](https://docs.arize.com/arize/) is a machine learning observability platform which can help you to monitor real-time model performance, root cause model failures/performance degradation using explainability & slice analysis and surface drift, data quality, data consistency issues etc.

To integrate with Arize, add Arize client code directly into your Serve deployment code. ([Example code](https://docs.arize.com/arize/integrations/integrations/anyscale-ray-serve))


(serve-resource-allocation)=

# Resource Allocation

This guide helps you configure Ray Serve to:

- Scale your deployments horizontally by specifying a number of replicas
- Scale up and down automatically to react to changing traffic
- Allocate hardware resources (CPUs, GPUs, etc) for each deployment


(serve-cpus-gpus)=

## Resource management (CPUs, GPUs)

You may want to specify a deployment's resource requirements to reserve cluster resources like GPUs.  To assign hardware resources per replica, you can pass resource requirements to
`ray_actor_options`.
By default, each replica reserves one CPU.
To learn about options to pass in, take a look at the [Resources with Actors guide](actor-resource-guide).

For example, to create a deployment where each replica uses a single GPU, you can do the
following:

```python
@serve.deployment(ray_actor_options={"num_gpus": 1})
def func(*args):
    return do_something_with_my_gpu()
```

(serve-fractional-resources-guide)=

### Fractional CPUs and fractional GPUs

Suppose you have two models and each doesn't fully saturate a GPU.  You might want to have them share a GPU by allocating 0.5 GPUs each.

To do this, the resources specified in `ray_actor_options` can be *fractional*.
For example, if you have two models and each doesn't fully saturate a GPU, you might want to have them share a GPU by allocating 0.5 GPUs each.

```python
@serve.deployment(ray_actor_options={"num_gpus": 0.5})
def func_1(*args):
    return do_something_with_my_gpu()

@serve.deployment(ray_actor_options={"num_gpus": 0.5})
def func_2(*args):
    return do_something_with_my_gpu()
```

In this example, each replica of each deployment will be allocated 0.5 GPUs.  The same can be done to multiplex over CPUs, using `"num_cpus"`.

### Custom resources, accelerator types, and more

You can also specify {ref}`custom resources <cluster-resources>` in `ray_actor_options`, for example to ensure that a deployment is scheduled on a specific node.
For example, if you have a deployment that requires 2 units of the `"custom_resource"` resource, you can specify it like this:

```python
@serve.deployment(ray_actor_options={"resources": {"custom_resource": 2}})
def func(*args):
    return do_something_with_my_custom_resource()
```

You can also specify {ref}`accelerator types <accelerator-types>` via the `accelerator_type` parameter in `ray_actor_options`.

Below is the full list of supported options in `ray_actor_options`; please see the relevant Ray Core documentation for more details about each option:

- `accelerator_type`
- `memory`
- `num_cpus`
- `num_gpus`
- `object_store_memory`
- `resources`
- `runtime_env`

(serve-omp-num-threads)=

## Configuring parallelism with OMP_NUM_THREADS

Deep learning models like PyTorch and Tensorflow often use multithreading when performing inference.
The number of CPUs they use is controlled by the `OMP_NUM_THREADS` environment variable.
Ray sets `OMP_NUM_THREADS=<num_cpus>` by default. To [avoid contention](omp-num-thread-note), Ray sets `OMP_NUM_THREADS=1` if `num_cpus` is not specified on the tasks/actors, to reduce contention between actors/tasks which run in a single thread.
If you *do* want to enable this parallelism in your Serve deployment, just set `num_cpus` (recommended) to the desired value, or manually set the `OMP_NUM_THREADS` environment variable when starting Ray or in your function/class definition.

```bash
OMP_NUM_THREADS=12 ray start --head
OMP_NUM_THREADS=12 ray start --address=$HEAD_NODE_ADDRESS
```

```{literalinclude} doc_code/managing_deployments.py
:start-after: __configure_parallism_start__
:end-before: __configure_parallism_end__
:language: python
```

:::{note}
Some other libraries may not respect `OMP_NUM_THREADS` and have their own way to configure parallelism.
For example, if you're using OpenCV, you'll need to manually set the number of threads using `cv2.setNumThreads(num_threads)` (set to 0 to disable multi-threading).
You can check the configuration using `cv2.getNumThreads()` and `cv2.getNumberOfCPUs()`.
:::


(serve-getting-started)=

# Getting Started

This tutorial will walk you through the process of writing and testing a Ray Serve application. It will show you how to

* convert a machine learning model to a Ray Serve deployment
* test a Ray Serve application locally over HTTP
* compose multiple-model machine learning models together into a single application

We'll use two models in this tutorial:

* [HuggingFace's TranslationPipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TranslationPipeline) as a text-translation model
* [HuggingFace's SummarizationPipeline](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/pipelines#transformers.SummarizationPipeline) as a text-summarizer model

You can also follow along using your own models from any Python framework.

After deploying those two models, we'll test them with HTTP requests.

:::{tip}
If you have suggestions on how to improve this tutorial,
    please [let us know](https://github.com/ray-project/ray/issues/new/choose)!
:::

To run this example, you will need to install the following:

```bash
pip install "ray[serve]" transformers requests torch
```


## Text Translation Model (before Ray Serve)

First, let's take a look at our text-translation model. Here's its code:

```{literalinclude} ../serve/doc_code/getting_started/models.py
:start-after: __start_translation_model__
:end-before: __end_translation_model__
:language: python
```

The Python file, called `model.py`, uses the `Translator` class to translate English text to French.

- The `self.model` variable inside `Translator`'s `__init__` method
  stores a function that uses the [t5-small](https://huggingface.co/t5-small)
  model to translate text.
- When `self.model` is called on English text, it returns translated French text
  inside a dictionary formatted as `[{"translation_text": "..."}]`.
- The `Translator`'s `translate` method extracts the translated text by indexing into the dictionary.

You can copy-paste this script and run it locally. It translates `"Hello world!"`
into `"Bonjour Monde!"`.

```console
$ python model.py

Bonjour Monde!
```

Keep in mind that the `TranslationPipeline` is an example ML model for this
tutorial. You can follow along using arbitrary models from any
Python framework. Check out our tutorials on scikit-learn,
PyTorch, and Tensorflow for more info and examples:

- {ref}`serve-ml-models-tutorial`

(converting-to-ray-serve-application)=
## Converting to a Ray Serve Application

In this section, we'll deploy the text translation model using Ray Serve, so
it can be scaled up and queried over HTTP. We'll start by converting
`Translator` into a Ray Serve deployment.

First, we open a new Python file and import `ray` and `ray.serve`:

```{literalinclude} ../serve/doc_code/getting_started/model_deployment.py
:start-after: __import_start__
:end-before: __import_end__
:language: python
```

After these imports, we can include our model code from above:

```{literalinclude} ../serve/doc_code/getting_started/model_deployment.py
:start-after: __model_start__
:end-before: __model_end__
:language: python
```

The `Translator` class has two modifications:
1. It has a decorator, `@serve.deployment`.
2. It has a new method, `__call__`.

The decorator converts `Translator` from a Python class into a Ray Serve `Deployment` object.

Each deployment stores a single Python function or class that you write and uses
it to serve requests. You can scale and configure each of your deployments independently using
parameters in the `@serve.deployment` decorator. The example configures a few common parameters:

* `num_replicas`: an integer that determines how many copies of our deployment process run in Ray. Requests are load balanced across these replicas, allowing you to scale your deployments horizontally.
* `ray_actor_options`: a dictionary containing configuration options for each replica.
    * `num_cpus`: a float representing the logical number of CPUs each replica should reserve. You can make this a fraction to pack multiple replicas together on a machine with fewer CPUs than replicas.
    * `num_gpus`: a float representing the logical number of GPUs each replica should reserve. You can make this a fraction to pack multiple replicas together on a machine with fewer GPUs than replicas.

All these parameters are optional, so feel free to omit them:

```python
...
@serve.deployment
class Translator:
  ...
```

Deployments receive Starlette HTTP `request` objects [^f1]. By default, the deployment class's `__call__` method is called on this `request` object. The return value is sent back in the HTTP response body.

This is why `Translator` needs a new `__call__` method. The method processes the incoming HTTP request by reading its JSON data and forwarding it to the `translate` method. The translated text is returned and sent back through the HTTP response. You can also use Ray Serve's FastAPI integration to avoid working with raw HTTP requests. Check out {ref}`serve-fastapi-http` for more info about FastAPI with Serve.

Next, we need to `bind` our `Translator` deployment to arguments that will be passed into its constructor. This defines a Ray Serve application that we can run locally or deploy to production (you'll see later that applications can consist of multiple deployments). Since `Translator`'s constructor doesn't take in any arguments, we can call the deployment's `bind` method without passing anything in:

```{literalinclude} ../serve/doc_code/getting_started/model_deployment.py
:start-after: __model_deploy_start__
:end-before: __model_deploy_end__
:language: python
```

With that, we are ready to test the application locally.

## Running a Ray Serve Application

Here's the full Ray Serve script that we built above:

```{literalinclude} ../serve/doc_code/getting_started/model_deployment_full.py
:start-after: __deployment_full_start__
:end-before: __deployment_full_end__
:language: python
```

To test locally, we run the script with the `serve run` CLI command. This command takes in an import path
to our deployment formatted as `module:application`. Make sure to run the command from a directory containing a local copy of this script saved as `serve_quickstart.py`, so it can import the application:

```console
$ serve run serve_quickstart:translator_app
```

This command will run the `translator_app` application and then block, streaming logs to the console. It can be killed with `Ctrl-C`, which will tear down the application.

We can now test our model over HTTP. It can be reached at the following URL by default:

```
http://127.0.0.1:8000/
```

We'll send a POST request with JSON data containing our English text.
`Translator`'s `__call__` method will unpack this text and forward it to the
`translate` method. Here's a client script that requests a translation for "Hello world!":

```{literalinclude} ../serve/doc_code/getting_started/model_deployment.py
:start-after: __client_function_start__
:end-before: __client_function_end__
:language: python
```

To test our deployment, first make sure `Translator` is running:

```
$ serve run serve_deployment:translator_app
```

While `Translator` is running, we can open a separate terminal window and run the client script. This will get a response over HTTP:

```console
$ python model_client.py

Bonjour monde!
```

## Composing Multiple Models

Ray Serve allows you to compose multiple deployments into a single Ray Serve application. This makes it easy to combine multiple machine learning models along with business logic to serve a single request.
We can use parameters like `autoscaling_config`, `num_replicas`, `num_cpus`, and `num_gpus` to independently configure and scale each deployment in the application.

For example, let's deploy a machine learning pipeline with two steps:

1. Summarize English text
2. Translate the summary into French

`Translator` already performs step 2. We can use [HuggingFace's SummarizationPipeline](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/pipelines#transformers.SummarizationPipeline) to accomplish step 1. Here's an example of the `SummarizationPipeline` that runs locally:

```{literalinclude} ../serve/doc_code/getting_started/models.py
:start-after: __start_summarization_model__
:end-before: __end_summarization_model__
:language: python
```

You can copy-paste this script and run it locally. It summarizes the snippet from _A Tale of Two Cities_ to `it was the best of times, it was worst of times .`

```console
$ python summary_model.py

it was the best of times, it was worst of times .
```

Here's an application that chains the two models together. The graph takes English text, summarizes it, and then translates it:

```{literalinclude} ../serve/doc_code/getting_started/translator.py
:start-after: __start_graph__
:end-before: __end_graph__
:language: python
```

This script contains our `Summarizer` class converted to a deployment and our `Translator` class with some modifications. In this script, the `Summarizer` class contains the `__call__` method since requests are sent to it first. It also takes in a handle to the `Translator` as one of its constructor arguments, so it can forward summarized texts to the `Translator` deployment. The `__call__` method also contains some new code:

```python
translation = await self.translator.translate.remote(summary)
```

`self.translator.translate.remote(summary)` issues an asynchronous call to the `Translator`'s `translate` method and returns a `DeploymentResponse` object immediately. Calling `await` on the response waits for the remote method call to execute and returns its return value. The response could also be passed directly to another `DeploymentHandle` call.

We define the full application as follows:

```python
app = Summarizer.bind(Translator.bind())
```

Here, we bind `Translator` to its (empty) constructor arguments, and then we pass in the bound `Translator` as the constructor argument for the `Summarizer`. We can run this deployment graph using the `serve run` CLI command. Make sure to run this command from a directory containing a local copy of the `serve_quickstart_composed.py` code:

```console
$ serve run serve_quickstart_composed:app
```

We can use this client script to make requests to the graph:

```{literalinclude} ../serve/doc_code/getting_started/translator.py
:start-after: __start_client__
:end-before: __end_client__
:language: python
```

While the application is running, we can open a separate terminal window and query it:

```console
$ python composed_client.py

c'était le meilleur des temps, c'était le pire des temps .
```

Composed Ray Serve applications let you deploy each part of your machine learning pipeline, such as inference and business logic steps, in separate deployments. Each of these deployments can be individually configured and scaled, ensuring you get maximal performance from your resources. See the guide on [model composition](serve-model-composition) to learn more.

## Next Steps

- Dive into the {doc}`key-concepts` to get a deeper understanding of Ray Serve.
- View details about your Serve application in the Ray Dashboard: {ref}`dash-serve-view`.
- Learn more about how to deploy your Ray Serve application to production: {ref}`serve-in-production`.
- Check more in-depth tutorials for popular machine learning frameworks: {doc}`examples`.

```{rubric} Footnotes
```

[^f1]: [Starlette](https://www.starlette.io/) is a web server framework used by Ray Serve.


(serve-autoscaling)=

# Ray Serve Autoscaling

Each [Ray Serve deployment](serve-key-concepts-deployment) has one [replica](serve-architecture-high-level-view) by default. This means there is one worker process running the model and serving requests. When traffic to your deployment increases, the single replica can become overloaded. To maintain high performance of your service, you need to scale out your deployment.

## Manual Scaling

Before jumping into autoscaling, which is more complex, the other option to consider is manual scaling. You can increase the number of replicas by setting a higher value for [num_replicas](serve-configure-deployment) in the deployment options through [in place updates](serve-inplace-updates). By default, `num_replicas` is 1. Increasing the number of replicas will horizontally scale out your deployment and improve latency and throughput for increased levels of traffic.

```yaml
# Deploy with a single replica
deployments:
- name: Model
  num_replicas: 1

# Scale up to 10 replicas
deployments:
- name: Model
  num_replicas: 10
```

## Autoscaling Basic Configuration

Instead of setting a fixed number of replicas for a deployment and manually updating it, you can configure a deployment to autoscale based on incoming traffic. The Serve autoscaler reacts to traffic spikes by monitoring queue sizes and making scaling decisions to add or remove replicas. Turn on autoscaling for a deployment by setting `num_replicas="auto"`. You can further configure it by tuning the [autoscaling_config](../serve/api/doc/ray.serve.config.AutoscalingConfig.rst) in deployment options.

The following config is what we will use in the example in the following section.
```yaml
- name: Model
  num_replicas: auto
```

Setting `num_replicas="auto"` is equivalent to the following deployment configuration.
```yaml
- name: Model
  max_ongoing_requests: 5
  autoscaling_config:
    target_ongoing_requests: 2
    min_replicas: 1
    max_replicas: 100
```
:::{note}
You can set `num_replicas="auto"` and override its default values (shown above) by specifying `autoscaling_config`, or you can omit `num_replicas="auto"` and fully configure autoscaling yourself.
:::

Let's dive into what each of these parameters do.

* **target_ongoing_requests** (replaces the deprecated `target_num_ongoing_requests_per_replica`) is the average number of ongoing requests per replica that the Serve autoscaler tries to ensure. You can adjust it based on your request processing length (the longer the requests, the smaller this number should be) as well as your latency objective (the shorter you want your latency to be, the smaller this number should be).
* **max_ongoing_requests** (replaces the deprecated `max_concurrent_queries`) is the maximum number of ongoing requests allowed for a replica. Note this parameter is not part of the autoscaling config because it's relevant to all deployments, but it's important to set it relative to the target value if you turn on autoscaling for your deployment.
* **min_replicas** is the minimum number of replicas for the deployment. Set this to 0 if there are long periods of no traffic and some extra tail latency during upscale is acceptable. Otherwise, set this to what you think you need for low traffic.
* **max_replicas** is the maximum number of replicas for the deployment. Set this to ~20% higher than what you think you need for peak traffic.

These guidelines are a great starting point. If you decide to further tune your autoscaling config for your application, see [Advanced Ray Serve Autoscaling](serve-advanced-autoscaling).

(resnet-autoscaling-example)=
## Basic example

This example is a synchronous workload that runs ResNet50. The application code and its autoscaling configuration are below. Alternatively, see the second tab for specifying the autoscaling config through a YAML file.

::::{tab-set}

:::{tab-item} Application Code
```{literalinclude} doc_code/resnet50_example.py
:language: python
:start-after: __serve_example_begin__
:end-before: __serve_example_end__
```
:::

:::{tab-item} (Alternative) YAML config

```yaml
applications:
  - name: default
    import_path: resnet:app
    deployments:
    - name: Model
      num_replicas: auto
```

:::
::::

This example uses [Locust](https://locust.io/) to run a load test against this application. The Locust load test runs a certain number of "users" that ping the ResNet50 service, where each user has a [constant wait time](https://docs.locust.io/en/stable/writing-a-locustfile.html#wait-time-attribute) of 0. Each user (repeatedly) sends a request, waits for a response, then immediately sends the next request. The number of users running over time is shown in the following graph:

![users](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/resnet50_users.png)

The results of the load test are as follows:

|  |  |  |
| -------- | --- | ------- |
| Replicas | <img src="https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/resnet50_replicas.png" alt="replicas" width="600"/> |
| QPS | <img src="https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/resnet50_rps.png" alt="qps"/> |
| P50 Latency | <img src="https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/resnet50_latency.png" alt="latency"/> |

Notice the following:
- Each Locust user constantly sends a single request and waits for a response. As a result, the number of autoscaled replicas is roughly half the number of Locust users over time as Serve attempts to satisfy the `target_ongoing_requests=2` setting.
- The throughput of the system increases with the number of users and replicas.
- The latency briefly spikes when traffic increases, but otherwise stays relatively steady.

## Ray Serve Autoscaler vs Ray Autoscaler

The Ray Serve Autoscaler is an application-level autoscaler that sits on top of the [Ray Autoscaler](cluster-index).
Concretely, this means that the Ray Serve autoscaler asks Ray to start a number of replica actors based on the request demand.
If the Ray Autoscaler determines there aren't enough available resources (e.g. CPUs, GPUs, etc.) to place these actors, it responds by requesting more Ray nodes.
The underlying cloud provider then responds by adding more nodes.
Similarly, when Ray Serve scales down and terminates replica Actors, it attempts to make as many nodes idle as possible so the Ray Autoscaler can remove them. To learn more about the architecture underlying Ray Serve Autoscaling, see [Ray Serve Autoscaling Architecture](serve-autoscaling-architecture).

(rayserve)=

# Ray Serve: Scalable and Programmable Serving

```{toctree}
:hidden:

getting_started
key-concepts
develop-and-deploy
model_composition
multi-app
model-multiplexing
configure-serve-deployment
http-guide
Production Guide <production-guide/index>
monitoring
resource-allocation
autoscaling-guide
advanced-guides/index
architecture
examples
api/index
```

:::{tip}
[Get in touch with us](https://docs.google.com/forms/d/1l8HT35jXMPtxVUtQPeGoe09VGp5jcvSv0TqPgyz6lGU) if you're using or considering using Ray Serve.
:::

```{image} logo.svg
:align: center
:height: 250px
:width: 400px
```

(rayserve-overview)=

Ray Serve is a scalable model serving library for building online inference APIs.
Serve is framework-agnostic, so you can use a single toolkit to serve everything from deep learning models built with frameworks like PyTorch, TensorFlow, and Keras, to Scikit-Learn models, to arbitrary Python business logic. It has several features and performance optimizations for serving Large Language Models such as response streaming, dynamic request batching, multi-node/multi-GPU serving, etc.

Ray Serve is particularly well suited for [model composition](serve-model-composition) and many model serving, enabling you to build a complex inference service consisting of multiple ML models and business logic all in Python code.

Ray Serve is built on top of Ray, so it easily scales to many machines and offers flexible scheduling support such as fractional GPUs so you can share resources and serve many machine learning models at low cost.

## Quickstart

Install Ray Serve and its dependencies:

```bash
pip install "ray[serve]"
```
Define a simple "hello world" application, run it locally, and query it over HTTP.

```{literalinclude} doc_code/quickstart.py
:language: python
```

## More examples

::::{tab-set}

:::{tab-item} Model composition

Use Serve's model composition API to combine multiple deployments into a single application.

```{literalinclude} doc_code/quickstart_composed.py
:language: python
```

:::

:::{tab-item} FastAPI integration

Use Serve's [FastAPI](https://fastapi.tiangolo.com/) integration to elegantly handle HTTP parsing and validation.

```{literalinclude} doc_code/fastapi_example.py
:language: python
```

:::

:::{tab-item} Hugging Face Transformers model

To run this example, install the following: ``pip install transformers``

Serve a pre-trained [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) model using Ray Serve.
The model we'll use is a sentiment analysis model: it will take a text string as input and return if the text was "POSITIVE" or "NEGATIVE."

```{literalinclude} doc_code/transformers_example.py
:language: python
```

:::

::::

## Why choose Serve?

:::{dropdown} Build end-to-end ML-powered applications
:animate: fade-in-slide-down

Many solutions for ML serving focus on "tensor-in, tensor-out" serving: that is, they wrap ML models behind a predefined, structured endpoint.
However, machine learning isn't useful in isolation.
It's often important to combine machine learning with business logic and traditional web serving logic such as database queries.

Ray Serve is unique in that it allows you to build and deploy an end-to-end distributed serving application in a single framework.
You can combine multiple ML models, business logic, and expressive HTTP handling using Serve's FastAPI integration (see {ref}`serve-fastapi-http`) to build your entire application as one Python program.

:::

:::{dropdown} Combine multiple models using a programmable API
:animate: fade-in-slide-down

Often solving a problem requires more than just a single machine learning model.
For instance, image processing applications typically require a multi-stage pipeline consisting of steps like preprocessing, segmentation, and filtering to achieve their end goal.
In many cases each model may use a different architecture or framework and require different resources (like CPUs vs GPUs).

Many other solutions support defining a static graph in YAML or some other configuration language.
This can be limiting and hard to work with.
Ray Serve, on the other hand, supports multi-model composition using a programmable API where calls to different models look just like function calls.
The models can use different resources and run across different machines in the cluster, but you can write it like a regular program. See {ref}`serve-model-composition` for more details.

:::

:::{dropdown} Flexibly scale up and allocate resources
:animate: fade-in-slide-down

Machine learning models are compute-intensive and therefore can be very expensive to operate.
A key requirement for any ML serving system is being able to dynamically scale up and down and allocate the right resources for each model to handle the request load while saving cost.

Serve offers a number of built-in primitives to help make your ML serving application efficient.
It supports dynamically scaling the resources for a model up and down by adjusting the number of replicas, batching requests to take advantage of efficient vectorized operations (especially important on GPUs), and a flexible resource allocation model that enables you to serve many models on limited hardware resources.

:::

:::{dropdown} Avoid framework or vendor lock-in
:animate: fade-in-slide-down

Machine learning moves fast, with new libraries and model architectures being released all the time, it's important to avoid locking yourself into a solution that is tied to a specific framework.
This is particularly important in serving, where making changes to your infrastructure can be time consuming, expensive, and risky.
Additionally, many hosted solutions are limited to a single cloud provider which can be a problem in today's multi-cloud world.

Ray Serve is not tied to any specific machine learning library or framework, but rather provides a general-purpose scalable serving layer.
Because it's built on top of Ray, you can run it anywhere Ray can: on your laptop, Kubernetes, any major cloud provider, or even on-premise.

:::


## How can Serve help me as a...

:::{dropdown} Data scientist
:animate: fade-in-slide-down

Serve makes it easy to go from a laptop to a cluster. You can test your models (and your entire deployment graph) on your local machine before deploying it to production on a cluster. You don't need to know heavyweight Kubernetes concepts or cloud configurations to use Serve.

:::

:::{dropdown} ML engineer
:animate: fade-in-slide-down

Serve helps you scale out your deployment and runs them reliably and efficiently to save costs. With Serve's first-class model composition API, you can combine models together with business logic and build end-to-end user-facing applications. Additionally, Serve runs natively on Kubernetes with minimal operation overhead.
:::

:::{dropdown} ML platform engineer
:animate: fade-in-slide-down

Serve specializes in scalable and reliable ML model serving. As such, it can be an important plug-and-play component of your ML platform stack.
Serve supports arbitrary Python code and therefore integrates well with the MLOps ecosystem. You can use it with model optimizers (ONNX, TVM), model monitoring systems (Seldon Alibi, Arize), model registries (MLFlow, Weights and Biases), machine learning frameworks (XGBoost, Scikit-learn), data app UIs (Gradio, Streamlit), and Web API frameworks (FastAPI, gRPC).

:::

:::{dropdown} LLM developer
:animate: fade-in-slide-down

Serve enables you to rapidly prototype, develop, and deploy scalable LLM applications to production. Many large language model (LLM) applications combine prompt preprocessing, vector database lookups, LLM API calls, and response validation. Because Serve supports any arbitrary Python code, you can write all these steps as a single Python module, enabling rapid development and easy testing. You can then quickly deploy your Ray Serve LLM application to production, and each application step can independently autoscale to efficiently accommodate user traffic without wasting resources. In order to improve performance of your LLM applications, Ray Serve has features for batching and can integrate with any model optimization technique. Ray Serve also supports streaming responses, a key feature for chatbot-like applications.

:::


## How does Serve compare to ...

:::{dropdown} TFServing, TorchServe, ONNXRuntime
:animate: fade-in-slide-down

Ray Serve is *framework-agnostic*, so you can use it alongside any other Python framework or library.
We believe data scientists should not be bound to a particular machine learning framework.
They should be empowered to use the best tool available for the job.

Compared to these framework-specific solutions, Ray Serve doesn't perform any model-specific optimizations to make your ML model run faster. However, you can still optimize the models yourself
and run them in Ray Serve. For example, you can run a model compiled by
[PyTorch JIT](https://pytorch.org/docs/stable/jit.html) or [ONNXRuntime](https://onnxruntime.ai/).
:::

:::{dropdown} AWS SageMaker, Azure ML, Google Vertex AI
:animate: fade-in-slide-down

As an open-source project, Ray Serve brings the scalability and reliability of these hosted offerings to your own infrastructure.
You can use the Ray [cluster launcher](cluster-index) to deploy Ray Serve to all major public clouds, K8s, as well as on bare-metal, on-premise machines.

Ray Serve is not a full-fledged ML Platform.
Compared to these other offerings, Ray Serve lacks the functionality for
managing the lifecycle of your models, visualizing their performance, etc. Ray
Serve primarily focuses on model serving and providing the primitives for you to
build your own ML platform on top.

:::

:::{dropdown} Seldon, KServe, Cortex
:animate: fade-in-slide-down

You can develop Ray Serve on your laptop, deploy it on a dev box, and scale it out
to multiple machines or a Kubernetes cluster, all with minimal or no changes to code. It's a lot
easier to get started with when you don't need to provision and manage a K8s cluster.
When it's time to deploy, you can use our [Kubernetes Operator](kuberay-quickstart)
to transparently deploy your Ray Serve application to K8s.
:::

:::{dropdown} BentoML, Comet.ml, MLflow
:animate: fade-in-slide-down

Many of these tools are focused on serving and scaling models independently.
In contrast, Ray Serve is framework-agnostic and focuses on model composition.
As such, Ray Serve works with any model packaging and registry format.
Ray Serve also provides key features for building production-ready machine learning applications, including best-in-class autoscaling and naturally integrating with business logic.
:::

We truly believe Serve is unique as it gives you end-to-end control
over your ML application while delivering scalability and high performance. To achieve
Serve's feature offerings with other tools, you would need to glue together multiple
frameworks like Tensorflow Serving and SageMaker, or even roll your own
micro-batching component to improve throughput.

## Learn More

Check out {ref}`serve-getting-started` and {ref}`serve-key-concepts`,
or head over to the {doc}`examples` to get started building your Ray Serve applications.


```{eval-rst}
.. grid:: 1 2 2 2
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        **Getting Started**
        ^^^

        Start with our quick start tutorials for :ref:`deploying a single model locally <serve-getting-started>` and how to :ref:`convert an existing model into a Ray Serve deployment <converting-to-ray-serve-application>` .

        +++
        .. button-ref:: serve-getting-started
            :color: primary
            :outline:
            :expand:

            Get Started with Ray Serve

    .. grid-item-card::
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        **Key Concepts**
        ^^^

        Understand the key concepts behind Ray Serve.
        Learn about :ref:`Deployments <serve-key-concepts-deployment>`, :ref:`how to query them <serve-key-concepts-ingress-deployment>`, and using :ref:`DeploymentHandles <serve-key-concepts-deployment-handle>` to compose multiple models and business logic together.

        +++
        .. button-ref:: serve-key-concepts
            :color: primary
            :outline:
            :expand:

            Learn Key Concepts

    .. grid-item-card::
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        **Examples**
        ^^^

        Follow the tutorials to learn how to integrate Ray Serve with :ref:`TensorFlow <serve-ml-models-tutorial>`, and :ref:`Scikit-Learn <serve-ml-models-tutorial>`.

        +++
        .. button-ref:: examples
            :color: primary
            :outline:
            :expand:
            :ref-type: doc

            Serve Examples

    .. grid-item-card::
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        **API Reference**
        ^^^

        Get more in-depth information about the Ray Serve API.

        +++
        .. button-ref:: serve-api
            :color: primary
            :outline:
            :expand:

            Read the API Reference

```

For more, see the following blog posts about Ray Serve:

- [Serving ML Models in Production: Common Patterns](https://www.anyscale.com/blog/serving-ml-models-in-production-common-patterns) by Simon Mo, Edward Oakes, and Michael Galarnyk
- [The Simplest Way to Serve your NLP Model in Production with Pure Python](https://medium.com/distributed-computing-with-ray/the-simplest-way-to-serve-your-nlp-model-in-production-with-pure-python-d42b6a97ad55) by Edward Oakes and Bill Chambers
- [Machine Learning Serving is Broken](https://medium.com/distributed-computing-with-ray/machine-learning-serving-is-broken-f59aff2d607f) by Simon Mo
- [How to Scale Up Your FastAPI Application Using Ray Serve](https://medium.com/distributed-computing-with-ray/how-to-scale-up-your-fastapi-application-using-ray-serve-c9a7b69e786) by Archit Kulkarni


(serve-set-up-fastapi-http)=
# Set Up FastAPI and HTTP

This section helps you understand how to:
- Send HTTP requests to Serve deployments
- Use Ray Serve to integrate with FastAPI
- Use customized HTTP adapters
- Choose which feature to use for your use case
- Set up keep alive timeout

## Choosing the right HTTP feature

Serve offers a layered approach to expose your model with the right HTTP API.

Considering your use case, you can choose the right level of abstraction:
- If you are comfortable working with the raw request object, use [`starlette.request.Requests` API](serve-http).
- If you want a fully fledged API server with validation and doc generation, use the [FastAPI integration](serve-fastapi-http).


(serve-http)=
## Calling Deployments via HTTP
When you deploy a Serve application, the [ingress deployment](serve-key-concepts-ingress-deployment) (the one passed to `serve.run`) is exposed over HTTP.

```{literalinclude} doc_code/http_guide/http_guide.py
:start-after: __begin_starlette__
:end-before: __end_starlette__
:language: python
```

Requests to the Serve HTTP server at `/` are routed to the deployment's `__call__` method with a [Starlette Request object](https://www.starlette.io/requests/) as the sole argument. The `__call__` method can return any JSON-serializable object or a [Starlette Response object](https://www.starlette.io/responses/) (e.g., to return a custom status code or custom headers). A Serve app's route prefix can be changed from `/` to another string by setting `route_prefix` in `serve.run()` or the Serve config file.

(serve-request-cancellation-http)=
### Request cancellation

When processing a request takes longer than the [end-to-end timeout](serve-performance-e2e-timeout) or an HTTP client disconnects before receiving a response, Serve cancels the in-flight request:

- If the proxy hasn't yet sent the request to a replica, Serve simply drops the request.
- If the request has been sent to a replica, Serve attempts to interrupt the replica and cancel the request. The `asyncio.Task` running the handler on the replica is cancelled, raising an `asyncio.CancelledError` the next time it enters an `await` statement. See [the asyncio docs](https://docs.python.org/3/library/asyncio-task.html#task-cancellation) for more info. Handle this exception in a try-except block to customize your deployment's behavior when a request is cancelled:

```{literalinclude} doc_code/http_guide/disconnects.py
:start-after: __start_basic_disconnect__
:end-before: __end_basic_disconnect__
:language: python
```

If no `await` statements are left in the deployment's code before the request completes, the replica processes the request as usual, sends the response back to the proxy, and the proxy discards the response. Use `await` statements for blocking operations in a deployment, so Serve can cancel in-flight requests without waiting for the blocking operation to complete.

Cancellation cascades to any downstream deployment handle, task, or actor calls that were spawned in the deployment's request-handling method. These can handle the `asyncio.CancelledError` in the same way as the ingress deployment.

To prevent an async call from being interrupted by `asyncio.CancelledError`, use `asyncio.shield()`:

```{literalinclude} doc_code/http_guide/disconnects.py
:start-after: __start_shielded_disconnect__
:end-before: __end_shielded_disconnect__
:language: python
```

When the request is cancelled, a cancellation error is raised inside the `SnoringSleeper` deployment's `__call__()` method. However, the cancellation is not raised inside the `snore()` call, so `ZZZ` is printed even if the request is cancelled. Note that `asyncio.shield` cannot be used on a `DeploymentHandle` call to prevent the downstream handler from being cancelled. You need to explicitly handle the cancellation error in that handler as well.

(serve-fastapi-http)=
## FastAPI HTTP Deployments

If you want to define more complex HTTP handling logic, Serve integrates with [FastAPI](https://fastapi.tiangolo.com/). This allows you to define a Serve deployment using the {mod}`@serve.ingress <ray.serve.ingress>` decorator that wraps a FastAPI app with its full range of features. The most basic example of this is shown below, but for more details on all that FastAPI has to offer such as variable routes, automatic type validation, dependency injection (e.g., for database connections), and more, please check out [their documentation](https://fastapi.tiangolo.com/).

:::{note}
A Serve application that's integrated with FastAPI still respects the `route_prefix` set through Serve. The routes are that registered through the FastAPI `app` object are layered on top of the route prefix. For instance, if your Serve application has `route_prefix = /my_app` and you decorate a method with `@app.get("/fetch_data")`, then you can call that method by sending a GET request to the path `/my_app/fetch_data`.
:::
```{literalinclude} doc_code/http_guide/http_guide.py
:start-after: __begin_fastapi__
:end-before: __end_fastapi__
:language: python
```

Now if you send a request to `/hello`, this will be routed to the `root` method of our deployment. We can also easily leverage FastAPI to define multiple routes with different HTTP methods:

```{literalinclude} doc_code/http_guide/http_guide.py
:start-after: __begin_fastapi_multi_routes__
:end-before: __end_fastapi_multi_routes__
:language: python
```

You can also pass in an existing FastAPI app to a deployment to serve it as-is:

```{literalinclude} doc_code/http_guide/http_guide.py
:start-after: __begin_byo_fastapi__
:end-before: __end_byo_fastapi__
:language: python
```

This is useful for scaling out an existing FastAPI app with no modifications necessary.
Existing middlewares, **automatic OpenAPI documentation generation**, and other advanced FastAPI features should work as-is.

### WebSockets

Serve supports WebSockets via FastAPI:

```{literalinclude} doc_code/http_guide/websockets_example.py
:start-after: __websocket_serve_app_start__
:end-before: __websocket_serve_app_end__
:language: python
```

Decorate the function that handles WebSocket requests with `@app.websocket`. Read more about FastAPI WebSockets in the [FastAPI documentation](https://fastapi.tiangolo.com/advanced/websockets/).

Query the deployment using the `websockets` package (`pip install websockets`):

```{literalinclude} doc_code/http_guide/websockets_example.py
:start-after: __websocket_serve_client_start__
:end-before: __websocket_serve_client_end__
:language: python
```

(serve-http-streaming-response)=
## Streaming Responses

Some applications must stream incremental results back to the caller.
This is common for text generation using large language models (LLMs) or video processing applications.
The full forward pass may take multiple seconds, so providing incremental results as they're available provides a much better user experience.

To use HTTP response streaming, return a [StreamingResponse](https://www.starlette.io/responses/#streamingresponse) that wraps a generator from your HTTP handler.
This is supported for basic HTTP ingress deployments using a `__call__` method and when using the [FastAPI integration](serve-fastapi-http).

The code below defines a Serve application that incrementally streams numbers up to a provided `max`.
The client-side code is also updated to handle the streaming outputs.
This code uses the `stream=True` option to the [requests](https://requests.readthedocs.io/en/latest/user/advanced.html#streaming-requests) library.

```{literalinclude} doc_code/http_guide/streaming_example.py
:start-after: __begin_example__
:end-before: __end_example__
:language: python
```

Save this code in `stream.py` and run it:

```bash
$ python stream.py
[2023-05-25 10:44:23]  INFO ray._private.worker::Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
(ServeController pid=40401) INFO 2023-05-25 10:44:25,296 controller 40401 deployment_state.py:1259 - Deploying new version of deployment default_StreamingResponder.
(ProxyActor pid=40403) INFO:     Started server process [40403]
(ServeController pid=40401) INFO 2023-05-25 10:44:25,333 controller 40401 deployment_state.py:1498 - Adding 1 replica to deployment default_StreamingResponder.
Got result 0.0s after start: '0'
Got result 0.1s after start: '1'
Got result 0.2s after start: '2'
Got result 0.3s after start: '3'
Got result 0.4s after start: '4'
Got result 0.5s after start: '5'
Got result 0.6s after start: '6'
Got result 0.7s after start: '7'
Got result 0.8s after start: '8'
Got result 0.9s after start: '9'
(ServeReplica:default_StreamingResponder pid=41052) INFO 2023-05-25 10:49:52,230 default_StreamingResponder default_StreamingResponder#qlZFCa yomKnJifNJ / default replica.py:634 - __CALL__ OK 1017.6ms
```

### Terminating the stream when a client disconnects

In some cases, you may want to cease processing a request when the client disconnects before the full stream has been returned.
If you pass an async generator to `StreamingResponse`, it is cancelled and raises an `asyncio.CancelledError` when the client disconnects.
Note that you must `await` at some point in the generator for the cancellation to occur.

In the example below, the generator streams responses forever until the client disconnects, then it prints that it was cancelled and exits. Save this code in `stream.py` and run it:


```{literalinclude} doc_code/http_guide/streaming_example.py
:start-after: __begin_cancellation__
:end-before: __end_cancellation__
:language: python
```

```bash
$ python stream.py
[2023-07-10 16:08:41]  INFO ray._private.worker::Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
(ServeController pid=50801) INFO 2023-07-10 16:08:42,296 controller 40401 deployment_state.py:1259 - Deploying new version of deployment default_StreamingResponder.
(ProxyActor pid=50803) INFO:     Started server process [50803]
(ServeController pid=50805) INFO 2023-07-10 16:08:42,963 controller 50805 deployment_state.py:1586 - Adding 1 replica to deployment default_StreamingResponder.
Got result 0.0s after start: '0'
Got result 0.1s after start: '1'
Got result 0.2s after start: '2'
Got result 0.3s after start: '3'
Got result 0.4s after start: '4'
Got result 0.5s after start: '5'
Got result 0.6s after start: '6'
Got result 0.7s after start: '7'
Got result 0.8s after start: '8'
Got result 0.9s after start: '9'
Got result 1.0s after start: '10'
Client disconnecting
(ServeReplica:default_StreamingResponder pid=50842) Cancelled! Exiting.
(ServeReplica:default_StreamingResponder pid=50842) INFO 2023-07-10 16:08:45,756 default_StreamingResponder default_StreamingResponder#cmpnmF ahteNDQSWx / default replica.py:691 - __CALL__ OK 1019.1ms
```


(serve-http-guide-keep-alive-timeout)=
## Set keep alive timeout

Serve uses a Uvicorn HTTP server internally to serve HTTP requests. By default, Uvicorn
keeps HTTP connections alive for 5 seconds between requests. Modify the keep-alive
timeout by setting the `keep_alive_timeout_s` in the `http_options` field of the Serve
config files. This config is global to your Ray cluster, and you can't update it during
runtime. You can also set the `RAY_SERVE_HTTP_KEEP_ALIVE_TIMEOUT_S` environment variable to
set the keep alive timeout. `RAY_SERVE_HTTP_KEEP_ALIVE_TIMEOUT_S` takes
precedence over the `keep_alive_timeout_s` config if both are set. See
Uvicorn's keep alive timeout [guide](https://www.uvicorn.org/server-behavior/#timeouts) for more information.


(serve-configure-deployment)=

# Configure Ray Serve deployments

Ray Serve default values for deployments are a good starting point for exploration. To further tailor scaling behavior, resource management, or performance tuning, you can configure parameters to alter the default behavior of Ray Serve deployments.

Use this guide to learn the essentials of configuring deployments:
- What parameters you can configure for a Ray Serve deployment
- The different locations where you can specify the parameters.

## Configurable parameters

You can also refer to the [API reference](../serve/api/doc/ray.serve.deployment_decorator.rst) for the `@serve.deployment` decorator.

- `name` - Name uniquely identifying this deployment within the application. If not provided, the name of the class or function is used.
- `num_replicas` - Controls the number of replicas to run that handle requests to this deployment. This can be a positive integer, in which case the number of replicas stays constant, or `auto`, in which case the number of replicas will autoscale with a default configuration (see [Ray Serve Autoscaling](serve-autoscaling) for more). Defaults to 1.
- `ray_actor_options` - Options to pass to the Ray Actor decorator, such as resource requirements. Valid options are: `accelerator_type`, `memory`, `num_cpus`, `num_gpus`, `object_store_memory`, `resources`, and `runtime_env` For more details - [Resource management in Serve](serve-cpus-gpus)
- `max_ongoing_requests` (replaces the deprecated `max_concurrent_queries`) - Maximum number of queries that are sent to a replica of this deployment without receiving a response. Defaults to 100 (the default will change to 5 in an upcoming release). This may be an important parameter to configure for [performance tuning](serve-perf-tuning).
- `autoscaling_config` - Parameters to configure autoscaling behavior. If this is set, you can't set `num_replicas` to a number. For more details on configurable parameters for autoscaling, see [Ray Serve Autoscaling](serve-autoscaling). 
- `user_config` -  Config to pass to the reconfigure method of the deployment. This can be updated dynamically without restarting the replicas of the deployment. The user_config must be fully JSON-serializable. For more details, see [Serve User Config](serve-user-config). 
- `health_check_period_s` - Duration between health check calls for the replica. Defaults to 10s. The health check is by default a no-op Actor call to the replica, but you can define your own health check using the "check_health" method in your deployment that raises an exception when unhealthy.
- `health_check_timeout_s` - Duration in seconds, that replicas wait for a health check method to return before considering it as failed. Defaults to 30s.
- `graceful_shutdown_wait_loop_s` - Duration that replicas wait until there is no more work to be done before shutting down. Defaults to 2s.
- `graceful_shutdown_timeout_s` - Duration to wait for a replica to gracefully shut down before being forcefully killed. Defaults to 20s.
- `logging_config` - Logging Config for the deployment (e.g. log level, log directory, JSON log format and so on). See [LoggingConfig](../serve/api/doc/ray.serve.schema.LoggingConfig.rst) for details.

## How to specify parameters

You can specify the above mentioned parameters in two locations:
1. In your application code.
2. In the Serve Config file, which is the recommended method for production.

### Specify parameters through the application code

You can specify parameters in the application code in two ways:
- In the `@serve.deployment` decorator when you first define a deployment
- With the `options()` method when you want to modify a deployment

Use the `@serve.deployment` decorator to specify deployment parameters when you are defining a deployment for the first time:

```{literalinclude} ../serve/doc_code/configure_serve_deployment/model_deployment.py
:start-after: __deployment_start__
:end-before: __deployment_end__
:language: python
```

Use the [`.options()`](../serve/api/doc/ray.serve.Deployment.rst) method to modify deployment parameters on an already-defined deployment. Modifying an existing deployment lets you reuse deployment definitions and dynamically set parameters at runtime.

```{literalinclude} ../serve/doc_code/configure_serve_deployment/model_deployment.py
:start-after: __deployment_end__
:end-before: __options_end__
:language: python
```

### Specify parameters through the Serve config file

In production, we recommend configuring individual deployments through the Serve config file. You can change parameter values without modifying your application code. Learn more about how to use the Serve Config in the [production guide](serve-in-production-config-file).

```yaml
applications:
- name: app1
  import_path: configure_serve:translator_app
  deployments:
  - name: Translator
    num_replicas: 2
    max_ongoing_requests: 100
    graceful_shutdown_wait_loop_s: 2.0
    graceful_shutdown_timeout_s: 20.0
    health_check_period_s: 10.0
    health_check_timeout_s: 30.0
    ray_actor_options:
      num_cpus: 0.2
      num_gpus: 0.0
```

### Order of Priority

You can set parameters to different values in various locations. For each individual parameter, the order of priority is (from highest to lowest):

1. Serve Config file
2. Application code (either through the `@serve.deployment` decorator or through `.options()`)
3. Serve defaults

In other words, if you specify a parameter for a deployment in the config file and the application code, Serve uses the config file's value. If it's only specified in the code, Serve uses the value you specified in the code. If you don't specify the parameter anywhere, Serve uses the default for that parameter.

For example, the following application code contains a single deployment `ExampleDeployment`:

```python
@serve.deployment(num_replicas=2, graceful_shutdown_timeout_s=6)
class ExampleDeployment:
    ...

example_app = ExampleDeployment.bind()
```

Then you deploy the application with the following config file:

```yaml
applications:
  - name: default
    import_path: models:example_app 
    deployments:
      - name: ExampleDeployment
        num_replicas: 5
```

Serve uses `num_replicas=5` from the value set in the config file and `graceful_shutdown_timeout_s=6` from the value set in the application code. All other deployment settings use Serve defaults because you didn't specify them in the code or the config. For instance, `health_check_period_s=10` because by default Serve health checks deployments once every 10 seconds.

:::{tip}
Remember that `ray_actor_options` counts as a single setting. The entire `ray_actor_options` dictionary in the config file overrides the entire `ray_actor_options` dictionary from the graph code. If you set individual options within `ray_actor_options` (e.g. `runtime_env`, `num_gpus`, `memory`) in the code but not in the config, Serve still won't use the code settings if the config has a `ray_actor_options` dictionary. It treats these missing options as though the user never set them and uses defaults instead. This dictionary overriding behavior also applies to `user_config` and `autoscaling_config`.
:::



(serve-architecture)=

# Architecture

In this section, we explore Serve's key architectural concepts and components. It will offer insight and overview into:
- the role of each component in Serve and how they work
- the different types of actors that make up a Serve application

% Figure source: https://docs.google.com/drawings/d/1jSuBN5dkSj2s9-0eGzlU_ldsRa3TsswQUZM-cMQ29a0/edit?usp=sharing

```{image} architecture-2.0.svg
:align: center
:width: 600px
```

(serve-architecture-high-level-view)=
## High-Level View

Serve runs on Ray and utilizes [Ray actors](actor-guide).

There are three kinds of actors that are created to make up a Serve instance:

- **Controller**: A global actor unique to each Serve instance that manages
  the control plane. The Controller is responsible for creating, updating, and
  destroying other actors. Serve API calls like creating or getting a deployment
  make remote calls to the Controller.
- **HTTP Proxy**: By default there is one HTTP proxy actor on the head node. This actor runs a [Uvicorn](https://www.uvicorn.org/) HTTP
  server that accepts incoming requests, forwards them to replicas, and
  responds once they are completed.  For scalability and high availability,
  you can also run a proxy on each node in the cluster via the `proxy_location` field inside [`serve.start()`](core-apis) or [the config file](serve-in-production-config-file).
- **gRPC Proxy**: If Serve is started with valid `port` and `grpc_servicer_functions`,
  then the gRPC proxy is started alongside with the HTTP proxy. This Actor runs a
  [grpcio](https://grpc.github.io/grpc/python/) server. The gRPC server that accepts
  incoming requests, forwards them to replicas, and responds once they are completed.
- **Replicas**: Actors that actually execute the code in response to a
  request. For example, they may contain an instantiation of an ML model. Each
  replica processes individual requests from the proxy. The replica may batch the requests
  using `@serve.batch`. See the [batching](serve-performance-batching-requests) docs.

## Lifetime of a request

When an HTTP or gRPC request is sent to the corresponding HTTP or gRPC proxy, the following happens:

1. The request is received and parsed.
2. Ray Serve looks up the correct deployment associated with the HTTP URL path or
  application name metadata. Serve places the request in a queue.
3. For each request in a deployment's queue, an available replica is looked up
  and the request is sent to it. If no replicas are available (that is, more
  than `max_ongoing_requests` requests are outstanding at each replica), the request
  is left in the queue until a replica becomes available.

Each replica maintains a queue of requests and executes requests one at a time, possibly
using `asyncio` to process them concurrently. If the handler (the deployment function or the `__call__` method of the deployment class) is declared with `async def`, the replica will not wait for the
handler to run.  Otherwise, the replica blocks until the handler returns.

When making a request via a [DeploymentHandle](serve-key-concepts-deployment-handle) instead of HTTP or gRPC for [model composition](serve-model-composition), the request is placed on a queue in the `DeploymentHandle`, and we skip to step 3 above.

(serve-ft-detail)=

## Fault tolerance

Application errors like exceptions in your model evaluation code are caught and
wrapped. A 500 status code will be returned with the traceback information. The
replica will be able to continue to handle requests.

Machine errors and faults are handled by Ray Serve as follows:

- When replica Actors fail, the Controller Actor replaces them with new ones.
- When the proxy Actor fails, the Controller Actor restarts it.
- When the Controller Actor fails, Ray restarts it.
- When using the [KubeRay RayService](kuberay-rayservice-quickstart), KubeRay recovers crashed nodes or a crashed cluster. You can avoid cluster crashes by using the [GCS FT feature](kuberay-gcs-ft).
- If you aren't using KubeRay, when the Ray cluster fails, Ray Serve cannot recover.

When a machine hosting any of the actors crashes, those actors are automatically restarted on another
available machine. All data in the Controller (routing policies, deployment
configurations, etc) is checkpointed to the Ray Global Control Store (GCS) on the head node. Transient data in the
router and the replica (like network connections and internal request queues) will be lost for this kind of failure.
See [the end-to-end fault tolerance guide](serve-e2e-ft) for more details on how actor crashes are detected.

(serve-autoscaling-architecture)=

## Ray Serve Autoscaling

Ray Serve's autoscaling feature automatically increases or decreases a deployment's number of replicas based on its load.

![pic](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling.svg)

- The Serve Autoscaler runs in the Serve Controller actor.
- Each `DeploymentHandle` and each replica periodically pushes its metrics to the autoscaler.
- For each deployment, the autoscaler periodically checks `DeploymentHandle` queues and in-flight queries on replicas to decide whether or not to scale the number of replicas.
- Each `DeploymentHandle` continuously polls the controller to check for new deployment replicas. Whenever new replicas are discovered, it sends any buffered or new queries to the replica until `max_ongoing_requests` is reached.  Queries are sent to replicas in round-robin fashion, subject to the constraint that no replica is handling more than `max_ongoing_requests` requests at a time.

:::{note}
When the controller dies, requests can still be sent via HTTP, gRPC and `DeploymentHandle`, but autoscaling is paused. When the controller recovers, the autoscaling resumes, but all previous metrics collected are lost.
:::

## Ray Serve API Server

Ray Serve provides a [CLI](serve-cli) for managing your Ray Serve instance, as well as a [REST API](serve-rest-api).
Each node in your Ray cluster provides a Serve REST API server that can connect to Serve and respond to Serve REST requests.

## FAQ

### How does Serve ensure horizontal scalability and availability?

You can configure Serve to start one proxy Actor per node with the `proxy_location` field inside [`serve.start()`](core-apis) or [the config file](serve-in-production-config-file). Each proxy binds to the same port. You
should be able to reach Serve and send requests to any models with any of the
servers.  You can use your own load balancer on top of Ray Serve.

This architecture ensures horizontal scalability for Serve. You can scale your HTTP and gRPC ingress by adding more nodes. You can also scale your model inference by increasing the number
of replicas via the `num_replicas` option of your deployment.

### How do DeploymentHandles work?

{mod}`DeploymentHandles <ray.serve.handle.DeploymentHandle>` wrap a handle to a "router" on the
same node which routes requests to replicas for a deployment. When a
request is sent from one replica to another via the handle, the
requests go through the same data path as incoming HTTP or gRPC requests. This enables
the same deployment selection and batching procedures to happen. DeploymentHandles are
often used to implement [model composition](serve-model-composition).

### What happens to large requests?

Serve utilizes Ray’s [shared memory object store](plasma-store) and in process memory
store. Small request objects are directly sent between actors via network
call. Larger request objects (100KiB+) are written to the object store and the replica can read them via zero-copy read.


(serve-model-multiplexing)=

# Model Multiplexing

This section helps you understand how to write multiplexed deployment by using the `serve.multiplexed` and `serve.get_multiplexed_model_id` APIs.

This is an experimental feature and the API may change in the future. You are welcome to try it out and give us feedback!

## Why model multiplexing?

Model multiplexing is a technique used to efficiently serve multiple models with similar input types from a pool of replicas. Traffic is routed to the corresponding model based on the request header. To serve multiple models with a pool of replicas, 
model multiplexing optimizes cost and load balances the traffic. This is useful in cases where you might have many models with the same shape but different weights that are sparsely invoked. If any replica for the deployment has the model loaded, incoming traffic for that model (based on request header) will automatically be routed to that replica avoiding unnecessary load time.

## Writing a multiplexed deployment

To write a multiplexed deployment, use the `serve.multiplexed` and `serve.get_multiplexed_model_id` APIs.

Assuming you have multiple Torch models inside an aws s3 bucket with the following structure:
```
s3://my_bucket/1/model.pt
s3://my_bucket/2/model.pt
s3://my_bucket/3/model.pt
s3://my_bucket/4/model.pt
...
```

Define a multiplexed deployment:
```{literalinclude} doc_code/multiplexed.py
:language: python
:start-after: __serve_deployment_example_begin__
:end-before: __serve_deployment_example_end__
```

:::{note}
The `serve.multiplexed` API also has a `max_num_models_per_replica` parameter. Use it to configure how many models to load in a single replica. If the number of models is larger than `max_num_models_per_replica`, Serve uses the LRU policy to evict the least recently used model.
:::

:::{tip}
This code example uses the Pytorch Model object. You can also define your own model class and use it here. To release resources when the model is evicted, implement the `__del__` method. Ray Serve internally calls the `__del__` method to release resources when the model is evicted.
:::


`serve.get_multiplexed_model_id` is used to retrieve the model id from the request header, and the model_id is then passed into the `get_model` function. If the model id is not found in the replica, Serve will load the model from the s3 bucket and cache it in the replica. If the model id is found in the replica, Serve will return the cached model.

:::{note}
Internally, serve router will route the traffic to the corresponding replica based on the model id in the request header.
If all replicas holding the model are over-subscribed, ray serve sends the request to a new replica that doesn't have the model loaded. The replica will load the model from the s3 bucket and cache it.
:::

To send a request to a specific model, include the `serve_multiplexed_model_id` field in the request header, and set the value to the model ID to which you want to send the request.
```{literalinclude} doc_code/multiplexed.py
:language: python
:start-after: __serve_request_send_example_begin__
:end-before: __serve_request_send_example_end__
```
:::{note}
`serve_multiplexed_model_id` is required in the request header, and the value should be the model ID you want to send the request to.

If the `serve_multiplexed_model_id` is not found in the request header, Serve will treat it as a normal request and route it to a random replica.
:::

After you run the above code, you should see the following lines in the deployment logs:
```
INFO 2023-05-24 01:19:03,853 default_Model default_Model#EjYmnQ CUpzhwUUNw / default replica.py:442 - Started executing request CUpzhwUUNw
INFO 2023-05-24 01:19:03,854 default_Model default_Model#EjYmnQ CUpzhwUUNw / default multiplex.py:131 - Loading model '1'.
INFO 2023-05-24 01:19:04,859 default_Model default_Model#EjYmnQ CUpzhwUUNw / default replica.py:542 - __CALL__ OK 1005.8ms
```

If you continue to load more models and exceed the `max_num_models_per_replica`, the least recently used model will be evicted and you will see the following lines in the deployment logs::
```
INFO 2023-05-24 01:19:15,988 default_Model default_Model#rimNjA WzjTbJvbPN / default replica.py:442 - Started executing request WzjTbJvbPN
INFO 2023-05-24 01:19:15,988 default_Model default_Model#rimNjA WzjTbJvbPN / default multiplex.py:145 - Unloading model '3'.
INFO 2023-05-24 01:19:15,988 default_Model default_Model#rimNjA WzjTbJvbPN / default multiplex.py:131 - Loading model '4'.
INFO 2023-05-24 01:19:16,993 default_Model default_Model#rimNjA WzjTbJvbPN / default replica.py:542 - __CALL__ OK 1005.7ms
```

You can also send a request to a specific model by using handle {mod}`options <ray.serve.handle.DeploymentHandle>` API.
```{literalinclude} doc_code/multiplexed.py
:language: python
:start-after: __serve_handle_send_example_begin__
:end-before: __serve_handle_send_example_end__
```

When using model composition, you can send requests from an upstream deployment to a multiplexed deployment using the Serve DeploymentHandle. You need to set the `multiplexed_model_id` in the options. For example:
```{literalinclude} doc_code/multiplexed.py
:language: python
:start-after: __serve_model_composition_example_begin__
:end-before: __serve_model_composition_example_end__
```


(serve-model-composition)=

# Deploy Compositions of Models

With this guide, you can:

* Compose multiple {ref}`deployments <serve-key-concepts-deployment>` containing ML models or business logic into a single {ref}`application <serve-key-concepts-application>`
* Independently scale and configure each of your ML models and business logic steps

:::{note}
The deprecated `RayServeHandle` and `RayServeSyncHandle` APIs have been fully removed as of Ray 2.10.
:::

## Compose deployments using DeploymentHandles

When building an application, you can `.bind()` multiple deployments and pass them to each other's constructors.
At runtime, inside the deployment code Ray Serve substitutes the bound deployments with 
{ref}`DeploymentHandles <serve-key-concepts-deployment-handle>` that you can use to call methods of other deployments.
This capability lets you divide your application's steps, such as preprocessing, model inference, and post-processing, into independent deployments that you can independently scale and configure.

Use {mod}`handle.remote <ray.serve.handle.DeploymentHandle.remote>` to send requests to a deployment.
These requests can contain ordinary Python args and kwargs, which DeploymentHandles can pass  directly to the method.
The method call returns a {mod}`DeploymentResponse <ray.serve.handle.DeploymentResponse>` that represents a future to the output.
You can `await` the response to retrieve its result or pass it to another downstream {mod}`DeploymentHandle <ray.serve.handle.DeploymentHandle>` call.

(serve-model-composition-deployment-handles)=
## Basic DeploymentHandle example

This example has two deployments:

```{literalinclude} doc_code/model_composition/language_example.py
:start-after: __hello_start__
:end-before: __hello_end__
:language: python
:linenos: true
```

In line 42, the `LanguageClassifier` deployment takes in the `spanish_responder` and `french_responder` as constructor arguments. At runtime, Ray Serve converts these arguments into `DeploymentHandles`. `LanguageClassifier` can then call the `spanish_responder` and `french_responder`'s deployment methods using this handle.

For example, the `LanguageClassifier`'s `__call__` method uses the HTTP request's values to decide whether to respond in Spanish or French. It then forwards the request's name to the `spanish_responder` or the `french_responder` on lines 19 and 21 using the `DeploymentHandle`s. The format of the calls is as follows:

```python
response: DeploymentResponse = self.spanish_responder.say_hello.remote(name)
```

This call has a few parts:
* `self.spanish_responder` is the `SpanishResponder` handle taken in through the constructor.
* `say_hello` is the `SpanishResponder` method to invoke.
* `remote` indicates that this is a `DeploymentHandle` call to another deployment.
* `name` is the argument for `say_hello`. You can pass any number of arguments or keyword arguments here.

This call returns a `DeploymentResponse` object, which is a reference to the result, rather than the result itself.
This pattern allows the call to execute asynchronously.
To get the actual result, `await` the response.
`await` blocks until the asynchronous call executes and then returns the result.
In this example, line 25 calls `await response` and returns the resulting string.

(serve-model-composition-await-warning)=
:::{warning}
You can use the `response.result()` method to get the return value of remote `DeploymentHandle` calls.
However, avoid calling `.result()` from inside a deployment because it blocks the deployment from executing any other code until the remote method call finishes.
Using `await` lets the deployment process other requests while waiting for the remote method call to finish.
You should use `await` instead of `.result()` inside deployments.
:::

You can copy the preceding `hello.py` script and run it with `serve run`. Make sure to run the command from a directory containing `hello.py`, so it can locate the script:

```console
$ serve run hello:language_classifier
```

You can use this client script to interact with the example:

```{literalinclude} doc_code/model_composition/language_example.py
:start-after: __hello_client_start__
:end-before: __hello_client_end__
:language: python
```

While the `serve run` command is running, open a separate terminal window and run the script:

```console
$ python hello_client.py

Hola Dora
```

:::{note}
Composition lets you break apart your application and independently scale each part. For instance, suppose this `LanguageClassifier` application's requests were 75% Spanish and 25% French. You could scale your `SpanishResponder` to have 3 replicas and your `FrenchResponder` to have 1 replica, so you can meet your workload's demand. This flexibility also applies to reserving resources like CPUs and GPUs, as well as any other configurations you can set for each deployment.

With composition, you can avoid application-level bottlenecks when serving models and business logic steps that use different types and amounts of resources.
:::

## Chaining DeploymentHandle calls

Ray Serve can directly pass the `DeploymentResponse` object that a `DeploymentHandle` returns, to another `DeploymentHandle` call to chain together multiple stages of a pipeline.
You don't need to `await` the first response, Ray Serve
manages the `await` behavior under the hood. When the first call finishes, Ray Serve passes the output of the first call, instead of the `DeploymentResponse` object, directly to the second call.

For example, the code sample below defines three deployments in an application:

- An `Adder` deployment that increments a value by its configured increment.
- A `Multiplier` deployment that multiplies a value by its configured multiple.
- An `Ingress` deployment that chains calls to the adder and multiplier together and returns the final response.

Note how the response from the `Adder` handle passes directly to the `Multiplier` handle, but inside the multiplier, the input argument resolves to the output of the `Adder` call.

```{literalinclude} doc_code/model_composition/chaining_example.py
:start-after: __chaining_example_start__
:end-before: __chaining_example_end__
:language: python
```

## Streaming DeploymentHandle calls

You can also use `DeploymentHandles` to make streaming method calls that return multiple outputs.
To make a streaming call, the method must be a generator and you must set `handle.options(stream=True)`.
Then, the handle call returns a {mod}`DeploymentResponseGenerator <ray.serve.handle.DeploymentResponseGenerator>` instead of a unary `DeploymentResponse`.
You can use `DeploymentResponseGenerators` as a sync or async generator, like in an `async for` code block.
Similar to `DeploymentResponse.result()`, avoid using a `DeploymentResponseGenerator` as a sync generator within a deployment, as that blocks other requests from executing concurrently on that replica.
Note that you can't pass `DeploymentResponseGenerators` to other handle calls.

Example:

```{literalinclude} doc_code/model_composition/streaming_example.py
:start-after: __streaming_example_start__
:end-before: __streaming_example_end__
:language: python
```

## Advanced: Pass a DeploymentResponse "by reference"

By default, when you pass a `DeploymentResponse` to another `DeploymentHandle` call, Ray Serve passes the result of the `DeploymentResponse` directly to the downstream method once it's ready.
However, in some cases you might want to start executing the downstream call before the result is ready. For example, to do some preprocessing or fetch a file from remote storage.
To accomplish this behavior, pass the `DeploymentResponse` "by reference" by embedding it in another Python object, such as a list or dictionary.
When you pass responses by reference, Ray Serve replaces them with Ray `ObjectRef`s instead of the resulting value and they can start executing before the result is ready.

The example below has two deployments: a preprocessor and a downstream model that takes the output of the preprocessor.
The downstream model has two methods:

- `pass_by_value` takes the output of the preprocessor "by value," so it doesn't execute until the preprocessor finishes.
- `pass_by_reference` takes the output "by reference," so it gets an `ObjectRef` and executes eagerly.

```{literalinclude} doc_code/model_composition/response_by_reference_example.py
:start-after: __response_by_reference_example_start__
:end-before: __response_by_reference_example_end__
:language: python
```

## Advanced: Convert a DeploymentResponse to a Ray ObjectRef

Under the hood, each `DeploymentResponse` corresponds to a Ray `ObjectRef`, or an `ObjectRefGenerator` for streaming calls.
To compose `DeploymentHandle` calls with Ray Actors or Tasks, you may want to resolve the response to its `ObjectRef`.
For this purpose, you can use the {mod}`DeploymentResponse._to_object_ref <ray.serve.handle.DeploymentResponse>` and {mod}`DeploymentResponse._to_object_ref_sync <ray.serve.handle.DeploymentResponse>` developer APIs.

Example:

```{literalinclude} doc_code/model_composition/response_to_object_ref_example.py
:start-after: __response_to_object_ref_example_start__
:end-before: __response_to_object_ref_example_end__
:language: python
```


(serve-develop-and-deploy)=

# Develop and Deploy an ML Application

The flow for developing a Ray Serve application locally and deploying it in production covers the following steps:

* Converting a Machine Learning model into a Ray Serve application
* Testing the application locally
* Building Serve config files for production deployment
* Deploying applications using a config file

## Convert a model into a Ray Serve application

This example uses a text-translation model:

```{literalinclude} ../serve/doc_code/getting_started/models.py
:start-after: __start_translation_model__
:end-before: __end_translation_model__
:language: python
```

The Python file, called `model.py`, uses the `Translator` class to translate English text to French.

- The `self.model` variable inside the `Translator`'s `__init__` method
  stores a function that uses the [t5-small](https://huggingface.co/t5-small)
  model to translate text.
- When `self.model` is called on English text, it returns translated French text
  inside a dictionary formatted as `[{"translation_text": "..."}]`.
- The `Translator`'s `translate` method extracts the translated text by indexing into the dictionary.

Copy and paste the script and run it locally. It translates `"Hello world!"`
into `"Bonjour Monde!"`.

```console
$ python model.py

Bonjour Monde!
```

Converting this model into a Ray Serve application with FastAPI requires three changes:
1. Import Ray Serve and Fast API dependencies
2. Add decorators for Serve deployment with FastAPI: `@serve.deployment` and `@serve.ingress(app)`
3. `bind` the `Translator` deployment to the arguments that are passed into its constructor

For other HTTP options, see [Set Up FastAPI and HTTP](serve-set-up-fastapi-http). 

```{literalinclude} ../serve/doc_code/develop_and_deploy.py
:start-after: __deployment_start__
:end-before: __deployment_end__
:language: python
```

Note that the code configures parameters for the deployment, such as `num_replicas` and `ray_actor_options`. These parameters help configure the number of copies of the deployment and the resource requirements for each copy. In this case, we set up 2 replicas of the model that take 0.2 CPUs and 0 GPUs each. For a complete guide on the configurable parameters on a deployment, see [Configure a Serve deployment](serve-configure-deployment).

## Test a Ray Serve application locally

To test locally, run the script with the `serve run` CLI command. This command takes in an import path formatted as `module:application`. Run the command from a directory containing a local copy of the script saved as `model.py`, so it can import the application:

```console
$ serve run model:translator_app
```

This command runs the `translator_app` application and then blocks streaming logs to the console. You can kill it with `Ctrl-C`, which tears down the application.

Now test the model over HTTP. Reach it at the following default URL:

```
http://127.0.0.1:8000/
```

Send a POST request with JSON data containing the English text. This client script requests a translation for "Hello world!":

```{literalinclude} ../serve/doc_code/develop_and_deploy.py
:start-after: __client_function_start__
:end-before: __client_function_end__
:language: python
```

While a Ray Serve application is deployed, use the `serve status` CLI command to check the status of the application and deployment. For more details on the output format of `serve status`, see [Inspect Serve in production](serve-in-production-inspecting).

```console
$ serve status
proxies:
  a85af35da5fcea04e13375bdc7d2c83c7d3915e290f1b25643c55f3a: HEALTHY
applications:
  default:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1693428451.894696
    deployments:
      Translator:
        status: HEALTHY
        replica_states:
          RUNNING: 2
        message: ''
```

## Build Serve config files for production deployment

To deploy Serve applications in production, you need to generate a Serve config YAML file. A Serve config file is the single source of truth for the cluster, allowing you to specify system-level configuration and your applications in one place. It also allows you to declaratively update your applications. The `serve build` CLI command takes as input the import path and saves to an output file using the `-o` flag. You can specify all deployment parameters in the Serve config files.

```console
$ serve build model:translator_app -o config.yaml
```

The `serve build` command adds a default application name that can be modified. The resulting Serve config file is:

```
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

grpc_options:
  port: 9000
  grpc_servicer_functions: []

applications:

- name: app1
  route_prefix: /
  import_path: model:translator_app
  runtime_env: {}
  deployments:
  - name: Translator
    num_replicas: 2
    ray_actor_options:
      num_cpus: 0.2
      num_gpus: 0.0
```

You can also use the Serve config file with `serve run` for local testing. For example:

```console
$ serve run config.yaml
```

```console
$ serve status
proxies:
  1894261b372d34854163ac5ec88405328302eb4e46ac3a2bdcaf8d18: HEALTHY
applications:
  app1:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1693430474.873806
    deployments:
      Translator:
        status: HEALTHY
        replica_states:
          RUNNING: 2
        message: ''
```

For more details, see [Serve Config Files](serve-in-production-config-file).

## Deploy Ray Serve in production

Deploy the Ray Serve application in production on Kubernetes using the [KubeRay] operator. Copy the YAML file generated in the previous step directly into the Kubernetes configuration. KubeRay supports zero-downtime upgrades, status reporting, and fault tolerance for your production application. See [Deploying on Kubernetes](serve-in-production-kubernetes) for more information. For production usage, consider implementing the recommended practice of setting up [head node fault tolerance](serve-e2e-ft-guide-gcs).

## Monitor Ray Serve

Use the Ray Dashboard to get a high-level overview of your Ray Cluster and Ray Serve application's states. The Ray Dashboard is available both during local testing and on a remote cluster in production. Ray Serve provides some in-built metrics and logging as well as utilities for adding custom metrics and logs in your application. For production deployments, exporting logs and metrics to your observability platforms is recommended. See [Monitoring](serve-monitoring) for more details. 

[KubeRay]: kuberay-index


---
orphan: true
---

# Serve Llama2-7b/70b on a single or multiple Intel Gaudi Accelerator

[Intel Gaudi AI Processors (HPUs)](https://habana.ai) are AI hardware accelerators designed by Intel Habana Labs. See [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/index.html) and [Gaudi Developer Docs](https://developer.habana.ai/) for more details.

This tutorial has two examples:

1. Deployment of [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) using a single HPU:

    * Load a model onto an HPU.

    * Perform generation on an HPU.

    * Enable HPU Graph optimizations.

2. Deployment of [Llama2-70b](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) using multiple HPUs on a single node:

    * Initialize a distributed backend.

    * Load a sharded model onto DeepSpeed workers.

    * Stream responses from DeepSpeed workers.

This tutorial serves a large language model (LLM) on HPUs.



## Environment setup

Use a prebuilt container to run these examples. To run a container, you need Docker. See [Install Docker Engine](https://docs.docker.com/engine/install/) for installation instructions.

Next, follow [Run Using Containers](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html?highlight=installer#run-using-containers) to install the Gaudi drivers and container runtime. To verify your installation, start a shell and run `hl-smi`. It should print status information about the HPUs on the machine:

```text
+-----------------------------------------------------------------------------+
| HL-SMI Version:                              hl-1.14.0-fw-48.0.1.0          |
| Driver Version:                                     1.15.0-c43dc7b          |
|-------------------------------+----------------------+----------------------+
| AIP  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | AIP-Util  Compute M. |
|===============================+======================+======================|
|   0  HL-225              N/A  | 0000:09:00.0     N/A |                   0  |
| N/A   26C   N/A    87W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   1  HL-225              N/A  | 0000:08:00.0     N/A |                   0  |
| N/A   28C   N/A    99W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   2  HL-225              N/A  | 0000:0a:00.0     N/A |                   0  |
| N/A   24C   N/A    98W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   3  HL-225              N/A  | 0000:0c:00.0     N/A |                   0  |
| N/A   27C   N/A    87W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   4  HL-225              N/A  | 0000:0b:00.0     N/A |                   0  |
| N/A   25C   N/A   112W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   5  HL-225              N/A  | 0000:0d:00.0     N/A |                   0  |
| N/A   26C   N/A   111W / 600W |  26835MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   6  HL-225              N/A  | 0000:0f:00.0     N/A |                   0  |
| N/A   24C   N/A    93W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   7  HL-225              N/A  | 0000:0e:00.0     N/A |                   0  |
| N/A   25C   N/A    86W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
| Compute Processes:                                               AIP Memory |
|  AIP       PID   Type   Process name                             Usage      |
|=============================================================================|
|   0        N/A   N/A    N/A                                      N/A        |
|   1        N/A   N/A    N/A                                      N/A        |
|   2        N/A   N/A    N/A                                      N/A        |
|   3        N/A   N/A    N/A                                      N/A        |
|   4        N/A   N/A    N/A                                      N/A        |
|   5        N/A   N/A    N/A                                      N/A        |
|   6        N/A   N/A    N/A                                      N/A        |
|   7        N/A   N/A    N/A                                      N/A        |
+=============================================================================+
```

Next, start the Gaudi container:
```bash
docker pull vault.habana.ai/gaudi-docker/1.14.0/ubuntu22.04/habanalabs/pytorch-installer-2.1.1:latest
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.14.0/ubuntu22.04/habanalabs/pytorch-installer-2.1.1:latest
```

To follow the examples in this tutorial, mount the directory containing the examples and models into the container. Inside the container, run:
```bash
pip install ray[tune,serve]
pip install git+https://github.com/huggingface/optimum-habana.git
# Replace 1.14.0 with the driver version of the container.
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.14.0
# Only needed by the DeepSpeed example.
export RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES=1
```

Start Ray in the container with `ray start --head`. You are now ready to run the examples.

## Running a model on a single HPU

This example shows how to deploy a Llama2-7b model on an HPU for inference. 

First, define a deployment that serves a Llama2-7b model using an HPU. Note that we enable [HPU graph optimizations](https://docs.habana.ai/en/latest/Gaudi_Overview/SynapseAI_Software_Suite.html?highlight=graph#graph-compiler-and-runtime) for better performance.

```{literalinclude} ../doc_code/intel_gaudi_inference_serve.py
:language: python
:start-after: __model_def_start__
:end-before: __model_def_end__
```

Copy the code above and save it as `intel_gaudi_inference_serve.py`. Start the deployment like this:

```bash
serve run intel_gaudi_inference_serve:entrypoint
```

The terminal should print logs as the deployment starts up:

```text
2024-02-01 05:38:34,021 INFO scripts.py:438 -- Running import path: 'ray_serve_7b:entrypoint'.
2024-02-01 05:38:36,112 INFO worker.py:1540 -- Connecting to existing Ray cluster at address: 10.111.128.177:6379...
2024-02-01 05:38:36,124 INFO worker.py:1715 -- Connected to Ray cluster. View the dashboard at 127.0.0.1:8265 
(ProxyActor pid=17179) INFO 2024-02-01 05:38:39,573 proxy 10.111.128.177 proxy.py:1141 - Proxy actor b0c697edb66f42a46f802f4603000000 starting on node 7776cd4634f69216c8354355018195b290314ad24fd9565404a2ed12.
(ProxyActor pid=17179) INFO 2024-02-01 05:38:39,580 proxy 10.111.128.177 proxy.py:1346 - Starting HTTP server on node: 7776cd4634f69216c8354355018195b290314ad24fd9565404a2ed12 listening on port 8000
(ProxyActor pid=17179) INFO:     Started server process [17179]
(ServeController pid=17084) INFO 2024-02-01 05:38:39,677 controller 17084 deployment_state.py:1545 - Deploying new version of deployment LlamaModel in application 'default'. Setting initial target number of replicas to 1.
(ServeController pid=17084) INFO 2024-02-01 05:38:39,780 controller 17084 deployment_state.py:1829 - Adding 1 replica to deployment LlamaModel in application 'default'.
(ServeReplica:default:LlamaModel pid=17272) [WARNING|utils.py:198] 2024-02-01 05:38:48,700 >> optimum-habana v1.11.0.dev0 has been validated for SynapseAI v1.14.0 but the driver version is v1.15.0, this could lead to undefined behavior!
(ServeReplica:default:LlamaModel pid=17272) /usr/local/lib/python3.10/dist-packages/transformers/models/auto/tokenization_auto.py:655: FutureWarning: The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.
(ServeReplica:default:LlamaModel pid=17272)   warnings.warn(
(ServeReplica:default:LlamaModel pid=17272) /usr/local/lib/python3.10/dist-packages/transformers/models/auto/configuration_auto.py:1020: FutureWarning: The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.
(ServeReplica:default:LlamaModel pid=17272)   warnings.warn(
(ServeReplica:default:LlamaModel pid=17272) /usr/local/lib/python3.10/dist-packages/transformers/models/auto/auto_factory.py:472: FutureWarning: The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.
(ServeReplica:default:LlamaModel pid=17272)   warnings.warn(
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
Loading checkpoint shards:  50%|█████     | 1/2 [00:17<00:17, 17.90s/it]
(ServeController pid=17084) WARNING 2024-02-01 05:39:09,835 controller 17084 deployment_state.py:2171 - Deployment 'LlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Loading checkpoint shards: 100%|██████████| 2/2 [00:24<00:00, 12.36s/it]
(ServeReplica:default:LlamaModel pid=17272) /usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:362: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
(ServeReplica:default:LlamaModel pid=17272)   warnings.warn(
(ServeReplica:default:LlamaModel pid=17272) /usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:367: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
(ServeReplica:default:LlamaModel pid=17272)   warnings.warn(
(ServeReplica:default:LlamaModel pid=17272) ============================= HABANA PT BRIDGE CONFIGURATION =========================== 
(ServeReplica:default:LlamaModel pid=17272)  PT_HPU_LAZY_MODE = 1
(ServeReplica:default:LlamaModel pid=17272)  PT_RECIPE_CACHE_PATH = 
(ServeReplica:default:LlamaModel pid=17272)  PT_CACHE_FOLDER_DELETE = 0
(ServeReplica:default:LlamaModel pid=17272)  PT_HPU_RECIPE_CACHE_CONFIG = 
(ServeReplica:default:LlamaModel pid=17272)  PT_HPU_MAX_COMPOUND_OP_SIZE = 9223372036854775807
(ServeReplica:default:LlamaModel pid=17272)  PT_HPU_LAZY_ACC_PAR_MODE = 1
(ServeReplica:default:LlamaModel pid=17272)  PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0
(ServeReplica:default:LlamaModel pid=17272) ---------------------------: System Configuration :---------------------------
(ServeReplica:default:LlamaModel pid=17272) Num CPU Cores : 156
(ServeReplica:default:LlamaModel pid=17272) CPU RAM       : 495094196 KB
(ServeReplica:default:LlamaModel pid=17272) ------------------------------------------------------------------------------
2024-02-01 05:39:25,873 SUCC scripts.py:483 -- Deployed Serve app successfully.
```

In another shell, use the following code to send requests to the deployment to perform generation tasks.

```{literalinclude} ../doc_code/intel_gaudi_inference_client.py
:language: python
:start-after: __main_code_start__
:end-before: __main_code_end__
```

Here is an example output:
```text
Once upon a time, in a far-off land, there was a magical kingdom called "Happily Ever Laughter." It was a place where laughter was the key to unlocking all the joys of life, and where everyone lived in perfect harmony.
In this kingdom, there was a beautiful princess named Lily. She was kind, gentle, and had a heart full of laughter. Every day, she would wake up with a smile on her face, ready to face whatever adventures the day might bring.
One day, a wicked sorcerer cast a spell on the kingdom, causing all
in a far-off land, there was a magical kingdom called "Happily Ever Laughter." It was a place where laughter was the key to unlocking all the joys of life, and where everyone lived in perfect harmony.
In this kingdom, there was a beautiful princess named Lily. She was kind, gentle, and had a heart full of laughter. Every day, she would wake up with a smile on her face, ready to face whatever adventures the day might bring.
One day, a wicked sorcerer cast a spell on the kingdom, causing all
```

## Running a sharded model on multiple HPUs

This example deploys a Llama2-70b model using 8 HPUs orchestrated by DeepSpeed. 

The example requires caching the Llama2-70b model. Run the following Python code in the Gaudi container to cache the model. 

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "meta-llama/Llama-2-70b-chat-hf",
    # Replace the path if necessary.
    cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
    # Specify your Hugging Face token.
    token=""
)
```

In this example, the deployment replica sends prompts to the DeepSpeed workers, which are running in Ray actors:

```{literalinclude} ../doc_code/intel_gaudi_inference_serve_deepspeed.py
:language: python
:start-after: __worker_def_start__
:end-before: __worker_def_end__
```

Next, define a deployment:

```{literalinclude} ../doc_code/intel_gaudi_inference_serve_deepspeed.py
:language: python
:start-after: __deploy_def_start__
:end-before: __deploy_def_end__
```

Copy both blocks of the preceding code and save them into `intel_gaudi_inference_serve_deepspeed.py`. Run this example using `serve run intel_gaudi_inference_serve_deepspeed:entrypoint`.

The terminal should print logs as the deployment starts up:
```text
2024-02-01 06:08:51,170 INFO scripts.py:438 -- Running import path: 'deepspeed_demo:entrypoint'.
2024-02-01 06:08:54,143 INFO worker.py:1540 -- Connecting to existing Ray cluster at address: 10.111.128.177:6379...
2024-02-01 06:08:54,154 INFO worker.py:1715 -- Connected to Ray cluster. View the dashboard at 127.0.0.1:8265 
(ServeController pid=44317) INFO 2024-02-01 06:08:54,348 controller 44317 deployment_state.py:1545 - Deploying new version of deployment DeepSpeedLlamaModel in application 'default'. Setting initial target number of replicas to 1.
(ServeController pid=44317) INFO 2024-02-01 06:08:54,457 controller 44317 deployment_state.py:1708 - Stopping 1 replicas of deployment 'DeepSpeedLlamaModel' in application 'default' with outdated versions.
(ServeController pid=44317) INFO 2024-02-01 06:08:57,326 controller 44317 deployment_state.py:2187 - Replica default#DeepSpeedLlamaModel#ToJmHV is stopped.
(ServeController pid=44317) INFO 2024-02-01 06:08:57,327 controller 44317 deployment_state.py:1829 - Adding 1 replica to deployment DeepSpeedLlamaModel in application 'default'.
(DeepSpeedInferenceWorker pid=48021) [WARNING|utils.py:198] 2024-02-01 06:09:12,355 >> optimum-habana v1.11.0.dev0 has been validated for SynapseAI v1.14.0 but the driver version is v1.15.0, this could lead to undefined behavior!
(DeepSpeedInferenceWorker pid=48016) /usr/local/lib/python3.10/dist-packages/habana_frameworks/torch/hpu/__init__.py:158: UserWarning: torch.hpu.setDeterministic is deprecated and will be removed in next release. Please use torch.use_deterministic_algorithms instead.
(DeepSpeedInferenceWorker pid=48016)   warnings.warn(
(DeepSpeedInferenceWorker pid=48019) [2024-02-01 06:09:14,005] [INFO] [real_accelerator.py:178:get_accelerator] Setting ds_accelerator to hpu (auto detect)
(DeepSpeedInferenceWorker pid=48019) [2024-02-01 06:09:16,908] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.12.4+hpu.synapse.v1.14.0, git-hash=fad45b2, git-branch=1.14.0
(DeepSpeedInferenceWorker pid=48019) [2024-02-01 06:09:16,910] [INFO] [logging.py:96:log_dist] [Rank -1] quantize_bits = 8 mlp_extra_grouping = False, quantize_groups = 1
Loading 15 checkpoint shards:   0%|          | 0/15 [00:00<?, ?it/s]
(DeepSpeedInferenceWorker pid=48019) [2024-02-01 06:09:16,955] [WARNING] [comm.py:163:init_deepspeed_backend] HCCL backend in DeepSpeed not yet implemented
(DeepSpeedInferenceWorker pid=48019) [2024-02-01 06:09:16,955] [INFO] [comm.py:637:init_distributed] cdb=None
(DeepSpeedInferenceWorker pid=48018) [WARNING|utils.py:198] 2024-02-01 06:09:13,528 >> optimum-habana v1.11.0.dev0 has been validated for SynapseAI v1.14.0 but the driver version is v1.15.0, this could lead to undefined behavior! [repeated 7x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/ray-logging.html#log-deduplication for more options.)
(ServeController pid=44317) WARNING 2024-02-01 06:09:27,403 controller 44317 deployment_state.py:2171 - Deployment 'DeepSpeedLlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
(DeepSpeedInferenceWorker pid=48018) /usr/local/lib/python3.10/dist-packages/habana_frameworks/torch/hpu/__init__.py:158: UserWarning: torch.hpu.setDeterministic is deprecated and will be removed in next release. Please use torch.use_deterministic_algorithms instead. [repeated 7x across cluster]
(DeepSpeedInferenceWorker pid=48018)   warnings.warn( [repeated 7x across cluster]
Loading 15 checkpoint shards:   0%|          | 0/15 [00:00<?, ?it/s] [repeated 7x across cluster]
(ServeController pid=44317) WARNING 2024-02-01 06:09:57,475 controller 44317 deployment_state.py:2171 - Deployment 'DeepSpeedLlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Loading 15 checkpoint shards:   7%|▋         | 1/15 [00:52<12:15, 52.53s/it]
(DeepSpeedInferenceWorker pid=48014) ============================= HABANA PT BRIDGE CONFIGURATION =========================== 
(DeepSpeedInferenceWorker pid=48014)  PT_HPU_LAZY_MODE = 1
(DeepSpeedInferenceWorker pid=48014)  PT_RECIPE_CACHE_PATH = 
(DeepSpeedInferenceWorker pid=48014)  PT_CACHE_FOLDER_DELETE = 0
(DeepSpeedInferenceWorker pid=48014)  PT_HPU_RECIPE_CACHE_CONFIG = 
(DeepSpeedInferenceWorker pid=48014)  PT_HPU_MAX_COMPOUND_OP_SIZE = 9223372036854775807
(DeepSpeedInferenceWorker pid=48014)  PT_HPU_LAZY_ACC_PAR_MODE = 0
(DeepSpeedInferenceWorker pid=48014)  PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0
(DeepSpeedInferenceWorker pid=48014) ---------------------------: System Configuration :---------------------------
(DeepSpeedInferenceWorker pid=48014) Num CPU Cores : 156
(DeepSpeedInferenceWorker pid=48014) CPU RAM       : 495094196 KB
(DeepSpeedInferenceWorker pid=48014) ------------------------------------------------------------------------------
Loading 15 checkpoint shards:   7%|▋         | 1/15 [00:57<13:28, 57.75s/it] [repeated 2x across cluster]
(ServeController pid=44317) WARNING 2024-02-01 06:10:27,504 controller 44317 deployment_state.py:2171 - Deployment 'DeepSpeedLlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Loading 15 checkpoint shards:   7%|▋         | 1/15 [00:58<13:42, 58.75s/it] [repeated 5x across cluster]
Loading 15 checkpoint shards:  13%|█▎        | 2/15 [01:15<07:21, 33.98s/it]
Loading 15 checkpoint shards:  13%|█▎        | 2/15 [01:16<07:31, 34.70s/it]
Loading 15 checkpoint shards:  20%|██        | 3/15 [01:35<05:34, 27.88s/it] [repeated 7x across cluster]
(ServeController pid=44317) WARNING 2024-02-01 06:10:57,547 controller 44317 deployment_state.py:2171 - Deployment 'DeepSpeedLlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Loading 15 checkpoint shards:  27%|██▋       | 4/15 [01:53<04:24, 24.03s/it] [repeated 8x across cluster]
(ServeController pid=44317) WARNING 2024-02-01 06:11:27,625 controller 44317 deployment_state.py:2171 - Deployment 'DeepSpeedLlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Loading 15 checkpoint shards:  27%|██▋       | 4/15 [01:54<04:21, 23.79s/it] [repeated 7x across cluster]
Loading 15 checkpoint shards:  40%|████      | 6/15 [02:30<03:06, 20.76s/it] [repeated 9x across cluster]
(ServeController pid=44317) WARNING 2024-02-01 06:11:57,657 controller 44317 deployment_state.py:2171 - Deployment 'DeepSpeedLlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Loading 15 checkpoint shards:  40%|████      | 6/15 [02:29<03:05, 20.61s/it] [repeated 7x across cluster]
Loading 15 checkpoint shards:  47%|████▋     | 7/15 [02:47<02:39, 19.88s/it]
Loading 15 checkpoint shards:  47%|████▋     | 7/15 [02:48<02:39, 19.90s/it]
Loading 15 checkpoint shards:  53%|█████▎    | 8/15 [03:06<02:17, 19.60s/it] [repeated 7x across cluster]
(ServeController pid=44317) WARNING 2024-02-01 06:12:27,721 controller 44317 deployment_state.py:2171 - Deployment 'DeepSpeedLlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Loading 15 checkpoint shards:  60%|██████    | 9/15 [03:26<01:56, 19.46s/it] [repeated 8x across cluster]
(ServeController pid=44317) WARNING 2024-02-01 06:12:57,725 controller 44317 deployment_state.py:2171 - Deployment 'DeepSpeedLlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Loading 15 checkpoint shards:  67%|██████▋   | 10/15 [03:27<01:09, 13.80s/it] [repeated 15x across cluster]
Loading 15 checkpoint shards:  73%|███████▎  | 11/15 [03:46<01:00, 15.14s/it]
Loading 15 checkpoint shards:  73%|███████▎  | 11/15 [03:45<01:00, 15.15s/it]
Loading 15 checkpoint shards:  80%|████████  | 12/15 [04:05<00:49, 16.47s/it] [repeated 7x across cluster]
(ServeController pid=44317) WARNING 2024-02-01 06:13:27,770 controller 44317 deployment_state.py:2171 - Deployment 'DeepSpeedLlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Loading 15 checkpoint shards:  87%|████████▋ | 13/15 [04:24<00:34, 17.26s/it] [repeated 8x across cluster]
(ServeController pid=44317) WARNING 2024-02-01 06:13:57,873 controller 44317 deployment_state.py:2171 - Deployment 'DeepSpeedLlamaModel' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
Loading 15 checkpoint shards:  87%|████████▋ | 13/15 [04:25<00:34, 17.35s/it] [repeated 7x across cluster]
Loading 15 checkpoint shards:  93%|█████████▎| 14/15 [04:44<00:17, 17.55s/it]
Loading 15 checkpoint shards: 100%|██████████| 15/15 [05:02<00:00, 18.30s/it] [repeated 8x across cluster]
2024-02-01 06:14:24,054 SUCC scripts.py:483 -- Deployed Serve app successfully.
```

Use the same code snippet introduced in the single HPU example to send generation requests. Here's an example output:
```text
Once upon a time, there was a young woman named Sophia who lived in a small village nestled in the rolling hills of Tuscany. Sophia was a curious and adventurous soul, always eager to explore the world around her. One day, while wandering through the village, she stumbled upon a hidden path she had never seen before.
The path was overgrown with weeds and vines, and it looked as though it hadn't been traversed in years. But Sophia was intrigued, and she decided to follow it to see where it led. She pushed aside the branches and stepped onto the path
Once upon a time, there was a young woman named Sophia who lived in a small village nestled in the rolling hills of Tuscany. Sophia was a curious and adventurous soul, always eager to explore the world around her. One day, while wandering through the village, she stumbled upon a hidden path she had never seen before.
The path was overgrown with weeds and vines, and it looked as though it hadn't been traversed in years. But Sophia was intrigued, and she decided to follow it to see where it led. She pushed aside the branches and stepped onto the path
```

## Next Steps
See [llm-on-ray](https://github.com/intel/llm-on-ray) for more ways to customize and deploy LLMs at scale.


---
orphan: true
---
(serve-java-tutorial)=

# Serve a Java App

To use Java Ray Serve, you need the following dependency in your pom.xml.

```xml
<dependency>
  <groupId>io.ray</groupId>
  <artifactId>ray-serve</artifactId>
  <version>${ray.version}</version>
  <scope>provided</scope>
</dependency>
```

> NOTE: After installing Ray with Python, the local environment includes the Java jar of Ray Serve. The `provided` scope ensures that you can compile the Java code using Ray Serve without version conflicts when you deploy on the cluster.

## Example model

This example use case is a production workflow of a financial application. The application needs to compute the best strategy to interact with different banks for a single task.

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/Strategy.java
:end-before: docs-strategy-end
:language: java
:start-after: docs-strategy-start
```

This example uses the `Strategy` class to calculate the indicators of a number of banks.

* The `calc` method is the entry of the calculation. The input parameters are the time interval of calculation and the map of the banks and their indicators. The `calc` method contains a two-tier `for` loop, traversing each indicator list of each bank, and calling the `calcBankIndicators` method to calculate the indicators of the specified bank.

- There is another layer of `for` loop in the `calcBankIndicators` method, which traverses each indicator, and then calls the `calcIndicator` method to calculate the specific indicator of the bank.
- The `calcIndicator` method is a specific calculation logic based on the bank, the specified time interval and the indicator.

This code uses the `Strategy` class:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/StrategyCalc.java
:end-before: docs-strategy-calc-end
:language: java
:start-after: docs-strategy-calc-start
```

When the scale of banks and indicators expands, the three-tier `for` loop slows down the calculation. Even if you use the thread pool to calculate each indicator in parallel, you may encounter a single machine performance bottleneck. Moreover, you can't use this `Strategy`  object as a resident service.

## Converting to a Ray Serve Deployment

Through Ray Serve, you can deploy the core computing logic of `Strategy` as a scalable distributed computing service.

First, extract the indicator calculation of each institution into a separate `StrategyOnRayServe` class:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/StrategyOnRayServe.java
:end-before: docs-strategy-end
:language: java
:start-after: docs-strategy-start
```

Next, start the Ray Serve runtime and deploy `StrategyOnRayServe` as a deployment.

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/StrategyCalcOnRayServe.java
:end-before: docs-deploy-end
:language: java
:start-after: docs-deploy-start
```

The `Deployment.create` makes a Deployment object named `strategy`. After executing `Deployment.deploy`, the Ray Serve instance deploys this `strategy` deployment with four replicas, and you can access it for distributed parallel computing.

## Testing the Ray Serve Deployment

You can test the `strategy` deployment using RayServeHandle inside Ray:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/StrategyCalcOnRayServe.java
:end-before: docs-calc-end
:language: java
:start-after: docs-calc-start
```

This code executes the calculation of each bank's each indicator serially, and sends it to Ray for execution. You can make the calculation concurrent, which not only improves the calculation efficiency, but also solves the bottleneck of single machine.

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/StrategyCalcOnRayServe.java
:end-before: docs-parallel-calc-end
:language: java
:start-after: docs-parallel-calc-start
```

You can use `StrategyCalcOnRayServe` like the example in the `main` method:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/StrategyCalcOnRayServe.java
:end-before: docs-main-end
:language: java
:start-after: docs-main-start
```

## Calling Ray Serve Deployment with HTTP

Another way to test or call a deployment is through the HTTP request. However, two limitations exist for the Java deployments:

- Only the `call` method of the user class can process the HTTP requests.

- The `call` method can only have one input parameter, and the type of the input parameter and the returned value can only be `String`.

If you want to call the `strategy` deployment with HTTP, then you can rewrite the class like this code:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/HttpStrategyOnRayServe.java
:end-before: docs-strategy-end
:language: java
:start-after: docs-strategy-start
```

After deploying this deployment, you can access it with the `curl` command:

```shell
curl -d '{"time":1641038674, "bank":"test_bank", "indicator":"test_indicator"}' http://127.0.0.1:8000/strategy
```

You can also access it using HTTP Client in Java code:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/HttpStrategyCalcOnRayServe.java
:end-before: docs-http-end
:language: java
:start-after: docs-http-start
```

The example of strategy calculation using HTTP to access deployment is as follows:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/HttpStrategyCalcOnRayServe.java
:end-before: docs-calc-end
:language: java
:start-after: docs-calc-start
```

You can also rewrite this code to support concurrency:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/HttpStrategyCalcOnRayServe.java
:end-before: docs-parallel-calc-end
:language: java
:start-after: docs-parallel-calc-start
```

Finally, the complete usage of `HttpStrategyCalcOnRayServe` is like this code:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/HttpStrategyCalcOnRayServe.java
:end-before: docs-main-end
:language: java
:start-after: docs-main-start
```


---
orphan: true
---

(serve-ml-models-tutorial)=

# Serve ML Models (Tensorflow, PyTorch, Scikit-Learn, others)

This guide shows how to train models from various machine learning frameworks and deploy them to Ray Serve.


See the [Key Concepts](serve-key-concepts) to learn more general information about Ray Serve.

:::::{tab-set} 

::::{tab-item} Keras and TensorFlow

This example trains and deploys a simple TensorFlow neural net.
In particular, it shows:

- How to train a TensorFlow model and load the model from your file system in your Ray Serve deployment.
- How to parse the JSON request and make a prediction.

Ray Serve is framework-agnostic--you can use any version of TensorFlow.
This tutorial uses TensorFlow 2 and Keras. You also need `requests` to send HTTP requests to your model deployment. If you haven't already, install TensorFlow 2 and requests by running:

```console
$ pip install "tensorflow>=2.0" requests
```

Open a new Python file called `tutorial_tensorflow.py`. First, import Ray Serve and some other helpers.

```{literalinclude} ../doc_code/tutorial_tensorflow.py
:start-after: __doc_import_begin__
:end-before: __doc_import_end__
```

Next, train a simple MNIST model using Keras.

```{literalinclude} ../doc_code/tutorial_tensorflow.py
:start-after: __doc_train_model_begin__
:end-before: __doc_train_model_end__
```

Next, define a `TFMnistModel` class that accepts HTTP requests and runs the MNIST model that you trained. The `@serve.deployment` decorator makes it a deployment object that you can deploy onto Ray Serve. Note that Ray Serve exposes the deployment over an HTTP route. By default, when the deployment receives a request over HTTP, Ray Serve invokes the `__call__` method.

```{literalinclude} ../doc_code/tutorial_tensorflow.py
:start-after: __doc_define_servable_begin__
:end-before: __doc_define_servable_end__
```

:::{note}
When you deploy and instantiate the `TFMnistModel` class, Ray Serve loads the TensorFlow model from your file system so that it can be ready to run inference on the model and serve requests later.
:::

Now that you've defined the Serve deployment, prepare it so that you can deploy it.

```{literalinclude} ../doc_code/tutorial_tensorflow.py
:start-after: __doc_deploy_begin__
:end-before: __doc_deploy_end__
```

:::{note}
`TFMnistModel.bind(TRAINED_MODEL_PATH)` binds the argument `TRAINED_MODEL_PATH` to the deployment and returns a `DeploymentNode` object, a wrapping of the `TFMnistModel` deployment object, that you can then use to connect with other `DeploymentNodes` to form a more complex [deployment graph](serve-model-composition).
:::

Finally, deploy the model to Ray Serve through the terminal.

```console
$ serve run tutorial_tensorflow:mnist_model
```

Next, query the model. While Serve is running, open a separate terminal window, and run the following in an interactive Python shell or a separate Python script:

```python
import requests
import numpy as np

resp = requests.get(
    "http://localhost:8000/", json={"array": np.random.randn(28 * 28).tolist()}
)
print(resp.json())
```

You should get an output like the following, although the exact prediction may vary:

```bash
{
 "prediction": [[-1.504277229309082, ..., -6.793371200561523]],
 "file": "/tmp/mnist_model.h5"
}
```
::::

::::{tab-item} PyTorch

This example loads and deploys a PyTorch ResNet model.
In particular, it shows:

- How to load the model from PyTorch's pre-trained Model Zoo.
- How to parse the JSON request, transform the payload and make a prediction.

This tutorial requires PyTorch and Torchvision. Ray Serve is framework agnostic and works with any version of PyTorch. You also need `requests` to send HTTP requests to your model deployment. If you haven't already, install them by running:

```console
$ pip install torch torchvision requests
```

Open a new Python file called `tutorial_pytorch.py`. First, import Ray Serve and some other helpers.

```{literalinclude} ../doc_code/tutorial_pytorch.py
:start-after: __doc_import_begin__
:end-before: __doc_import_end__
```

Define a class `ImageModel` that parses the input data, transforms the images, and runs the ResNet18 model loaded from `torchvision`. The `@serve.deployment` decorator makes it a deployment object that you can deploy onto Ray Serve.  Note that Ray Serve exposes the deployment over an HTTP route. By default, when the deployment receives a request over HTTP, Ray Serve invokes the `__call__` method.

```{literalinclude} ../doc_code/tutorial_pytorch.py
:start-after: __doc_define_servable_begin__
:end-before: __doc_define_servable_end__
```

:::{note}
When you deploy and instantiate an `ImageModel` class, Ray Serve loads the ResNet18 model from `torchvision` so that it can be ready to run inference on the model and serve requests later.
:::

Now that you've defined the Serve deployment, prepare it so that you can deploy it.

```{literalinclude} ../doc_code/tutorial_pytorch.py
:start-after: __doc_deploy_begin__
:end-before: __doc_deploy_end__
```

:::{note}
`ImageModel.bind()` returns a `DeploymentNode` object, a wrapping of the `ImageModel` deployment object, that you can then use to connect with other `DeploymentNodes` to form a more complex [deployment graph](serve-model-composition).
:::

Finally, deploy the model to Ray Serve through the terminal.
```console
$ serve run tutorial_pytorch:image_model
```

Next, query the model. While Serve is running, open a separate terminal window, and run the following in an interactive Python shell or a separate Python script:

```python
import requests

ray_logo_bytes = requests.get(
    "https://raw.githubusercontent.com/ray-project/"
    "ray/master/doc/source/images/ray_header_logo.png"
).content

resp = requests.post("http://localhost:8000/", data=ray_logo_bytes)
print(resp.json())
```

You should get an output like the following, although the exact number may vary:

```bash
{'class_index': 919}
```
::::

::::{tab-item} Scikit-learn

This example trains and deploys a simple scikit-learn classifier.
In particular, it shows:

- How to load the scikit-learn model from file system in your Ray Serve definition.
- How to parse the JSON request and make a prediction.

Ray Serve is framework-agnostic. You can use any version of sklearn. You also need `requests` to send HTTP requests to your model deployment. If you haven't already, install scikit-learn and requests by running:

```console
$ pip install scikit-learn requests
```

Open a new Python file called `tutorial_sklearn.py`. Import Ray Serve and some other helpers.

```{literalinclude} ../doc_code/tutorial_sklearn.py
:start-after: __doc_import_begin__
:end-before: __doc_import_end__
```

**Train a Classifier**

Next, train a classifier with the [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).


First, instantiate a `GradientBoostingClassifier` loaded from scikit-learn.

```{literalinclude} ../doc_code/tutorial_sklearn.py
:start-after: __doc_instantiate_model_begin__
:end-before: __doc_instantiate_model_end__
```

Next, load the Iris dataset and split the data into training and validation sets.

```{literalinclude} ../doc_code/tutorial_sklearn.py
:start-after: __doc_data_begin__
:end-before: __doc_data_end__
```

Then, train the model and save it to a file.

```{literalinclude} ../doc_code/tutorial_sklearn.py
:start-after: __doc_train_model_begin__
:end-before: __doc_train_model_end__
```

**Deploy with Ray Serve**

Finally, you're ready to deploy the classifier using Ray Serve.

Define a `BoostingModel` class that runs inference on the `GradientBoosingClassifier` model you trained and returns the resulting label. It's decorated with `@serve.deployment` to make it a deployment object so you can deploy it onto Ray Serve. Note that Ray Serve exposes the deployment over an HTTP route. By default, when the deployment receives a request over HTTP, Ray Serve invokes the `__call__` method.

```{literalinclude} ../doc_code/tutorial_sklearn.py
:start-after: __doc_define_servable_begin__
:end-before: __doc_define_servable_end__
```

:::{note}
When you deploy and instantiate a `BoostingModel` class, Ray Serve loads the classifier model that you trained from the file system so that it can be ready to run inference on the model and serve requests later.
:::

After you've defined the Serve deployment, prepare it so that you can deploy it.

```{literalinclude} ../doc_code/tutorial_sklearn.py
:start-after: __doc_deploy_begin__
:end-before: __doc_deploy_end__
```

:::{note}
`BoostingModel.bind(MODEL_PATH, LABEL_PATH)` binds the arguments `MODEL_PATH` and `LABEL_PATH` to the deployment and returns a `DeploymentNode` object, a wrapping of the `BoostingModel` deployment object, that you can then use to connect with other `DeploymentNodes` to form a more complex [deployment graph](serve-model-composition).
:::

Finally, deploy the model to Ray Serve through the terminal.
```console
$ serve run tutorial_sklearn:boosting_model
```

Next, query the model. While Serve is running, open a separate terminal window, and run the following in an interactive Python shell or a separate Python script:

```python
import requests

sample_request_input = {
    "sepal length": 1.2,
    "sepal width": 1.0,
    "petal length": 1.1,
    "petal width": 0.9,
}
response = requests.get("http://localhost:8000/", json=sample_request_input)
print(response.text)
```

You should get an output like the following, although the exact prediction may vary:
```python
{"result": "versicolor"}
```

::::

:::::


---
orphan: true
---

(serve-stable-diffusion-tutorial)=

# Serve a Stable Diffusion Model
This example runs a Stable Diffusion application with Ray Serve.

To run this example, install the following:

```bash
pip install "ray[serve]" requests torch diffusers==0.12.1 transformers
```

This example uses the [stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) model and [FastAPI](https://fastapi.tiangolo.com/) to build the example. Save the following code to a file named stable_diffusion.py. 

The Serve code is as follows:
```{literalinclude} ../doc_code/stable_diffusion.py
:language: python
:start-after: __example_code_start__
:end-before: __example_code_end__
```

Use `serve run stable_diffusion:entrypoint` to start the Serve application.

:::{note}
The autoscaling config sets `min_replicas` to 0, which means the deployment starts with no `ObjectDetection` replicas. These replicas spawn only when a request arrives. When no requests arrive after a certain period of time, Serve downscales `ObjectDetection` back to 0 replica to save GPU resources.
:::

You should see these messages in the output:
```text
(ServeController pid=362, ip=10.0.44.233) INFO 2023-03-08 16:44:57,579 controller 362 http_state.py:129 - Starting HTTP proxy with name 'SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-7396d5a9efdb59ee01b7befba448433f6c6fc734cfa5421d415da1b3' on node '7396d5a9efdb59ee01b7befba448433f6c6fc734cfa5421d415da1b3' listening on '127.0.0.1:8000'
(ServeController pid=362, ip=10.0.44.233) INFO 2023-03-08 16:44:57,588 controller 362 http_state.py:129 - Starting HTTP proxy with name 'SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-a30ea53938547e0bf88ce8672e578f0067be26a7e26d23465c46300b' on node 'a30ea53938547e0bf88ce8672e578f0067be26a7e26d23465c46300b' listening on '127.0.0.1:8000'
(ProxyActor pid=439, ip=10.0.44.233) INFO:     Started server process [439]
(ProxyActor pid=5779) INFO:     Started server process [5779]
(ServeController pid=362, ip=10.0.44.233) INFO 2023-03-08 16:44:59,362 controller 362 deployment_state.py:1333 - Adding 1 replica to deployment 'APIIngress'.
2023-03-08 16:45:01,316 SUCC <string>:93 -- Deployed Serve app successfully.
```

Use the following code to send requests:
```python
import requests

prompt = "a cute cat is dancing on the grass."
input = "%20".join(prompt.split(" "))
resp = requests.get(f"http://127.0.0.1:8000/imagine?prompt={input}")
with open("output.png", 'wb') as f:
    f.write(resp.content)
```
The app saves the `output.png` file locally. The following is an example of an output image.
![image](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/stable_diffusion_output.png)


---
orphan: true
---
(serve-streaming-tutorial)=

# Serve a Chatbot with Request and Response Streaming

This example deploys a chatbot that streams output back to the
user. It shows:

* How to stream outputs from a Serve application
* How to use WebSockets in a Serve application
* How to combine batching requests with streaming outputs

This tutorial should help you with following use cases:

* You want to serve a large language model and stream results back token-by-token.
* You want to serve a chatbot that accepts a stream of inputs from the user.

This tutorial serves the [DialoGPT](https://huggingface.co/microsoft/DialoGPT-small) language model. Install the Hugging Face library to access it:

```
pip install transformers
```

## Create a streaming deployment

Open a new Python file called `textbot.py`. First, add the imports and the [Serve logger](serve-logging).

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __textbot_setup_start__
:end-before: __textbot_setup_end__
```

Create a [FastAPI deployment](serve-fastapi-http), and initialize the model and the tokenizer in the
constructor:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __textbot_constructor_start__
:end-before: __textbot_constructor_end__
```

Note that the constructor also caches an `asyncio` loop. This behavior is useful when you need to run a model and concurrently stream its tokens back to the user.

Add the following logic to handle requests sent to the `Textbot`:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __textbot_logic_start__
:end-before: __textbot_logic_end__
```

`Textbot` uses three methods to handle requests:

* `handle_request`: the entrypoint for HTTP requests. FastAPI automatically unpacks the `prompt` query parameter and passes it into `handle_request`. This method then creates a `TextIteratorStreamer`. Hugging Face provides this streamer as a convenient interface to access tokens generated by a language model. `handle_request` then kicks off the model in a background thread using `self.loop.run_in_executor`. This behavior lets the model generate tokens while `handle_request` concurrently calls `self.consume_streamer` to stream the tokens back to the user. `self.consume_streamer` is a generator that yields tokens one by one from the streamer. Lastly, `handle_request` passes the `self.consume_streamer` generator into a Starlette `StreamingResponse` and returns the response. Serve unpacks the Starlette `StreamingResponse` and yields the contents of the generator back to the user one by one.
* `generate_text`: the method that runs the model. This method runs in a background thread kicked off by `handle_request`. It pushes generated tokens into the streamer constructed by `handle_request`.
* `consume_streamer`: a generator method that consumes the streamer constructed by `handle_request`. This method keeps yielding tokens from the streamer until the model in `generate_text` closes the streamer. This method avoids blocking the event loop by calling `asyncio.sleep` with a brief timeout whenever the streamer is empty and waiting for a new token.

Bind the `Textbot` to a language model. For this tutorial, use the `"microsoft/DialoGPT-small"` model:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __textbot_bind_start__
:end-before: __textbot_bind_end__
```

Run the model with `serve run textbot:app`, and query it from another terminal window with this script:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __stream_client_start__
:end-before: __stream_client_end__
```

You should see the output printed token by token.

## Stream inputs and outputs using WebSockets

WebSockets let you stream input into the application and stream output back to the client. Use WebSockets to create a chatbot that stores a conversation with a user.

Create a Python file called `chatbot.py`. First add the imports:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __chatbot_setup_start__
:end-before: __chatbot_setup_end__
```

Create a FastAPI deployment, and initialize the model and the tokenizer in the
constructor:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __chatbot_constructor_start__
:end-before: __chatbot_constructor_end__
```

Add the following logic to handle requests sent to the `Chatbot`:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __chatbot_logic_start__
:end-before: __chatbot_logic_end__
```

The `generate_text` and `consume_streamer` methods are the same as they were for the `Textbot`. The `handle_request` method has been updated to handle WebSocket requests.

The `handle_request` method is decorated with a `fastapi_app.websocket` decorator, which lets it accept WebSocket requests. First it `awaits` to accept the client's WebSocket request. Then, until the client disconnects, it does the following:

* gets the prompt from the client with `ws.receive_text`
* starts a new `TextIteratorStreamer` to access generated tokens
* runs the model in a background thread on the conversation so far
* streams the model's output back using `ws.send_text`
* stores the prompt and the response in the `conversation` string

Each time `handle_request` gets a new prompt from a client, it runs the whole conversation–with the new prompt appended–through the model. When the model finishes generating tokens, `handle_request` sends the `"<<Response Finished>>"` string to inform the client that the model has generated all tokens. `handle_request` continues to run until the client explicitly disconnects. This disconnect raises a `WebSocketDisconnect` exception, which ends the call.

Read more about WebSockets in the [FastAPI documentation](https://fastapi.tiangolo.com/advanced/websockets/).

Bind the `Chatbot` to a language model. For this tutorial, use the `"microsoft/DialoGPT-small"` model:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __chatbot_bind_start__
:end-before: __chatbot_bind_end__
```

Run the model with `serve run chatbot:app`. Query it using the `websockets` package, using `pip install websockets`:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __ws_client_start__
:end-before: __ws_client_end__
```

You should see the outputs printed token by token.

## Batch requests and stream the output for each

Improve model utilization and request latency by batching requests together when running the model.

Create a Python file called `batchbot.py`. First add the imports:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __batchbot_setup_start__
:end-before: __batchbot_setup_end__
```

:::{warning}
Hugging Face's support for `Streamers` is still under development and may change in the future. `RawQueue` is compatible with the `Streamers` interface in Hugging Face 4.30.2. However, the `Streamers` interface may change, making the `RawQueue` incompatible with Hugging Face models in the future.
:::

Similar to `Textbot` and `Chatbot`, the `Batchbot` needs a streamer to stream outputs from batched requests, but Hugging Face `Streamers` don't support batched requests. Add this custom `RawStreamer` to process batches of tokens:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __raw_streamer_start__
:end-before: __raw_streamer_end__
```

Create a FastAPI deployment, and initialize the model and the tokenizer in the
constructor:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __batchbot_constructor_start__
:end-before: __batchbot_constructor_end__
```

Unlike `Textbot` and `Chatbot`, the `Batchbot` constructor also sets a `pad_token`. You need to set this token to batch prompts with different lengths.

Add the following logic to handle requests sent to the `Batchbot`:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __batchbot_logic_start__
:end-before: __batchbot_logic_end__
```

`Batchbot` uses four methods to handle requests:

* `handle_request`: the entrypoint method. This method simply takes in the request's prompt and calls the `run_model` method on it. `run_model` is a generator method that also handles batching the requests. `handle_request` passes `run_model` into a Starlette `StreamingResponse` and returns the response, so the bot can stream generated tokens back to the client.
* `run_model`: a generator method that performs batching. Since `run_model` is decorated with `@serve.batch`, it automatically takes in a batch of prompts. See the [batching guide](serve-batch-tutorial) for more info. `run_model` creates a `RawStreamer` to access the generated tokens. It calls `generate_text` in a background thread, and passes in the `prompts` and the `streamer`, similar to the `Textbot`. Then it iterates through the `consume_streamer` generator, repeatedly yielding a batch of tokens generated by the model.
* `generate_text`: the method that runs the model. It's mostly the same as `generate_text` in `Textbot`, with two differences. First, it takes in and processes a batch of prompts instead of a single prompt. Second, it sets `padding=True`, so prompts with different lengths can be batched together.
* `consume_streamer`: a generator method that consumes the streamer constructed by `handle_request`. It's mostly the same as `consume_streamer` in `Textbot`, with one difference. It uses the `tokenizer` to decode the generated tokens. Usually, the Hugging Face streamer handles the decoding. Because this implementation uses the custom `RawStreamer`, `consume_streamer` must handle the decoding.

:::{tip}
Some inputs within a batch may generate fewer outputs than others. When a particular input has nothing left to yield, pass a `StopIteration` object into the output iterable to terminate that input's request. See [Streaming batched requests](serve-streaming-batched-requests-guide) for more details.
:::

Bind the `Batchbot` to a language model. For this tutorial, use the `"microsoft/DialoGPT-small"` model:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __batchbot_bind_start__
:end-before: __batchbot_bind_end__
```

Run the model with `serve run batchbot:app`. Query it from two other terminal windows with this script:

```{literalinclude} ../doc_code/streaming_tutorial.py
:language: python
:start-after: __stream_client_start__
:end-before: __stream_client_end__
```

You should see the output printed token by token in both windows.


---
orphan: true
---

(serve-batch-tutorial)=

# Serve a Text Generator with Request Batching

This example deploys a simple text generator that takes in
a batch of queries and processes them at once. In particular, it shows:

- How to implement and deploy a Ray Serve deployment that accepts batches.
- How to configure the batch size.
- How to query the model in Python.

This tutorial is a guide for serving online queries when your model can take advantage of batching. For example, linear regressions and neural networks use CPU and GPU's vectorized instructions to perform computation in parallel. Performing inference with batching can increase the *throughput* of the model as well as *utilization* of the hardware.

For _offline_ batch inference with large datasets, see [batch inference with Ray Data](batch_inference_home).


## Define the Deployment
Open a new Python file called `tutorial_batch.py`. First, import Ray Serve and some other helpers.

```{literalinclude} ../doc_code/tutorial_batch.py
:end-before: __doc_import_end__
:start-after: __doc_import_begin__
```

You can use the `@serve.batch` decorator to annotate a function or a method.
This annotation automatically causes calls to the function to be batched together.
The function must handle a list of objects and is called with a single object.
This function must also be `async def` so that you can handle multiple queries concurrently:

```python
@serve.batch
async def my_batch_handler(self, requests: List):
    pass
```

The batch handler can then be called from another `async def` method in your deployment.
These calls together are batched and executed together, but return an individual result as if
they were a normal function call:

```python
class BatchingDeployment:
    @serve.batch
    async def my_batch_handler(self, requests: List):
        results = []
        for request in requests:
            results.append(request.json())
        return results

    async def __call__(self, request):
        return await self.my_batch_handler(request)
```

:::{note}
By default, Ray Serve performs *opportunistic batching*. This means that as
soon as the batch handler is called, the method is executed without
waiting for a full batch. If there are more queries available after this call
finishes, the larger batch may be executed. You can tune this behavior using the
`batch_wait_timeout_s` option to `@serve.batch` (defaults to 0). Increasing this
timeout may improve throughput at the cost of latency under low load.
:::

Next, define a deployment that takes in a list of input strings and runs 
vectorized text generation on the inputs.

```{literalinclude} ../doc_code/tutorial_batch.py
:end-before: __doc_define_servable_end__
:start-after: __doc_define_servable_begin__
```

Next, prepare to deploy the deployment. Note that in the `@serve.batch` decorator, you
are specifying the maximum batch size with `max_batch_size=4`. This option limits
the maximum possible batch size that Ray Serve executes at once.

```{literalinclude} ../doc_code/tutorial_batch.py
:end-before: __doc_deploy_end__
:start-after: __doc_deploy_begin__
```

## Deploy the Deployment
Deploy the deployment by running the following through the terminal.
```console
$ serve run tutorial_batch:generator
```

Define a [Ray remote task](ray-remote-functions) to send queries in
parallel. While Serve is running, open a separate terminal window, and run the 
following in an interactive Python shell or a separate Python script:

```python
import ray
import requests
import numpy as np

@ray.remote
def send_query(text):
    resp = requests.get("http://localhost:8000/?text={}".format(text))
    return resp.text

# Use Ray to send all queries in parallel
texts = [
    'Once upon a time,',
    'Hi my name is Lewis and I like to',
    'My name is Mary, and my favorite',
    'My name is Clara and I am',
    'My name is Julien and I like to',
    'Today I accidentally',
    'My greatest wish is to',
    'In a galaxy far far away',
    'My best talent is',
]
results = ray.get([send_query.remote(text) for text in texts])
print("Result returned:", results)
```

You should get an output like the following. The first batch has a 
batch size of 1, and the subsequent queries have a batch size of 4. Even though the client script issues each 
query independently, Ray Serve evaluates them in batches.
```python
(pid=...) Our input array has length: 1
(pid=...) Our input array has length: 4
(pid=...) Our input array has length: 4
Result returned: [
    'Once upon a time, when I got to look at and see the work of my parents (I still can\'t stand them,) they said, "Boys, you\'re going to like it if you\'ll stay away from him or make him look',

    "Hi my name is Lewis and I like to look great. When I'm not playing against, it's when I play my best and always feel most comfortable. I get paid by the same people who make my games, who work hardest for me.", 

    "My name is Mary, and my favorite person in these two universes, the Green Lantern and the Red Lantern, are the same, except they're two of the Green Lanterns, but they also have their own different traits. Now their relationship is known", 

    'My name is Clara and I am married and live in Philadelphia. I am an English language teacher and translator. I am passionate about the issues that have so inspired me and my journey. My story begins with the discovery of my own child having been born', 

    'My name is Julien and I like to travel with my son on vacations... In fact I really prefer to spend more time with my son."\n\nIn 2011, the following year he was diagnosed with terminal Alzheimer\'s disease, and since then,', 

    "Today I accidentally got lost and went on another tour in August. My story was different, but it had so many emotions that it made me happy. I'm proud to still be able to go back to Oregon for work.\n\nFor the longest", 

    'My greatest wish is to return your loved ones to this earth where they can begin their own free and prosperous lives. This is true only on occasion as it is not intended or even encouraged to be so.\n\nThe Gospel of Luke 8:29', 

    'In a galaxy far far away, the most brilliant and powerful beings known would soon enter upon New York, setting out to restore order to the state. When the world turned against them, Darth Vader himself and Obi-Wan Kenobi, along with the Jedi', 

    'My best talent is that I can make a movie with somebody who really has a big and strong voice. I do believe that they would be great writers. I can tell you that to make sure."\n\n\nWith this in mind, "Ghostbusters'
]
```

## Deploy the Deployment using Python API
If you want to evaluate a whole batch in Python, Ray Serve allows you to send
queries with the Python API. A batch of queries can either come from the web server
or the Python API.

To query the deployment with the Python API, use `serve.run()`, which is part
of the Python API, instead of running `serve run` from the console. Add the following
to the Python script `tutorial_batch.py`:

```python
from ray.serve.handle import DeploymentHandle

handle: DeploymentHandle = serve.run(generator)
)
```

Generally, to enqueue a query, you can call `handle.method.remote(data)`. This call 
immediately returns a `DeploymentResponse`. You can call `.result()` to 
retrieve the result. Add the following to the same Python script.

```python
input_batch = [
    'Once upon a time,',
    'Hi my name is Lewis and I like to',
    'My name is Mary, and my favorite',
    'My name is Clara and I am',
    'My name is Julien and I like to',
    'Today I accidentally',
    'My greatest wish is to',
    'In a galaxy far far away',
    'My best talent is',
]
print("Input batch is", input_batch)

import ray
responses = [handle.handle_batch.remote(batch) for batch in input_batch]
results = [r.result() for r in responses]
print("Result batch is", results)
```

Finally, run the script.
```console
$ python tutorial_batch.py
```

You should get an output similar to the previous example.


---
orphan: true
---
(serve-text-classification-tutorial)=

# Serve a Text Classification Model
This example uses a DistilBERT model to build an IMDB review classification application with Ray Serve.

To run this example, install the following:

```bash
pip install "ray[serve]" requests torch transformers
```

This example uses the [distilbert-base-uncased](https://huggingface.co/docs/transformers/tasks/sequence_classification) model and [FastAPI](https://fastapi.tiangolo.com/). Save the following code to a file named distilbert_app.py:

Use the following Serve code:
```{literalinclude} ../doc_code/distilbert.py
:language: python
:start-after: __example_code_start__
:end-before: __example_code_end__
```

Use `serve run distilbert_app:entrypoint` to start the Serve application.

:::{note}
The autoscaling config sets `min_replicas` to 0, which means the deployment starts with no `ObjectDetection` replicas. These replicas spawn only when a request arrives. When no requests arrive after a certain period of time, Serve downscales `ObjectDetection` back to 0 replica to save GPU resources.
:::

You should see the following messages in the logs:
```text
(ServeController pid=362, ip=10.0.44.233) INFO 2023-03-08 16:44:57,579 controller 362 http_state.py:129 - Starting HTTP proxy with name 'SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-7396d5a9efdb59ee01b7befba448433f6c6fc734cfa5421d415da1b3' on node '7396d5a9efdb59ee01b7befba448433f6c6fc734cfa5421d415da1b3' listening on '127.0.0.1:8000'
(ServeController pid=362, ip=10.0.44.233) INFO 2023-03-08 16:44:57,588 controller 362 http_state.py:129 - Starting HTTP proxy with name 'SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-a30ea53938547e0bf88ce8672e578f0067be26a7e26d23465c46300b' on node 'a30ea53938547e0bf88ce8672e578f0067be26a7e26d23465c46300b' listening on '127.0.0.1:8000'
(ProxyActor pid=439, ip=10.0.44.233) INFO:     Started server process [439]
(ProxyActor pid=5779) INFO:     Started server process [5779]
(ServeController pid=362, ip=10.0.44.233) INFO 2023-03-08 16:44:59,362 controller 362 deployment_state.py:1333 - Adding 1 replica to deployment 'APIIngress'.
2023-03-08 16:45:01,316 SUCC <string>:93 -- Deployed Serve app successfully.
```

Use the following code to send requests:
```python
import requests

prompt = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
input = "%20".join(prompt.split(" "))
resp = requests.get(f"http://127.0.0.1:8000/classify?sentence={prompt}")
print(resp.status_code, resp.json())
```
The output of the client code is the response status code, the label, which is positive in this example, and the label's score.
```text
200 [{'label': 'LABEL_1', 'score': 0.9994940757751465}]
```


---
orphan: true
---
# Serving models with Triton Server in Ray Serve
This guide shows how to build an application with stable diffusion model using [NVIDIA Triton Server](https://github.com/triton-inference-server/server) in Ray Serve.

## Preparation

### Installation
It is recommended to use the `nvcr.io/nvidia/tritonserver:23.12-py3` image which already has the Triton Server python API library installed, and install the ray serve lib by `pip install "ray[serve]"` inside the image.

### Build and export a model
For this application, the encoder is exported to ONNX format and the stable diffusion model is exported to be TensorRT engine format which is being compatible with Triton Server.
Here is the example to export models to be in ONNX format.([source](https://github.com/triton-inference-server/tutorials/blob/main/Triton_Inference_Server_Python_API/scripts/stable_diffusion/export.py))

```python
import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

prompt = "Draw a dog"
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True
)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

vae.forward = vae.decode
torch.onnx.export(
    vae,
    (torch.randn(1, 4, 64, 64), False),
    "vae.onnx",
    input_names=["latent_sample", "return_dict"],
    output_names=["sample"],
    dynamic_axes={
        "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
    },
    do_constant_folding=True,
    opset_version=14,
)

text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

torch.onnx.export(
    text_encoder,
    (text_input.input_ids.to(torch.int32)),
    "encoder.onnx",
    input_names=["input_ids"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
    },
    opset_version=14,
    do_constant_folding=True,
)
```

From the script, the outputs are `vae.onnx` and `encoder.onnx`.

After the ONNX model exported, convert the ONNX model to the TensorRT engine serialized file. ([Details](https://github.com/NVIDIA/TensorRT/blob/release/9.2/samples/trtexec/README.md?plain=1#L22) about trtexec cli)
```bash
trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16
```

### Prepare the model repository
Triton Server requires a [model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md) to store the models, which is a local directory or remote blob store (e.g. AWS S3) containing the model configuration and the model files.
In our example, we will use a local directory as the model repository to save all the model files.

```bash
model_repo/
├── stable_diffusion
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
├── text_encoder
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
└── vae
    ├── 1
    │   └── model.plan
    └── config.pbtxt
```

The model repository contains three models: `stable_diffusion`, `text_encoder` and `vae`. Each model has a `config.pbtxt` file and a model file. The `config.pbtxt` file contains the model configuration, which is used to describe the model type and input/output formats.(you can learn more about model config file [here](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md)). To get config files for our example, you can download them from [here](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_6-building_complex_pipelines/model_repository). We use `1` as the version of each model. The model files are saved in the version directory.


## Start the Triton Server inside a Ray Serve application
In each serve replica, there is a single Triton Server instance running. The API takes the model repository path as the parameter, and the Triton Serve instance is started during the replica initialization. The models can be loaded during the inference requests, and the loaded models are cached in the Triton Server instance.

Here is the inference code example for serving a model with Triton Server.([source](https://github.com/triton-inference-server/tutorials/blob/main/Triton_Inference_Server_Python_API/examples/rayserve/tritonserver_deployment.py))

```python
import numpy
import requests
import tritonserver
from fastapi import FastAPI
from PIL import Image
from ray import serve


app = FastAPI()

@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):
        self._triton_server = tritonserver

        model_repository = ["/workspace/models"]

        self._triton_server = tritonserver.Server(
            model_repository=model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )
        self._triton_server.start(wait_until_ready=True)

    @app.get("/generate")
    def generate(self, prompt: str, filename: str = "generated_image.jpg") -> None:
        if not self._triton_server.model("stable_diffusion").ready():
            try:
                self._triton_server.load("text_encoder")
                self._triton_server.load("vae")
                self._stable_diffusion = self._triton_server.load("stable_diffusion")
                if not self._stable_diffusion.ready():
                    raise Exception("Model not ready")
            except Exception as error:
                print(f"Error can't load stable diffusion model, {error}")
                return

        for response in self._stable_diffusion.infer(inputs={"prompt": [[prompt]]}):
            generated_image = (
                numpy.from_dlpack(response.outputs["generated_image"])
                .squeeze()
                .astype(numpy.uint8)
            )

            image_ = Image.fromarray(generated_image)
            image_.save(filename)


if __name__ == "__main__":
    # Deploy the deployment.
    serve.run(TritonDeployment.bind())

    # Query the deployment.
    requests.get(
        "http://localhost:8000/generate",
        params={"prompt": "dogs in new york, realistic, 4k, photograph"},
    )
```

Save the above code to a file named e.g. `triton_serve.py`, then run `python triton_serve.py` to start the server and send classify requests. After you run the above code, you should see the image generated `generated_image.jpg`. Check it out!
![image](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/triton_server_stable_diffusion.jpg)


:::{note}
You can also use remote model repository, such as AWS S3, to store the model files. To use remote model repository, you need to set the `model_repository` variable to the remote model repository path.  For example `model_repository = s3://<bucket_name>/<model_repository_path>`.
:::

If you find any bugs or have any suggestions, please let us know by [filing an issue](https://github.com/ray-project/ray/issues) on GitHub.


---
orphan: true
---
(aws-neuron-core-inference-tutorial)=

# Serve an Inference Model on AWS NeuronCores Using FastAPI (Experimental)
This example compiles a BERT-based model and deploys the traced model on an AWS Inferentia (Inf2) or Tranium (Trn1)
instance using Ray Serve and FastAPI.


:::{note}
  Before starting this example:
  * Set up [PyTorch Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx)
  * Install AWS NeuronCore drivers and tools, and torch-neuronx based on the instance-type

:::

```bash
python -m pip install "ray[serve]" requests transformers
```

This example uses the [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) model and [FastAPI](https://fastapi.tiangolo.com/).

Use the following code to compile the model:
```{literalinclude} ../doc_code/aws_neuron_core_inference_serve.py
:language: python
:start-after: __compile_neuron_code_start__
:end-before: __compile_neuron_code_end__
```


For compiling the model, you should see the following log messages:
```text
Downloading (…)lve/main/config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.00k/1.00k [00:00<00:00, 242kB/s]
Downloading pytorch_model.bin: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 329M/329M [00:01<00:00, 217MB/s]
Downloading (…)okenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 294/294 [00:00<00:00, 305kB/s]
Downloading (…)olve/main/vocab.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 798k/798k [00:00<00:00, 22.0MB/s]
Downloading (…)olve/main/merges.txt: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 57.0MB/s]
Downloading (…)/main/tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.36M/1.36M [00:00<00:00, 6.16MB/s]
Downloading (…)cial_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 239/239 [00:00<00:00, 448kB/s]
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Saved Neuron-compiled model ./sentiment_neuron.pt
```

The traced model should be ready for deployment. Save the following code to a file named aws_neuron_core_inference_serve.py.

Use `serve run aws_neuron_core_inference_serve:entrypoint` to start the Serve application.
```{literalinclude} ../doc_code/aws_neuron_core_inference_serve.py
:language: python
:start-after: __neuron_serve_code_start__
:end-before: __neuron_serve_code_end__
```


You should see the following log messages when a deployment is successful:
```text
(ServeController pid=43105) INFO 2023-08-23 20:29:32,694 controller 43105 deployment_state.py:1372 - Deploying new version of deployment default_BertBaseModel.
(ServeController pid=43105) INFO 2023-08-23 20:29:32,695 controller 43105 deployment_state.py:1372 - Deploying new version of deployment default_APIIngress.
(ProxyActor pid=43147) INFO 2023-08-23 20:29:32,620 http_proxy 10.0.1.234 http_proxy.py:1328 - Proxy actor 8be14f6b6b10c0190cd0c39101000000 starting on node 46a7f740898fef723c3360ef598c1309701b07d11fb9dc45e236620a.
(ProxyActor pid=43147) INFO:     Started server process [43147]
(ServeController pid=43105) INFO 2023-08-23 20:29:32,799 controller 43105 deployment_state.py:1654 - Adding 1 replica to deployment default_BertBaseModel.
(ServeController pid=43105) INFO 2023-08-23 20:29:32,801 controller 43105 deployment_state.py:1654 - Adding 1 replica to deployment default_APIIngress.
2023-08-23 20:29:44,690 SUCC scripts.py:462 -- Deployed Serve app successfully.
```

Use the following code to send requests:
```python
import requests

response = requests.get(f"http://127.0.0.1:8000/infer?sentence=Ray is super cool")
print(response.status_code, response.json())
```
The response includes a status code and the classifier output:

```text
200 joy
```


---
orphan: true
---

(serve-object-detection-tutorial)=

# Serve an Object Detection Model
This example runs an object detection application with Ray Serve.

To run this example, install the following:

```bash
pip install "ray[serve]" requests torch
```

This example uses the [ultralytics/yolov5](https://github.com/ultralytics/yolov5) model and [FastAPI](https://fastapi.tiangolo.com/). Save the following code to a file named object_detection.py.

Use the following Serve code:
```{literalinclude} ../doc_code/object_detection.py
:language: python
:start-after: __example_code_start__
:end-before: __example_code_end__
```

Use `serve run object_detection:entrypoint` to start the Serve application.

:::{note}
The autoscaling config sets `min_replicas` to 0, which means the deployment starts with no `ObjectDetection` replicas. These replicas spawn only when a request arrives. After a period time when no requests arrive, Serve downscales `ObjectDetection` back to 0 replicas to save GPU resources.

:::

You should see the following log messages:
```text
(ServeReplica:ObjectDection pid=4747)   warnings.warn(
(ServeReplica:ObjectDection pid=4747) Downloading: "https://github.com/ultralytics/yolov5/zipball/master" to /home/ray/.cache/torch/hub/master.zip
(ServeReplica:ObjectDection pid=4747) YOLOv5 🚀 2023-3-8 Python-3.9.16 torch-1.13.0+cu116 CUDA:0 (Tesla T4, 15110MiB)
(ServeReplica:ObjectDection pid=4747) 
(ServeReplica:ObjectDection pid=4747) Fusing layers... 
(ServeReplica:ObjectDection pid=4747) YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients
(ServeReplica:ObjectDection pid=4747) Adding AutoShape... 
2023-03-08 21:10:21,685 SUCC <string>:93 -- Deployed Serve app successfully.
```

:::{tip}
While running, the Serve app may raise an error similar to the following:

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

This error usually occurs when running `opencv-python`, an image recognition library used in this example, on a headless environment, such as a container. This environment may lack dependencies that `opencv-python` needs. `opencv-python-headless` has fewer external dependencies and is suitable for headless environments.

In your Ray cluster, try running the following command:

```
pip uninstall opencv-python; pip install opencv-python-headless
```

:::

Use the following code to send requests:
```python
import requests

image_url = "https://ultralytics.com/images/zidane.jpg"
resp = requests.get(f"http://127.0.0.1:8000/detect?image_url={image_url}")

with open("output.jpeg", 'wb') as f:
    f.write(resp.content)
```
The app saves the output.png file locally. The following is an example of an output image.
![image](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/object_detection_output.jpeg)


---
orphan: true
---

# Serve an Inference with Stable Diffusion Model on AWS NeuronCores Using FastAPI
This example uses a precompiled Stable Diffusion XL model and deploys on an AWS Inferentia2 (Inf2)
instance using Ray Serve and FastAPI.


:::{note}
  Before starting this example: 
  * Set up [PyTorch Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx)
  * Install AWS NeuronCore drivers and tools, and torch-neuronx based on the instance-type

:::

```bash
pip install "optimum-neuron==0.0.13" "diffusers==0.21.4"
pip install "ray[serve]" requests transformers
```

This example uses the [Stable Diffusion-XL](https://huggingface.co/aws-neuron/stable-diffusion-xl-base-1-0-1024x1024) model and [FastAPI](https://fastapi.tiangolo.com/).
This model is compiled with AWS Neuron and is ready to run inference. However, you can choose a different Stable Diffusion model and compile it to be compatible for running inference on AWS Inferentia2
instances.

The model in this example is ready for deployment. Save the following code to a file named aws_neuron_core_inference_serve_stable_diffusion.py.

Use `serve run aws_neuron_core_inference_serve_stable_diffusion:entrypoint` to start the Serve application.
```{literalinclude} ../doc_code/aws_neuron_core_inference_serve_stable_diffusion.py
:language: python
:start-after: __neuron_serve_code_start__
:end-before: __neuron_serve_code_end__
```


You should see the following log messages when a deployment using RayServe is successful:
```text
2024-02-07 17:53:28,299	INFO worker.py:1715 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265 
(ProxyActor pid=25282) INFO 2024-02-07 17:53:31,751 proxy 172.31.10.188 proxy.py:1128 - Proxy actor fd464602af1e456162edf6f901000000 starting on node 5a8e0c24b22976f1f7672cc54f13ace25af3664a51429d8e332c0679.
(ProxyActor pid=25282) INFO 2024-02-07 17:53:31,755 proxy 172.31.10.188 proxy.py:1333 - Starting HTTP server on node: 5a8e0c24b22976f1f7672cc54f13ace25af3664a51429d8e332c0679 listening on port 8000
(ProxyActor pid=25282) INFO:     Started server process [25282]
(ServeController pid=25233) INFO 2024-02-07 17:53:31,921 controller 25233 deployment_state.py:1545 - Deploying new version of deployment StableDiffusionV2 in application 'default'. Setting initial target number of replicas to 1.
(ServeController pid=25233) INFO 2024-02-07 17:53:31,922 controller 25233 deployment_state.py:1545 - Deploying new version of deployment APIIngress in application 'default'. Setting initial target number of replicas to 1.
(ServeController pid=25233) INFO 2024-02-07 17:53:32,024 controller 25233 deployment_state.py:1829 - Adding 1 replica to deployment StableDiffusionV2 in application 'default'.
(ServeController pid=25233) INFO 2024-02-07 17:53:32,029 controller 25233 deployment_state.py:1829 - Adding 1 replica to deployment APIIngress in application 'default'.
Fetching 20 files: 100%|██████████| 20/20 [00:00<00:00, 195538.65it/s]
(ServeController pid=25233) WARNING 2024-02-07 17:54:02,114 controller 25233 deployment_state.py:2171 - Deployment 'StableDiffusionV2' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
(ServeController pid=25233) WARNING 2024-02-07 17:54:32,170 controller 25233 deployment_state.py:2171 - Deployment 'StableDiffusionV2' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
(ServeController pid=25233) WARNING 2024-02-07 17:55:02,344 controller 25233 deployment_state.py:2171 - Deployment 'StableDiffusionV2' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
(ServeController pid=25233) WARNING 2024-02-07 17:55:32,418 controller 25233 deployment_state.py:2171 - Deployment 'StableDiffusionV2' in application 'default' has 1 replicas that have taken more than 30s to initialize. This may be caused by a slow __init__ or reconfigure method.
2024-02-07 17:55:46,263	SUCC scripts.py:483 -- Deployed Serve app successfully.
```

Use the following code to send requests:
```python
import requests

prompt = "a zebra is dancing in the grass, river, sunlit"
input = "%20".join(prompt.split(" "))
resp = requests.get(f"http://127.0.0.1:8000/imagine?prompt={input}")
print("Write the response to `output.png`.")
with open("output.png", "wb") as f:
    f.write(resp.content)
```

You should see the following log messages when a request is sent to the endpoint:
```text
(ServeReplica:default:StableDiffusionV2 pid=25320) Prompt:  a zebra is dancing in the grass, river, sunlit
  0%|          | 0/50 [00:00<?, ?it/s]2 pid=25320) 
  2%|▏         | 1/50 [00:00<00:14,  3.43it/s]320) 
  4%|▍         | 2/50 [00:00<00:13,  3.62it/s]320) 
  6%|▌         | 3/50 [00:00<00:12,  3.73it/s]320) 
  8%|▊         | 4/50 [00:01<00:12,  3.78it/s]320) 
 10%|█         | 5/50 [00:01<00:11,  3.81it/s]320) 
 12%|█▏        | 6/50 [00:01<00:11,  3.82it/s]320) 
 14%|█▍        | 7/50 [00:01<00:11,  3.83it/s]320) 
 16%|█▌        | 8/50 [00:02<00:10,  3.84it/s]320) 
 18%|█▊        | 9/50 [00:02<00:10,  3.85it/s]320) 
 20%|██        | 10/50 [00:02<00:10,  3.85it/s]20) 
 22%|██▏       | 11/50 [00:02<00:10,  3.85it/s]20) 
 24%|██▍       | 12/50 [00:03<00:09,  3.86it/s]20) 
 26%|██▌       | 13/50 [00:03<00:09,  3.86it/s]20) 
 28%|██▊       | 14/50 [00:03<00:09,  3.85it/s]20) 
 30%|███       | 15/50 [00:03<00:09,  3.85it/s]20) 
 32%|███▏      | 16/50 [00:04<00:08,  3.85it/s]20) 
 34%|███▍      | 17/50 [00:04<00:08,  3.85it/s]20) 
 36%|███▌      | 18/50 [00:04<00:08,  3.85it/s]20) 
 38%|███▊      | 19/50 [00:04<00:08,  3.86it/s]20) 
 40%|████      | 20/50 [00:05<00:07,  3.85it/s]20) 
 42%|████▏     | 21/50 [00:05<00:07,  3.85it/s]20) 
 44%|████▍     | 22/50 [00:05<00:07,  3.85it/s]20) 
 46%|████▌     | 23/50 [00:06<00:07,  3.81it/s]20) 
 48%|████▊     | 24/50 [00:06<00:06,  3.81it/s]20) 
 50%|█████     | 25/50 [00:06<00:06,  3.82it/s]20) 
 52%|█████▏    | 26/50 [00:06<00:06,  3.83it/s]20) 
 54%|█████▍    | 27/50 [00:07<00:05,  3.84it/s]20) 
 56%|█████▌    | 28/50 [00:07<00:05,  3.84it/s]20) 
 58%|█████▊    | 29/50 [00:07<00:05,  3.84it/s]20) 
 60%|██████    | 30/50 [00:07<00:05,  3.84it/s]20) 
 62%|██████▏   | 31/50 [00:08<00:04,  3.84it/s]20) 
 64%|██████▍   | 32/50 [00:08<00:04,  3.84it/s]20) 
 66%|██████▌   | 33/50 [00:08<00:04,  3.85it/s]20) 
 68%|██████▊   | 34/50 [00:08<00:04,  3.85it/s]20) 
 70%|███████   | 35/50 [00:09<00:03,  3.84it/s]20) 
 72%|███████▏  | 36/50 [00:09<00:03,  3.84it/s]20) 
 74%|███████▍  | 37/50 [00:09<00:03,  3.84it/s]20) 
 76%|███████▌  | 38/50 [00:09<00:03,  3.84it/s]20) 
 78%|███████▊  | 39/50 [00:10<00:02,  3.84it/s]20) 
 80%|████████  | 40/50 [00:10<00:02,  3.84it/s]20) 
 82%|████████▏ | 41/50 [00:10<00:02,  3.84it/s]20) 
 84%|████████▍ | 42/50 [00:10<00:02,  3.84it/s]20) 
 86%|████████▌ | 43/50 [00:11<00:01,  3.84it/s]20) 
 88%|████████▊ | 44/50 [00:11<00:01,  3.84it/s]20) 
 90%|█████████ | 45/50 [00:11<00:01,  3.84it/s]20) 
 92%|█████████▏| 46/50 [00:11<00:01,  3.85it/s]20) 
 94%|█████████▍| 47/50 [00:12<00:00,  3.85it/s]20) 
 96%|█████████▌| 48/50 [00:12<00:00,  3.84it/s]20) 
 98%|█████████▊| 49/50 [00:12<00:00,  3.84it/s]20) 
100%|██████████| 50/50 [00:13<00:00,  3.83it/s]20) 
(ServeReplica:default:StableDiffusionV2 pid=25320) INFO 2024-02-07 17:58:36,604 default_StableDiffusionV2 OXPzZm 33133be7-246f-4492-9ab6-6a4c2666b306 /imagine replica.py:772 - GENERATE OK 14167.2ms
```


The app saves the `output.png` file locally. The following is an example of an output image.
![image](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/stable_diffusion_inferentia2_output.png)


---
orphan: true
---

# Scale a Gradio App with Ray Serve

This guide shows how to scale up your [Gradio](https://gradio.app/) application using Ray Serve. Keep the internal architecture of your Gradio app intact, with no code changes. Simply wrap the app within Ray Serve as a deployment and scale it to access more resources.
## Dependencies

To follow this tutorial, you need Ray Serve and Gradio. If you haven't already, install them by running:
```console
$ pip install "ray[serve]"
$ pip install gradio==3.50.2
```
This tutorial uses Gradio apps that run text summarization and generation models and use [Hugging Face's Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) to access these models. **Note that you can substitute this Gradio app for any Gradio app of your own.**

First, install the transformers module.
```console
$ pip install transformers
```

## Quickstart: Deploy your Gradio app with Ray Serve

This section shows an easy way to deploy your app onto Ray Serve. First, create a new Python file named `demo.py`. Second, import `GradioServer` from Ray Serve to deploy your Gradio app later, `gradio`, and `transformers.pipeline` to load text summarization models.
```{literalinclude} ../doc_code/gradio-integration.py
:start-after: __doc_import_begin__
:end-before: __doc_import_end__
```

Then, write a builder function that constructs the Gradio app `io`. This application takes in text and uses the [T5 Small](https://huggingface.co/t5-small) text summarization model loaded using [Hugging Face's Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) to summarize that text.
:::{note} 
Remember you can substitute this app with your own Gradio app if you want to try scaling up your own Gradio app.
:::
```{literalinclude} ../doc_code/gradio-integration.py
:start-after: __doc_gradio_app_begin__
:end-before: __doc_gradio_app_end__
```

### Deploying Gradio Server
In order to deploy your Gradio app onto Ray Serve, you need to wrap your Gradio app in a Serve [deployment](serve-key-concepts-deployment). `GradioServer` acts as that wrapper. It serves your Gradio app remotely on Ray Serve so that it can process and respond to HTTP requests. 

By wrapping your application in `GradioServer`, you can increase the number of CPUs and/or GPUs available to the application.
:::{note}
Ray Serve doesn't support routing requests to multiple replicas of `GradioServer`, so you should only have a single replica.
:::

:::{note} 
`GradioServer` is simply `GradioIngress` but wrapped in a Serve deployment. You can use `GradioServer` for the simple wrap-and-deploy use case, but in the next section, you can use `GradioIngress` to define your own Gradio Server for more customized use cases.
:::

:::{note} 
Ray can’t pickle Gradio. Instead, pass a builder function that constructs the Gradio interface.
:::

Using either the Gradio app `io`, which the builder function constructed, or your own Gradio app of type `Interface`, `Block`, `Parallel`, etc., wrap the app in your Gradio Server. Pass the builder function as input to your Gradio Server. Ray Serves uses the builder function to construct your Gradio app on the Ray cluster.

```{literalinclude} ../doc_code/gradio-integration.py
:start-after: __doc_app_begin__
:end-before: __doc_app_end__
```

Finally, deploy your Gradio Server. Run the following in your terminal:
```console
$ serve run demo:app
```

Access your Gradio app at `http://localhost:8000` The output should look like the following image:
![Gradio Result](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/gradio_result.png)

See the [Production Guide](serve-in-production) for more information on how to deploy your app in production.


## Parallelizing models with Ray Serve
You can run multiple models in parallel with Ray Serve by using [model composition](serve-model-composition) in Ray Serve.

### Original approach
Suppose you want to run the following program.

1. Take two text generation models, [`gpt2`](https://huggingface.co/gpt2) and [`distilgpt2`](https://huggingface.co/distilgpt2).
2. Run the two models on the same input text, so that the generated text has a minimum length of 20 and maximum length of 100.
3. Display the outputs of both models using Gradio.

This code is a typical implementation:

```{literalinclude} ../doc_code/gradio-original.py
:start-after: __doc_code_begin__
:end-before: __doc_code_end__
```
Launch the Gradio app with this command:
```
demo.launch()
```

### Parallelize using Ray Serve

With Ray Serve, you can parallelize the two text generation models by wrapping each model in a separate Ray Serve [deployment](serve-key-concepts-deployment). You can define deployments by decorating a Python class or function with `@serve.deployment`. The deployments usually wrap the models that you want to deploy on Ray Serve to handle incoming requests.

Follow these steps to achieve parallelism. First, import the dependencies. Note that you need to import `GradioIngress` instead of `GradioServer` like before because in this case, you're building a customized `MyGradioServer` that can run models in parallel.

```{literalinclude} ../doc_code/gradio-integration-parallel.py
:start-after: __doc_import_begin__
:end-before: __doc_import_end__
```

Then, wrap the `gpt2` and `distilgpt2` models in Serve deployments, named `TextGenerationModel`.
```{literalinclude} ../doc_code/gradio-integration-parallel.py
:start-after: __doc_models_begin__
:end-before: __doc_models_end__
```

Next, instead of simply wrapping the Gradio app in a `GradioServer` deployment, build your own `MyGradioServer` that reroutes the Gradio app so that it runs the `TextGenerationModel` deployments.

```{literalinclude} ../doc_code/gradio-integration-parallel.py
:start-after: __doc_gradio_server_begin__
:end-before: __doc_gradio_server_end__
```

Lastly, link everything together:
```{literalinclude} ../doc_code/gradio-integration-parallel.py
:start-after: __doc_app_begin__
:end-before: __doc_app_end__
```

:::{note} 
This step binds the two text generation models, which you wrapped in Serve deployments, to `MyGradioServer._d1` and `MyGradioServer._d2`, forming a [model composition](serve-model-composition). In the example, the Gradio Interface `io` calls `MyGradioServer.fanout()`, which sends requests to the two text generation models that you deployed on Ray Serve.
:::

Now, you can run your scalable app, to serve the two text generation models in parallel on Ray Serve.
Run your Gradio app with the following command:

```console
$ serve run demo:app
```

Access your Gradio app at `http://localhost:8000`, and you should see the following interactive interface:
![Gradio Result](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/gradio_result_parallel.png)

See the [Production Guide](serve-in-production) for more information on how to deploy your app in production.


(serve-in-production-deploying)=

# Deploy on VM

You can deploy your Serve application to production on a Ray cluster using the Ray Serve CLI.
`serve deploy` takes in a config file path and it deploys that file to a Ray cluster over HTTP.
This could either be a local, single-node cluster as in this example or a remote, multi-node cluster started with the [Ray Cluster Launcher](cloud-vm-index).

This section should help you:

- understand how to deploy a Ray Serve config file using the  CLI.
- understand how to update your application using the CLI.
- understand how to deploy to a remote cluster started with the [Ray Cluster Launcher](cloud-vm-index).

Start by deploying this [config](production-config-yaml) for the Text ML Application [example](serve-in-production-example):

```console
$ ls
text_ml.py
serve_config.yaml

$ ray start --head
...

$ serve deploy serve_config.yaml
2022-06-20 17:26:31,106	SUCC scripts.py:139 --
Sent deploy request successfully!
 * Use `serve status` to check deployments' statuses.
 * Use `serve config` to see the running app's config.
```

`ray start --head` starts a long-lived Ray cluster locally. `serve deploy serve_config.yaml` deploys the `serve_config.yaml` file to this local cluster. To stop Ray cluster, run the CLI command `ray stop`.

The message `Sent deploy request successfully!` means:
* The Ray cluster has received your config file successfully.
* It will start a new Serve application if one hasn't already started.
* The Serve application will deploy the deployments from your deployment graph, updated with the configurations from your config file.

It does **not** mean that your Serve application, including your deployments, has already started running successfully. This happens asynchronously as the Ray cluster attempts to update itself to match the settings from your config file. See [Inspect an application](serve-in-production-inspecting) for how to get the current status.

(serve-in-production-remote-cluster)=

## Using a remote cluster

By default, `serve deploy` deploys to a cluster running locally. However, you should also use `serve deploy` whenever you want to deploy your Serve application to a remote cluster. `serve deploy` takes in an optional `--address/-a` argument where you can specify your remote Ray cluster's dashboard address. This address should be of the form:

```
[RAY_CLUSTER_URI]:[DASHBOARD_PORT]
```

As an example, the address for the local cluster started by `ray start --head` is `http://127.0.0.1:8265`. We can explicitly deploy to this address using the command

```console
$ serve deploy config_file.yaml -a http://127.0.0.1:8265
```

The Ray Dashboard's default port is 8265. To set it to a different value, use the `--dashboard-port` argument when running `ray start`.

:::{note}
When running on a remote cluster, you need to ensure that the import path is accessible. See [Handle Dependencies](serve-handling-dependencies) for how to add a runtime environment.
:::

:::{tip}
By default, all the Serve CLI commands assume that you're working with a local cluster. All Serve CLI commands, except `serve start` and `serve run` use the Ray Dashboard address associated with a local cluster started by `ray start --head`. However, if the `RAY_DASHBOARD_ADDRESS` environment variable is set, these Serve CLI commands will default to that value instead.

Similarly, `serve start` and `serve run`, use the Ray head node address associated with a local cluster by default. If the `RAY_ADDRESS` environment variable is set, they will use that value instead.

You can check `RAY_DASHBOARD_ADDRESS`'s value by running:

```console
$ echo $RAY_DASHBOARD_ADDRESS
```

You can set this variable by running the CLI command:

```console
$ export RAY_DASHBOARD_ADDRESS=[YOUR VALUE]
```

You can unset this variable by running the CLI command:

```console
$ unset RAY_DASHBOARD_ADDRESS
```

Check for this variable in your environment to make sure you're using your desired Ray Dashboard address.
:::

To inspect the status of the Serve application in production, see [Inspect an application](serve-in-production-inspecting).

Make heavyweight code updates (like `runtime_env` changes) by starting a new Ray Cluster, updating your Serve config file, and deploying the file with `serve deploy` to the new cluster. Once the new deployment is finished, switch your traffic to the new cluster.


(serve-app-builder-guide)=
# Pass Arguments to Applications

This section describes how to pass arguments to your applications using an application builder function.

## Defining an application builder

When writing an application, there are often parameters that you want to be able to easily change in development or production.
For example, you might have a path to trained model weights and want to test out a newly trained model.
In Ray Serve, these parameters are typically passed to the constructor of your deployments using `.bind()`.
This pattern allows you to be configure deployments using ordinary Python code but it requires modifying the code anytime one of the parameters needs to change.

To pass arguments without changing the code, define an "application builder" function that takes an arguments dictionary (or [Pydantic object](typed-app-builders)) and returns the built application to be run.

```{literalinclude} ../doc_code/app_builder.py
:start-after: __begin_untyped_builder__
:end-before: __end_untyped_builder__
:language: python
```

You can use this application builder function as the import path in the `serve run` CLI command or the config file (as shown below).
To avoid writing code to handle type conversions and missing arguments, use a [Pydantic object](typed-app-builders) instead.

### Passing arguments via `serve run`

Pass arguments to the application builder from `serve run` using the following syntax:

```bash
$ serve run hello:app_builder key1=val1 key2=val2
```

The arguments are passed to the application builder as a dictionary, in this case `{"key1": "val1", "key2": "val2"}`.
For example, to pass a new message to the `HelloWorld` app defined above (with the code saved in `hello.py`):

```bash
% serve run hello:app_builder message="Hello from CLI"
2023-05-16 10:47:31,641 INFO scripts.py:404 -- Running import path: 'hello:app_builder'.
2023-05-16 10:47:33,344 INFO worker.py:1615 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
(ServeController pid=56826) INFO 2023-05-16 10:47:35,115 controller 56826 deployment_state.py:1244 - Deploying new version of deployment default_HelloWorld.
(ServeController pid=56826) INFO 2023-05-16 10:47:35,141 controller 56826 deployment_state.py:1483 - Adding 1 replica to deployment default_HelloWorld.
(ProxyActor pid=56828) INFO:     Started server process [56828]
(ServeReplica:default_HelloWorld pid=56830) Message: Hello from CLI
2023-05-16 10:47:36,131 SUCC scripts.py:424 -- Deployed Serve app successfully.
```

Notice that the "Hello from CLI" message is printed from within the deployment constructor.

### Passing arguments via config file

Pass arguments to the application builder in the config file's `args` field:

```yaml
applications:
  - name: MyApp
    import_path: hello:app_builder
    args:
      message: "Hello from config"
```

For example, to pass a new message to the `HelloWorld` app defined above (with the code saved in `hello.py` and the config saved in `config.yaml`):

```bash
% serve run config.yaml
2023-05-16 10:49:25,247 INFO scripts.py:351 -- Running config file: 'config.yaml'.
2023-05-16 10:49:26,949 INFO worker.py:1615 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
2023-05-16 10:49:28,678 SUCC scripts.py:419 -- Submitted deploy config successfully.
(ServeController pid=57109) INFO 2023-05-16 10:49:28,676 controller 57109 controller.py:559 - Building application 'MyApp'.
(ProxyActor pid=57111) INFO:     Started server process [57111]
(ServeController pid=57109) INFO 2023-05-16 10:49:28,940 controller 57109 application_state.py:202 - Built application 'MyApp' successfully.
(ServeController pid=57109) INFO 2023-05-16 10:49:28,942 controller 57109 deployment_state.py:1244 - Deploying new version of deployment MyApp_HelloWorld.
(ServeController pid=57109) INFO 2023-05-16 10:49:29,016 controller 57109 deployment_state.py:1483 - Adding 1 replica to deployment MyApp_HelloWorld.
(ServeReplica:MyApp_HelloWorld pid=57113) Message: Hello from config
```

Notice that the "Hello from config" message is printed from within the deployment constructor.

(typed-app-builders)=
### Typing arguments with Pydantic

To avoid writing logic to parse and validate the arguments by hand, define a [Pydantic model](https://pydantic-docs.helpmanual.io/usage/models/) as the single input parameter's type to your application builder function (the parameter must be type annotated).
Arguments are passed the same way, but the resulting dictionary is used to construct the Pydantic model using `model.parse_obj(args_dict)`.

```{literalinclude} ../doc_code/app_builder.py
:start-after: __begin_typed_builder__
:end-before: __end_typed_builder__
:language: python
```

```bash
% serve run hello:app_builder message="Hello from CLI"
2023-05-16 10:47:31,641 INFO scripts.py:404 -- Running import path: 'hello:app_builder'.
2023-05-16 10:47:33,344 INFO worker.py:1615 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
(ServeController pid=56826) INFO 2023-05-16 10:47:35,115 controller 56826 deployment_state.py:1244 - Deploying new version of deployment default_HelloWorld.
(ServeController pid=56826) INFO 2023-05-16 10:47:35,141 controller 56826 deployment_state.py:1483 - Adding 1 replica to deployment default_HelloWorld.
(ProxyActor pid=56828) INFO:     Started server process [56828]
(ServeReplica:default_HelloWorld pid=56830) Message: Hello from CLI
2023-05-16 10:47:36,131 SUCC scripts.py:424 -- Deployed Serve app successfully.
```

## Common patterns

### Multiple parametrized applications using the same builder

You can use application builders to run multiple applications with the same code but different parameters.
For example, multiple applications may share preprocessing and HTTP handling logic but use many different trained model weights.
The same application builder `import_path` can take different arguments to define multiple applications as follows:

```yaml
applications:
  - name: Model1
    import_path: my_module:my_model_code
    args:
      model_uri: s3://my_bucket/model_1
  - name: Model2
    import_path: my_module:my_model_code
    args:
      model_uri: s3://my_bucket/model_2
  - name: Model3
    import_path: my_module:my_model_code
    args:
      model_uri: s3://my_bucket/model_3
```

### Configuring multiple composed deployments

You can use the arguments passed to an application builder to configure multiple deployments in a single application.
For example a model composition application might take weights to two different models as follows:

```{literalinclude} ../doc_code/app_builder.py
:start-after: __begin_composed_builder__
:end-before: __end_composed_builder__
:language: python
```


(serve-performance-batching-requests)=
# Dynamic Request Batching

Serve offers a request batching feature that can improve your service throughput without sacrificing latency. This is possible because ML models can utilize efficient vectorized computation to process a batch of request at a time. Batching is also necessary when your model is expensive to use and you want to maximize the utilization of hardware.

Machine Learning (ML) frameworks such as Tensorflow, PyTorch, and Scikit-Learn support evaluating multiple samples at the same time.
Ray Serve allows you to take advantage of this feature via dynamic request batching.
When a request arrives, Serve puts the request in a queue. This queue buffers the requests to form a batch. The deployment picks up the batch and evaluates it. After the evaluation, the resulting batch will be split up, and each response is returned individually.

## Enable batching for your deployment
You can enable batching by using the {mod}`ray.serve.batch` decorator. Let's take a look at a simple example by modifying the `Model` class to accept a batch.
```{literalinclude} ../doc_code/batching_guide.py
---
start-after: __single_sample_begin__
end-before: __single_sample_end__
---
```

The batching decorators expect you to make the following changes in your method signature:
- The method is declared as an async method because the decorator batches in asyncio event loop.
- The method accepts a list of its original input types as input. For example, `arg1: int, arg2: str` should be changed to `arg1: List[int], arg2: List[str]`.
- The method returns a list. The length of the return list and the input list must be of equal lengths for the decorator to split the output evenly and return a corresponding response back to its respective request.

```{literalinclude} ../doc_code/batching_guide.py
---
start-after: __batch_begin__
end-before: __batch_end__
emphasize-lines: 6-9
---
```

You can supply two optional parameters to the decorators.
- `batch_wait_timeout_s` controls how long Serve should wait for a batch once the first request arrives.
- `max_batch_size` controls the size of the batch.
Once the first request arrives, the batching decorator will wait for a full batch (up to `max_batch_size`) until `batch_wait_timeout_s` is reached. If the timeout is reached, the batch will be sent to the model regardless the batch size.

:::{tip}
You can reconfigure your `batch_wait_timeout_s` and `max_batch_size` parameters using the `set_batch_wait_timeout_s` and `set_max_batch_size` methods:

```{literalinclude} ../doc_code/batching_guide.py
---
start-after: __batch_params_update_begin__
end-before: __batch_params_update_end__
---
```

Use these methods in the constructor or the `reconfigure` [method](serve-user-config) to control the `@serve.batch` parameters through your Serve configuration file.
:::

(serve-streaming-batched-requests-guide)=

## Streaming batched requests

Use an async generator to stream the outputs from your batched requests. Let's convert the `StreamingResponder` class to accept a batch.

```{literalinclude} ../doc_code/batching_guide.py
---
start-after: __single_stream_begin__
end-before: __single_stream_end__
---
```

Decorate async generator functions with the {mod}`ray.serve.batch` decorator. Similar to non-streaming methods, the function takes in a `List` of inputs and in each iteration it `yield`s an iterable of outputs with the same length as the input batch size.

```{literalinclude} ../doc_code/batching_guide.py
---
start-after: __batch_stream_begin__
end-before: __batch_stream_end__
---
```

Calling the `serve.batch`-decorated function returns an async generator that can be awaited to receive results.

Some inputs within a batch may generate fewer outputs than others. When a particular input has nothing left to yield, pass a `StopIteration` object into the output iterable. This terminates the generator that was returned when the `serve.batch` function was called with that input. When streaming generators returned by `serve.batch`-decorated functions over HTTP, this allows the end client's connection to terminate once its call is done, instead of waiting until the entire batch is done.

## Tips for fine-tuning batching parameters

`max_batch_size` ideally should be a power of 2 (2, 4, 8, 16, ...) because CPUs and GPUs are both optimized for data of these shapes. Large batch sizes incur a high memory cost as well as latency penalty for the first few requests.

`batch_wait_timeout_s` should be set considering the end to end latency SLO (Service Level Objective). For example, if your latency target is 150ms, and the model takes 100ms to evaluate the batch, the `batch_wait_timeout_s` should be set to a value much lower than 150ms - 100ms = 50ms.

When using batching in a Serve Deployment Graph, the relationship between an upstream node and a downstream node might affect the performance as well. Consider a chain of two models where first model sets `max_batch_size=8` and second model sets `max_batch_size=6`. In this scenario, when the first model finishes a full batch of 8, the second model will finish one batch of 6 and then to fill the next batch, which will initially only be partially filled with 8 - 6 = 2 requests, incurring latency costs. The batch size of downstream models should ideally be multiples or divisors of the upstream models to ensure the batches play well together.



(serve-inplace-updates)=

# Updating Applications In-Place

You can update your Serve applications once they're in production by updating the settings in your config file and redeploying it using the `serve deploy` command. In the redeployed config file, you can add new deployment settings or remove old deployment settings. This is because `serve deploy` is **idempotent**, meaning your Serve application's config always matches (or honors) the latest config you deployed successfully – regardless of what config files you deployed before that.

(serve-in-production-lightweight-update)=

## Lightweight Config Updates

Lightweight config updates modify running deployment replicas without tearing them down and restarting them, so there's less downtime as the deployments update. For each deployment, modifying the following values is considered a lightweight config update, and won't tear down the replicas for that deployment:
- `num_replicas`
- `autoscaling_config`
- `user_config`
- `max_ongoing_requests`
- `graceful_shutdown_timeout_s`
- `graceful_shutdown_wait_loop_s`
- `health_check_period_s`
- `health_check_timeout_s`

(serve-updating-user-config)=

## Updating the user config
This example uses the text summarization and translation application [from the production guide](production-config-yaml). Both of the individual deployments contain a `reconfigure()` method. This method allows you to issue lightweight updates to the deployments by updating the `user_config`.

First let's deploy the graph. Make sure to stop any previous Ray cluster using the CLI command `ray stop` for this example:

```console
$ ray start --head
$ serve deploy serve_config.yaml
```

Then send a request to the application:
```{literalinclude} ../doc_code/production_guide/text_ml.py
:language: python
:start-after: __start_client__
:end-before: __end_client__
```

Change the language that the text is translated into from French to German by changing the `language` attribute in the `Translator` user config:

```yaml
...

applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env:
    pip:
      - torch
      - transformers
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: german

...
```

Without stopping the Ray cluster, redeploy the app using `serve deploy`:

```console
$ serve deploy serve_config.yaml
...
```

We can inspect our deployments with `serve status`. Once the application's `status` returns to `RUNNING`, we can try our request one more time:

```console
$ serve status
proxies:
  cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec: HEALTHY
applications:
  default:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1694041157.2211847
    deployments:
      Translator:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
      Summarizer:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
```

The language has updated. Now the returned text is in German instead of French.
```{literalinclude} ../doc_code/production_guide/text_ml.py
:language: python
:start-after: __start_second_client__
:end-before: __end_second_client__
```

## Code Updates

Changing the following values in a deployment's config will trigger redeployment and restart all the deployment's replicas.
- `ray_actor_options`
- `placement_group_bundles`
- `placement_group_strategy`

Changing the following application-level config values is also considered a code update, and all deployments in the application will be restarted.
- `import_path`
- `runtime_env`

:::{warning}
Although you can update your Serve application by deploying an entirely new deployment graph using a different `import_path` and a different `runtime_env`, this is NOT recommended in production.

The best practice for large-scale code updates is to start a new Ray cluster, deploy the updated code to it using `serve deploy`, and then switch traffic from your old cluster to the new one.
:::


(serve-container-runtime-env-guide)=
# Run Multiple Applications in Different Containers

This section explains how to run multiple Serve applications on the same cluster in separate containers with different images.

This feature is experimental and the API is subject to change. If you have additional feature requests or run into issues, please submit them on [Github](https://github.com/ray-project/ray/issues).

## Install Podman

The `container` runtime environment feature uses [Podman](https://podman.io/) to start and run containers. Follow the [Podman Installation Instructions](https://podman.io/docs/installation) to install Podman in the environment for all head and worker nodes.

:::{note}
For Ubuntu, the Podman package is only available in the official repositories for Ubuntu 20.10 and newer. To install Podman in Ubuntu 20.04 or older, you need to first add the software repository as a debian source. Follow these instructions to install Podman on Ubuntu 20.04 or older:

```bash
sudo sh -c "echo 'deb http://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_20.04/ /' > /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list"
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 4D64390375060AA4
sudo apt-get update
sudo apt-get install podman -y
```
:::

## Run a Serve application in a container

This example deploys two applications in separate containers: a Whisper model and a Resnet50 image classification model.

First, install the required dependencies in the images.

:::{warning}
The Ray version and Python version in the container *must* match those of the host environment exactly. Note that for Python, the versions must match down to the patch number.
:::

Save the following to files named `whisper.Dockerfile` and `resnet.Dockerfile`.

::::{tab-set}
:::{tab-item} whisper.Dockerfile
```dockerfile
# Use the latest Ray GPU image, `rayproject/ray:latest-py38-gpu`, so the Whisper model can run on GPUs.
FROM rayproject/ray:latest-py38-gpu

# Install the package `faster_whisper`, which is a dependency for the Whisper model.
RUN pip install faster_whisper==0.10.0
RUN sudo apt-get update && sudo apt-get install curl -y

# Download the source code for the Whisper application into `whisper_example.py`.
RUN curl -O https://raw.githubusercontent.com/ray-project/ray/master/doc/source/serve/doc_code/whisper_example.py
```
:::
:::{tab-item} resnet.Dockerfile
```dockerfile
# Use the latest Ray CPU image, `rayproject/ray:latest-py38-cpu`.
FROM rayproject/ray:latest-py38-cpu

# Install the packages `torch` and `torchvision`, which are dependencies for the ResNet model.
RUN pip install torch==2.0.1 torchvision==0.15.2
RUN sudo apt-get update && sudo apt-get install curl -y

# Download the source code for the ResNet application into `resnet50_example.py`.
RUN curl -O https://raw.githubusercontent.com/ray-project/ray/master/doc/source/serve/doc_code/resnet50_example.py
```
:::
::::

Then, build the corresponding Docker images and push it to your choice of Docker registry. This tutorial uses `alice/whisper_image:latest` and `alice/resnet_image:latest` as placeholder names for the images, but make sure to swap out `alice` for a repo name of your choice.

::::{tab-set}
:::{tab-item} Whisper
```bash
# Build the Docker image from the Dockerfile
export IMG1=alice/whisper_image:latest
docker build -t $IMG1 -f whisper.Dockerfile .
# Push to a Docker registry. This step is unnecessary if you are deploying Serve locally.
docker push $IMG1
```
:::
:::{tab-item} Resnet
```bash
# Build the Docker image from the Dockerfile
export IMG2=alice/resnet_image:latest
docker build -t $IMG2 -f resnet.Dockerfile .
# Push to a Docker registry. This step is unnecessary if you are deploying Serve locally.
docker push $IMG2
```
:::
::::

Finally, you can specify the Docker image within which you want to run each application in the `container` field of an application's runtime environment specification. The `container` field has three fields:
- `image`: (Required) The image to run your application in.
- `worker_path`: The absolute path to `default_worker.py` inside the container.
- `run_options`: Additional options to pass to the `podman run` command used to start a Serve deployment replica in a container. See [podman run documentation](https://docs.podman.io/en/latest/markdown/podman-run.1.html) for a list of all options. If you are familiar with `docker run`, most options work the same way.

The following Serve config runs the `whisper` app with the image `IMG1`, and the `resnet` app with the image `IMG2`. Concretely, all deployment replicas in the applications start and run in containers with the respective images.

```yaml
applications:
  - name: whisper
    import_path: whisper_example:entrypoint
    route_prefix: /whisper
    runtime_env:
      container:
        image: {IMG1}
        worker_path: /home/ray/anaconda3/lib/python3.8/site-packages/ray/_private/workers/default_worker.py
  - name: resnet
    import_path: resnet50_example:app
    route_prefix: /resnet
    runtime_env:
      container:
        image: {IMG2}
        worker_path: /home/ray/anaconda3/lib/python3.8/site-packages/ray/_private/workers/default_worker.py
```

### Send queries



```python
>>> import requests
>>> audio_file = "https://storage.googleapis.com/public-lyrebird-test/test_audio_22s.wav"
>>> resp = requests.post("http://localhost:8000/whisper", json={"filepath": audio_file}) # doctest: +SKIP
>>> resp.json() # doctest: +SKIP
{
    "language": "en",
    "language_probability": 1,
    "duration": 21.775,
    "transcript_text": " Well, think about the time of our ancestors. A ping, a ding, a rustling in the bushes is like, whoo, that means an immediate response. Oh my gosh, what's that thing? Oh my gosh, I have to do it right now. And dude, it's not a tiger, right? Like, but our, our body treats stress as if it's life-threatening because to quote Robert Sapolsky or butcher his quote, he's a Robert Sapolsky is like one of the most incredible stress physiologists of",
    "whisper_alignments": [
        [
            0.0,
            0.36,
            " Well,",
            0.3125
        ],
        ...
    ]
}

>>> image_uri = "https://serve-resnet-benchmark-data.s3.us-west-1.amazonaws.com/000000000019.jpeg"
>>> resp = requests.post("http://localhost:8000/resnet", json={"uri": image_uri}) # doctest: +SKIP
>>> resp.text # doctest: +SKIP
ox
```

## Advanced

### `worker_path` field

Specifying `worker_path` is necessary if the Ray installation directory is different than that of the host (where raylet is running). For instance, if you are running on bare metal, and using a standard Ray Docker image as the base image for your application image, then the Ray installation directory on host likely differs from that of the container.

To find this path, do the following:
1. Run the image using `docker run -it image_id`.
2. Find the Ray installation directory by running `import ray; print(ray.__file__)` in a Python interpreter. This command should print `{path-to-ray-directory}/__init__.py`. Note that if you are using a standard Ray Docker image as the base image, then the Ray installation directory is always at `/home/ray/anaconda3/lib/{python-version}/site-packages/ray/`, where `{python-version}` is something like `python3.8`.
3. The worker path is at `{path-to-ray-directory}/_private/workers/default_worker.py`.

If you set the `worker_path` to an incorrect file path, you will see an error from the raylet like the following:
```
(raylet) python: can't open file '/some/incorrect/path': [Errno 2] No such file or directory
```
This error results from the raylet trying to execute `default_worker.py` inside the container, but not being able to find it.

### Compatibility with other runtime environment fields

Currently, use of the `container` field is not supported with any other field in `runtime_env`. If you have a use case for pairing `container` with another runtime environment feature, submit a feature request on [Github](https://github.com/ray-project/ray/issues).

### Environment variables

All environment variables that start with the prefix `RAY_` (including the two special variables `RAY_RAYLET_PID` and `RAY_JOB_ID`) are propagated into the container's environment at runtime.

### Running the Ray cluster in a Docker container

If raylet is running inside a container, then that container needs the necessary permissions to start a new container. To setup correct permissions, you need to start the container that runs the raylet with the flag `--privileged`.

### Troubleshooting
* **Permission denied: '/tmp/ray/session_2023-11-28_15-27-22_167972_6026/ports_by_node.json.lock'**
  * This error likely occurs because the user running inside the Podman container is different from the host user that started the Ray cluster. The folder `/tmp/ray`, which is volume mounted into the podman container, is owned by the host user that started Ray. The container, on the other hand, is started with the flag `--userns=keep-id`, meaning the host user is mapped into the container as itself. Therefore, permissions issues should only occur if the user inside the container is different from the host user. For instance, if the user on host is `root`, and you're using a container whose base image is a standard Ray image, then by default the container starts with user `ray(1000)`, who won't be able to access the mounted `/tmp/ray` volume.
* **ERRO[0000] 'overlay' is not supported over overlayfs: backing file system is unsupported for this graph driver**
  * This error should only occur when you are running the Ray cluster inside a container. If you see this error when starting the replica actor, try volume mounting `/var/lib/containers` in the container that runs raylet. That is, add `-v /var/lib/containers:/var/lib/containers` to the command that starts the Docker container.
* **cannot clone: Operation not permitted; Error: cannot re-exec process**
  * This error should only occur when you are running the Ray cluster inside a container. This error implies that you don't have the permissions to use Podman to start a container. You need to start the container that runs raylet, with privileged permissions by adding `--privileged`.

(serve-dev-workflow)=

# Development Workflow

This page describes the recommended workflow for developing Ray Serve applications. If you're ready to go to production, jump to the [Production Guide](serve-in-production) section.

## Local Development using `serve.run`

You can use `serve.run` in a Python script to run and test your application locally, using a handle to send requests programmatically rather than over HTTP.

Benefits:

- Self-contained Python is convenient for writing local integration tests.
- No need to deploy to a cloud provider or manage infrastructure.

Drawbacks:

- Doesn't test HTTP endpoints.
- Can't use GPUs if your local machine doesn't have them.

Let's see a simple example.

```{literalinclude} ../doc_code/local_dev.py
:start-after: __local_dev_start__
:end-before: __local_dev_end__
:language: python
```

We can add the code below to deploy and test Serve locally.

```{literalinclude} ../doc_code/local_dev.py
:start-after: __local_dev_handle_start__
:end-before: __local_dev_handle_end__
:language: python
```

## Local Development with HTTP requests

You can use the `serve run` CLI command to run and test your application locally using HTTP to send requests (similar to how you might use the `uvicorn` command if you're familiar with [Uvicorn](https://www.uvicorn.org/)).

Recall our example above:

```{literalinclude} ../doc_code/local_dev.py
:start-after: __local_dev_start__
:end-before: __local_dev_end__
:language: python
```

Now run the following command in your terminal:

```bash
serve run local_dev:app
# 2022-08-11 11:31:47,692 INFO scripts.py:294 -- Deploying from import path: "local_dev:app".
# 2022-08-11 11:31:50,372 INFO worker.py:1481 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265.
# (ServeController pid=9865) INFO 2022-08-11 11:31:54,039 controller 9865 proxy_state.py:129 - Starting HTTP proxy with name 'SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-dff7dc5b97b4a11facaed746f02448224aa0c1fb651988ba7197e949' on node 'dff7dc5b97b4a11facaed746f02448224aa0c1fb651988ba7197e949' listening on '127.0.0.1:8000'
# (ServeController pid=9865) INFO 2022-08-11 11:31:55,373 controller 9865 deployment_state.py:1232 - Adding 1 replicas to deployment 'Doubler'.
# (ServeController pid=9865) INFO 2022-08-11 11:31:55,389 controller 9865 deployment_state.py:1232 - Adding 1 replicas to deployment 'HelloDeployment'.
# (HTTPProxyActor pid=9872) INFO:     Started server process [9872]
# 2022-08-11 11:31:57,383 SUCC scripts.py:315 -- Deployed successfully.
```

The `serve run` command blocks the terminal and can be canceled with Ctrl-C. Typically, `serve run` should not be run simultaneously from multiple terminals, unless each `serve run` is targeting a separate running Ray cluster.

Now that Serve is running, we can send HTTP requests to the application.
For simplicity, we'll just use the `curl` command to send requests from another terminal.

```bash
curl -X PUT "http://localhost:8000/?name=Ray"
# Hello, Ray! Hello, Ray!
```

After you're done testing, you can shut down Ray Serve by interrupting the `serve run` command (e.g., with Ctrl-C):

```console
^C2022-08-11 11:47:19,829       INFO scripts.py:323 -- Got KeyboardInterrupt, shutting down...
(ServeController pid=9865) INFO 2022-08-11 11:47:19,926 controller 9865 deployment_state.py:1257 - Removing 1 replicas from deployment 'Doubler'.
(ServeController pid=9865) INFO 2022-08-11 11:47:19,929 controller 9865 deployment_state.py:1257 - Removing 1 replicas from deployment 'HelloDeployment'.
```

Note that rerunning `serve run` will redeploy all deployments. To prevent redeploying those deployments whose code hasn't changed, you can use `serve deploy`; see the [Production Guide](serve-in-production) for details.

## Testing on a remote cluster

To test on a remote cluster, you'll use `serve run` again, but this time you'll pass in an `--address` argument to specify the address of the Ray cluster to connect to.  For remote clusters, this address has the form `ray://<head-node-ip-address>:10001`; see [Ray Client](ray-client-ref) for more information.

When making the transition from your local machine to a remote cluster, you'll need to make sure your cluster has a similar environment to your local machine--files, environment variables, and Python packages, for example.

Let's see a simple example that just packages the code. Run the following command on your local machine, with your remote cluster head node IP address substituted for `<head-node-ip-address>` in the command:

```bash
serve run  --address=ray://<head-node-ip-address>:10001 --working-dir="./project/src" local_dev:app
```

This connects to the remote cluster with the Ray Client, uploads the `working_dir` directory, and runs your Serve application.  Here, the local directory specified by `working_dir` must contain `local_dev.py` so that it can be uploaded to the cluster and imported by Ray Serve.

Once this is up and running, we can send requests to the application:

```bash
curl -X PUT http://<head-node-ip-address>:8000/?name=Ray
# Hello, Ray! Hello, Ray!
```

For more complex dependencies, including files outside the working directory, environment variables, and Python packages, you can use {ref}`Runtime Environments<runtime-environments>`. This example uses the --runtime-env-json argument:

```bash
serve run  --address=ray://<head-node-ip-address>:10001 --runtime-env-json='{"env_vars": {"MY_ENV_VAR": "my-value"}, "working_dir": "./project/src", "pip": ["requests", "chess"]}' local_dev:app
```

You can also specify the `runtime_env` in a YAML file; see [serve run](#serve-cli) for details.

## What's Next?

View details about your Serve application in the [Ray dashboard](dash-serve-view).
Once you are ready to deploy to production, see the [Production Guide](serve-in-production).


(serve-java-api)=
# Experimental Java API

:::{warning}
Java API support is an experimental feature and subject to change.

The Java API is not currently supported on KubeRay.
:::

Java is a mainstream programming language for production services. Ray Serve offers a native Java API for creating, updating, and managing deployments. You can create Ray Serve deployments using Java and call them via Python, or vice versa.

This section helps you to:

- create, query, and update Java deployments
- configure Java deployment resources
- manage Python deployments using the Java API

```{contents}
```

## Creating a Deployment

By specifying the full name of the class as an argument to the `Serve.deployment()` method, as shown in the code below, you can create and deploy a deployment of the class.

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/ManageDeployment.java
:start-after: docs-create-start
:end-before: docs-create-end
:language: java
```

## Accessing a Deployment

Once a deployment is deployed, you can fetch its instance by name.

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/ManageDeployment.java
:start-after: docs-query-start
:end-before: docs-query-end
:language: java
```

## Updating a Deployment

You can update a deployment's code and configuration and then redeploy it. The following example updates the `"counter"` deployment's initial value to 2.

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/ManageDeployment.java
:start-after: docs-update-start
:end-before: docs-update-end
:language: java
```

## Configuring a Deployment

Ray Serve lets you configure your deployments to:

- scale out by increasing the number of [deployment replicas](serve-architecture-high-level-view)
- assign [replica resources](serve-cpus-gpus) such as CPUs and GPUs.

The next two sections describe how to configure your deployments.

### Scaling Out

By specifying the `numReplicas` parameter, you can change the number of deployment replicas:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/ManageDeployment.java
:start-after: docs-scale-start
:end-before: docs-scale-end
:language: java
```

### Resource Management (CPUs, GPUs)

Through the `rayActorOptions` parameter, you can reserve resources for each deployment replica, such as one GPU:

```{literalinclude} ../../../../java/serve/src/test/java/io/ray/serve/docdemo/ManageDeployment.java
:start-after: docs-resource-start
:end-before: docs-resource-end
:language: java
```

## Managing a Python Deployment

A Python deployment can also be managed and called by the Java API. Suppose you have a Python file `counter.py` in the `/path/to/code/` directory:

```python
from ray import serve

@serve.deployment
class Counter(object):
    def __init__(self, value):
        self.value = int(value)

    def increase(self, delta):
        self.value += int(delta)
        return str(self.value)

```

You can deploy it through the Java API and call it through a `RayServeHandle`:

```java
import io.ray.api.Ray;
import io.ray.serve.api.Serve;
import io.ray.serve.deployment.Deployment;
import io.ray.serve.generated.DeploymentLanguage;
import java.io.File;

public class ManagePythonDeployment {

  public static void main(String[] args) {

    System.setProperty(
        "ray.job.code-search-path",
        System.getProperty("java.class.path") + File.pathSeparator + "/path/to/code/");

    Serve.start(true, false, null);

    Deployment deployment =
        Serve.deployment()
            .setDeploymentLanguage(DeploymentLanguage.PYTHON)
            .setName("counter")
            .setDeploymentDef("counter.Counter")
            .setNumReplicas(1)
            .setInitArgs(new Object[] {"1"})
            .create();
    deployment.deploy(true);

    System.out.println(Ray.get(deployment.getHandle().method("increase").remote("2")));
  }
}

```

:::{note}
Before `Ray.init` or `Serve.start`, you need to specify a directory to find the Python code. For details, please refer to [Cross-Language Programming](cross_language).
:::

## Future Roadmap

In the future, Ray Serve plans to provide more Java features, such as:
- an improved Java API that matches the Python version
- HTTP ingress support
- bring-your-own Java Spring project as a deployment


(serve-advanced-guides)=
# Advanced Guides

```{toctree}
:hidden:

app-builder-guide
advanced-autoscaling
performance
dyn-req-batch
inplace-updates
dev-workflow
grpc-guide
managing-java-deployments
deploy-vm
multi-app-container
```

If you’re new to Ray Serve, start with the [Ray Serve Quickstart](serve-getting-started).

Use these advanced guides for more options and configurations:
- [Pass Arguments to Applications](app-builder-guide)
- [Advanced Ray Serve Autoscaling](serve-advanced-autoscaling)
- [Performance Tuning](serve-perf-tuning)
- [Dynamic Request Batching](serve-performance-batching-requests)
- [In-Place Updates for Serve](serve-inplace-updates)
- [Development Workflow](serve-dev-workflow)
- [gRPC Support](serve-set-up-grpc-service)
- [Ray Serve Dashboard](dash-serve-view)
- [Experimental Java API](serve-java-api)
- [Run Applications in Different Containers](serve-container-runtime-env-guide)


(serve-perf-tuning)=
# Performance Tuning

This section should help you:

- understand Ray Serve's performance characteristics
- find ways to debug and tune your Serve application's performance

:::{note}
This section offers some tips and tricks to improve your Ray Serve application's performance. Check out the [architecture page](serve-architecture) for helpful context, including an overview of the HTTP proxy actor and deployment replica actors.
:::

```{contents}
```

## Performance and benchmarks

Ray Serve is built on top of Ray, so its scalability is bounded by Ray’s scalability. See Ray’s [scalability envelope](https://github.com/ray-project/ray/blob/master/release/benchmarks/README.md) to learn more about the maximum number of nodes and other limitations.

## Debugging performance issues

The performance issue you're most likely to encounter is high latency or low throughput for requests.

Once you set up [monitoring](serve-monitoring) with Ray and Ray Serve, these issues may appear as:

* `serve_num_router_requests_total` staying constant while your load increases
* `serve_deployment_processing_latency_ms` spiking up as queries queue up in the background

The following are ways to address these issues:

1. Make sure you are using the right hardware and resources:
   * Are you reserving GPUs for your deployment replicas using `ray_actor_options` (e.g., `ray_actor_options={“num_gpus”: 1}`)?
   * Are you reserving one or more cores for your deployment replicas using `ray_actor_options` (e.g., `ray_actor_options={“num_cpus”: 2}`)?
   * Are you setting [OMP_NUM_THREADS](serve-omp-num-threads) to increase the performance of your deep learning framework?
2. Try batching your requests. See [Dynamic Request Batching](serve-performance-batching-requests).
3. Consider using `async` methods in your callable. See [the section below](serve-performance-async-methods).
4. Set an end-to-end timeout for your HTTP requests. See [the section below](serve-performance-e2e-timeout).


(serve-performance-async-methods)=
### Using `async` methods

:::{note}
According to the [FastAPI documentation](https://fastapi.tiangolo.com/async/#very-technical-details), `def` endpoint functions are called in a separate threadpool, so you might observe many requests running at the same time inside one replica, and this scenario might cause OOM or resource starvation. In this case, you can try to use `async def` to control the workload performance.
:::

Are you using `async def` in your callable? If you are using `asyncio` and
hitting the same queuing issue mentioned above, you might want to increase
`max_ongoing_requests`. Serve sets a low number (100) by default so the client gets
proper backpressure. You can increase the value in the deployment decorator; e.g.,
`@serve.deployment(max_ongoing_requests=1000)`.

(serve-performance-e2e-timeout)=
### Set an end-to-end request timeout

By default, Serve lets client HTTP requests run to completion no matter how long they take. However, slow requests could bottleneck the replica processing, blocking other requests that are waiting. Set an end-to-end timeout, so slow requests can be terminated and retried.

You can set an end-to-end timeout for HTTP requests by setting the `request_timeout_s` in the `http_options` field of the Serve config. HTTP Proxies wait for that many seconds before terminating an HTTP request. This config is global to your Ray cluster, and it can't be updated during runtime. Use [client-side retries](serve-best-practices-http-requests) to retry requests that time out due to transient failures.

### Give the Serve Controller more time to process requests

The Serve Controller runs on the Ray head node and is responsible for a variety of tasks,
including receiving autoscaling metrics from other Ray Serve components.
If the Serve Controller becomes overloaded
(symptoms might include high CPU usage and a large number of pending `ServeController.record_handle_metrics` tasks),
you can increase the interval between cycles of the control loop
by setting the `RAY_SERVE_CONTROL_LOOP_INTERVAL_S` environment variable (defaults to `0.1` seconds).
This setting gives the Controller more time to process requests and may help alleviate the overload.


(serve-advanced-autoscaling)=

# Advanced Ray Serve Autoscaling

This guide goes over more advanced autoscaling parameters in [autoscaling_config](../api/doc/ray.serve.config.AutoscalingConfig.rst) and an advanced model composition example.


(serve-autoscaling-config-parameters)=
## Autoscaling config parameters

In this section, we go into more detail about Serve autoscaling concepts as well as how to set your autoscaling config.

### [Required] Define the steady state of your system

To define what the steady state of your deployments should be, set values for `target_ongoing_requests` and `max_ongoing_requests`.

#### **target_num_ongoing_requests_per_replica [default=1]**
This parameter is renamed to `target_ongoing_requests`. `target_num_ongoing_requests_per_replica` will be removed in a future release.

#### **target_ongoing_requests [default=1]**
:::{note}
The default for `target_ongoing_requests` will be changed to 2.0 in an upcoming Ray release. You can continue to set it manually to override the default.
:::
Serve scales the number of replicas for a deployment up or down based on the average number of ongoing requests per replica. Specifically, Serve compares the *actual* number of ongoing requests per replica with the target value you set in the autoscaling config and makes upscale or downscale decisions from that. Set the target value with `target_ongoing_requests`, and Serve attempts to ensure that each replica has roughly that number
of requests being processed and waiting in the queue. 

Always load test your workloads. For example, if the use case is latency sensitive, you can lower the `target_ongoing_requests` number to maintain high performance. Benchmark your application code and set this number based on an end-to-end latency objective.

:::{note}
As an example, suppose you have two replicas of a synchronous deployment that has 100ms latency, serving a traffic load of 30 QPS. Then Serve assigns requests to replicas faster than the replicas can finish processing them; more and more requests queue up at the replica (these requests are "ongoing requests") as time progresses, and then the average number of ongoing requests at each replica steadily increases. Latency also increases because new requests have to wait for old requests to finish processing. If you set `target_ongoing_requests = 1`, Serve detects a higher than desired number of ongoing requests per replica, and adds more replicas. At 3 replicas, your system would be able to process 30 QPS with 1 ongoing request per replica on average.
:::

#### **max_concurrent_queries [default=100] (DEPRECATED)**
This parameter is renamed to `max_ongoing_requests`. `max_concurrent_queries` will be removed in a future release.

#### **max_ongoing_requests [default=100]**
:::{note}
The default for `max_ongoing_requests` will be changed to 5 in an upcoming Ray release. You can continue to set it manually to override the default.
:::
There is also a maximum queue limit that proxies respect when assigning requests to replicas. Define the limit with `max_ongoing_requests`. Set `max_ongoing_requests` to ~20 to 50% higher than `target_ongoing_requests`. Note that `target_ongoing_requests` should always be strictly less than `max_ongoing_requests`, otherwise the deployment never scales up.

- Setting it too low limits upscaling. For instance, if your target value is 50 and `max_ongoing_requests` is 51, then even if the traffic increases significantly, the requests will queue up at the proxy instead of at the replicas. As a result, the autoscaler only increases the number of replicas at most 2% at a time, which is very slow.
- Setting it too high can lead to imbalanced routing. Concretely, this can lead to very high tail latencies during upscale, because when the autoscaler is scaling a deployment up due to a traffic spike, most or all of the requests might be assigned to the existing replicas before the new replicas are started.

### [Required] Define upper and lower autoscaling limits

To use autoscaling, you need to define the minimum and maximum number of resources allowed for your system.

* **min_replicas [default=1]**: This is the minimum number of replicas for the deployment. If you want to ensure your system can deal with a certain level of traffic at all times, set `min_replicas` to a positive number. On the other hand, if you anticipate periods of no traffic and want to scale to zero to save cost, set `min_replicas = 0`. Note that setting `min_replicas = 0` causes higher tail latencies; when you start sending traffic, the deployment scales up, and there will be a cold start time as Serve waits for replicas to be started to serve the request.
* **max_replicas [default=1]**: This is the maximum number of replicas for the deployment. This should be greater than `min_replicas`. Ray Serve Autoscaling relies on the Ray Autoscaler to scale up more nodes when the currently available cluster resources (CPUs, GPUs, etc.) are not enough to support more replicas.
* **initial_replicas**: This is the number of replicas that are started initially for the deployment. This defaults to the value for `min_replicas`.


### [Optional] Define how the system reacts to changing traffic

Given a steady stream of traffic and appropriately configured `min_replicas` and `max_replicas`, the steady state of your system is essentially fixed for a chosen configuration value for `target_ongoing_requests`. Before reaching steady state, however, your system is reacting to traffic shifts. How you want your system to react to changes in traffic determines how you want to set the remaining autoscaling configurations.

* **upscale_delay_s [default=30s]**: This defines how long Serve waits before scaling up the number of replicas in your deployment. In other words, this parameter controls the frequency of upscale decisions. If the replicas are *consistently* serving more requests than desired for an `upscale_delay_s` number of seconds, then Serve scales up the number of replicas based on aggregated ongoing requests metrics. For example, if your service is likely to experience bursts of traffic, you can lower `upscale_delay_s` so that your application can react quickly to increases in traffic.

* **downscale_delay_s [default=600s]**: This defines how long Serve waits before scaling down the number of replicas in your deployment. In other words, this parameter controls the frequency of downscale decisions. If the replicas are *consistently* serving less requests than desired for a `downscale_delay_s` number of seconds, then Serve scales down the number of replicas based on aggregated ongoing requests metrics. For example, if your application initializes slowly, you can increase `downscale_delay_s` to make the downscaling happen more infrequently and avoid reinitialization when the application needs to upscale again in the future.

* **upscale_smoothing_factor [default_value=1.0] (DEPRECATED)**: This parameter is renamed to `upscaling_factor`. `upscale_smoothing_factor` will be removed in a future release.

* **downscale_smoothing_factor [default_value=1.0] (DEPRECATED)**: This parameter is renamed to `downscaling_factor`. `downscale_smoothing_factor` will be removed in a future release.

* **upscaling_factor [default_value=1.0]**: The multiplicative factor to amplify or moderate each upscaling decision. For example, when the application has high traffic volume in a short period of time, you can increase `upscaling_factor` to scale up the resource quickly. This parameter is like a "gain" factor to amplify the response of the autoscaling algorithm.

* **downscaling_factor [default_value=1.0]**: The multiplicative factor to amplify or moderate each downscaling decision. For example, if you want your application to be less sensitive to drops in traffic and scale down more conservatively, you can decrease `downscaling_factor` to slow down the pace of downscaling.

* **metrics_interval_s [default_value=10]**: This controls how often each replica sends reports on current ongoing requests to the autoscaler. Note that the autoscaler can't make new decisions if it doesn't receive updated metrics, so you most likely want to set `metrics_interval_s` to a value that is less than or equal to the upscale and downscale delay values. For instance, if you set `upscale_delay_s = 3`, but keep `metrics_interval_s = 10`, the autoscaler only upscales roughly every 10 seconds.

* **look_back_period_s [default_value=30]**: This is the window over which the average number of ongoing requests per replica is calculated.

## Model composition example

Determining the autoscaling configuration for a multi-model application requires understanding each deployment's scaling requirements. Every deployment has a different latency and differing levels of concurrency. As a result, finding the right autoscaling config for a model-composition application requires experimentation.

This example is a simple application with three deployments composed together to build some intuition about multi-model autoscaling. Assume these deployments:
* `HeavyLoad`: A mock 200ms workload with high CPU usage.
* `LightLoad`: A mock 100ms workload with high CPU usage.
* `Driver`: A driver deployment that fans out to the `HeavyLoad` and `LightLoad` deployments and aggregates the two outputs.


### Attempt 1: One `Driver` replica

First consider the following deployment configurations. Because the driver deployment has low CPU usage and is only asynchronously making calls to the downstream deployments, allocating one fixed `Driver` replica is reasonable.

::::{tab-set}

:::{tab-item} Driver

```yaml
- name: Driver
  num_replicas: 1
  max_ongoing_requests: 200
```

:::

:::{tab-item} HeavyLoad

```yaml
- name: HeavyLoad
  max_ongoing_requests: 3
  autoscaling_config:
    target_ongoing_requests: 1
    min_replicas: 0
    initial_replicas: 0
    max_replicas: 200
    upscale_delay_s: 3
    downscale_delay_s: 60
    upscaling_factor: 0.3
    downscaling_factor: 0.3
    metrics_interval_s: 2
    look_back_period_s: 10
```

:::

:::{tab-item} LightLoad

```yaml
- name: LightLoad
  max_ongoing_requests: 3
  autoscaling_config:
    target_ongoing_requests: 1
    min_replicas: 0
    initial_replicas: 0
    max_replicas: 200
    upscale_delay_s: 3
    downscale_delay_s: 60
    upscaling_factor: 0.3
    downscaling_factor: 0.3
    metrics_interval_s: 2
    look_back_period_s: 10
```

:::

:::{tab-item} Application Code

```{literalinclude} ../doc_code/autoscale_model_comp_example.py
:language: python
:start-after: __serve_example_begin__
:end-before: __serve_example_end__
```

:::

::::


Running the same Locust load test from the [Resnet workload](resnet-autoscaling-example) generates the following results:

| | |
| ----------------------- | ---------------------- |
| HeavyLoad and LightLoad Number Replicas | <img src="https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/model_composition_replicas.png" alt="comp" width="600" /> |


As you might expect, the number of autoscaled `LightLoad` replicas is roughly half that of autoscaled `HeavyLoad` replicas. Although the same number of requests per second are sent to both deployments, `LightLoad` replicas can process twice as many requests per second as `HeavyLoad` replicas can, so the deployment should need half as many replicas to handle the same traffic load.

Unfortunately, the service latency rises to from 230 to 400 ms when the number of Locust users increases to 100.

| P50 Latency | QPS |
| ------- | --- |
| ![comp_latency](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/model_comp_latency.svg) | ![comp_rps](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/model_comp_rps.svg) |

Note that the number of `HeavyLoad` replicas should roughly match the number of Locust users to adequately serve the Locust traffic. However, when the number of Locust users increased to 100, the `HeavyLoad` deployment struggled to reach 100 replicas, and instead only reached 65 replicas. The per-deployment latencies reveal the root cause. While `HeavyLoad` and `LightLoad` latencies stayed steady at 200ms and 100ms, `Driver` latencies rose from 230 to 400 ms. This suggests that the high Locust workload may be overwhelming the `Driver` replica and impacting its asynchronous event loop's performance.


### Attempt 2: Autoscale `Driver`

For this attempt, set an autoscaling configuration for `Driver` as well, with the setting `target_ongoing_requests = 20`. Now the deployment configurations are as follows:

::::{tab-set}

:::{tab-item} Driver

```yaml
- name: Driver
  max_ongoing_requests: 200
  autoscaling_config:
    target_ongoing_requests: 20
    min_replicas: 1
    initial_replicas: 1
    max_replicas: 10
    upscale_delay_s: 3
    downscale_delay_s: 60
    upscaling_factor: 0.3
    downscaling_factor: 0.3
    metrics_interval_s: 2
    look_back_period_s: 10
```

:::

:::{tab-item} HeavyLoad

```yaml
- name: HeavyLoad
  max_ongoing_requests: 3
  autoscaling_config:
    target_ongoing_requests: 1
    min_replicas: 0
    initial_replicas: 0
    max_replicas: 200
    upscale_delay_s: 3
    downscale_delay_s: 60
    upscaling_factor: 0.3
    downscaling_factor: 0.3
    metrics_interval_s: 2
    look_back_period_s: 10
```

:::

:::{tab-item} LightLoad

```yaml
- name: LightLoad
  max_ongoing_requests: 3
  autoscaling_config:
    target_ongoing_requests: 1
    min_replicas: 0
    initial_replicas: 0
    max_replicas: 200
    upscale_delay_s: 3
    downscale_delay_s: 60
    upscaling_factor: 0.3
    downscaling_factor: 0.3
    metrics_interval_s: 2
    look_back_period_s: 10
```

:::
::::

Running the same Locust load test again generates the following results:

| | |
| ------------------------------------ | ------------------- |
| HeavyLoad and LightLoad Number Replicas | <img src="https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/model_composition_improved_replicas.png" alt="heavy" width="600"/> |
| Driver Number Replicas | <img src="https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/model_composition_improved_driver_replicas.png" alt="driver" width="600"/>

With up to 6 `Driver` deployments to receive and distribute the incoming requests, the `HeavyLoad` deployment successfully scales up to 90+ replicas, and `LightLoad` up to 47 replicas. This configuration helps the application latency stay consistent as the traffic load increases.

| Improved P50 Latency | Improved RPS |
| ---------------- | ------------ |
| ![comp_latency](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/model_composition_improved_latency.svg) | ![comp_latency](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/model_comp_improved_rps.svg) |


## Troubleshooting guide


### Unstable number of autoscaled replicas

If the number of replicas in your deployment keeps oscillating even though the traffic is relatively stable, try the following:

* Set a smaller `upscaling_factor` and `downscaling_factor`. Setting both values smaller than one helps the autoscaler make more conservative upscale and downscale decisions. It effectively smooths out the replicas graph, and there will be less "sharp edges".
* Set a `look_back_period_s` value that matches the rest of the autoscaling config. For longer upscale and downscale delay values, a longer look back period can likely help stabilize the replica graph, but for shorter upscale and downscale delay values, a shorter look back period may be more appropriate. For instance, the following replica graphs show how a deployment with `upscale_delay_s = 3` works with a longer vs shorter look back period.

| `look_back_period_s = 30` | `look_back_period_s = 3` |
| ------------------------------------------------ | ----------------------------------------------- |
| ![look-back-before](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/look_back_period_before.png) | ![look-back-after](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/look_back_period_after.png) |


### High spikes in latency during bursts of traffic

If you expect your application to receive bursty traffic, and at the same time want the deployments to scale down in periods of inactivity, you are likely concerned about how quickly the deployment can scale up and respond to bursts of traffic. While an increase in latency initially during a burst in traffic may be unavoidable, you can try the following to improve latency during bursts of traffic.

* Set a lower `upscale_delay_s`. The autoscaler always waits `upscale_delay_s` seconds before making a decision to upscale, so lowering this delay allows the autoscaler to react more quickly to changes, especially bursts, of traffic.
* Set a larger `upscaling_factor`. If `upscaling_factor > 1`, then the autoscaler scales up more aggressively than normal. This setting can allow your deployment to be more sensitive to bursts of traffic.
* Lower the `metric_interval_s`. Always set `metric_interval_s` to be less than or equal to `upscale_delay_s`, otherwise upscaling is delayed because the autoscaler doesn't receive fresh information often enough.
* Set a lower `max_ongoing_requests`. If `max_ongoing_requests` is too high relative to `target_ongoing_requests`, then when traffic increases, Serve might assign most or all of the requests to the existing replicas before the new replicas are started. This setting can lead to very high latencies during upscale.


### Deployments scaling down too quickly

You may observe that deployments are scaling down too quickly. Instead, you may want the downscaling to be much more conservative to maximize the availability of your service.

* Set a longer `downscale_delay_s`. The autoscaler always waits `downscale_delay_s` seconds before making a decision to downscale, so by increasing this number, your system has a longer "grace period" after traffic drops before the autoscaler starts to remove replicas.
* Set a smaller `downscaling_factor`. If `downscaling_factor < 1`, then the autoscaler removes *less replicas* than what it thinks it should remove to achieve the target number of ongoing requests. In other words, the autoscaler makes more conservative downscaling decisions.

| `downscaling_factor = 1` | `downscaling_factor = 0.5` |
| ------------------------------------------------ | ----------------------------------------------- |
| ![downscale-smooth-before](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/downscale_smoothing_factor_before.png) | ![downscale-smooth-after](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling-guide/downscale_smoothing_factor_after.png) |


(serve-set-up-grpc-service)=
# Set Up a gRPC Service

This section helps you understand how to:
- Build a user defined gRPC service and protobuf
- Start Serve with gRPC enabled
- Deploy gRPC applications
- Send gRPC requests to Serve deployments
- Check proxy health
- Work with gRPC metadata 
- Use streaming and model composition
- Handle errors
- Use gRPC context


(custom-serve-grpc-service)=
## Define a gRPC service
Running a gRPC server starts with defining gRPC services, RPC methods, and
protobufs similar to the one below.

```{literalinclude} ../doc_code/grpc_proxy/user_defined_protos.proto
:start-after: __begin_proto__
:end-before: __end_proto__
:language: proto
```


This example creates a file named `user_defined_protos.proto` with two
gRPC services: `UserDefinedService` and `ImageClassificationService`.
`UserDefinedService` has three RPC methods: `__call__`, `Multiplexing`, and `Streaming`.
`ImageClassificationService` has one RPC method: `Predict`. Their corresponding input
and output types are also defined specifically for each RPC method.

Once you define the `.proto` services, use `grpcio-tools` to compile python
code for those services. Example command looks like the following:
```bash
python -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. ./user_defined_protos.proto
```

It generates two files: `user_defined_protos_pb2.py` and
`user_defined_protos_pb2_grpc.py`.

For more details on `grpcio-tools` see [https://grpc.io/docs/languages/python/basics/#generating-client-and-server-code](https://grpc.io/docs/languages/python/basics/#generating-client-and-server-code).

:::{note}
Ensure that the generated files are in the same directory as where the Ray cluster
is running so that Serve can import them when starting the proxies.
:::

(start-serve-with-grpc-proxy)=
## Start Serve with gRPC enabled
The [Serve start](https://docs.ray.io/en/releases-2.7.0/serve/api/index.html#serve-start) CLI,
[`ray.serve.start`](https://docs.ray.io/en/releases-2.7.0/serve/api/doc/ray.serve.start.html#ray.serve.start) API,
and [Serve config files](https://docs.ray.io/en/releases-2.7.0/serve/production-guide/config.html#serve-config-files-serve-build)
all support starting Serve with a gRPC proxy. Two options are related to Serve's
gRPC proxy: `grpc_port` and `grpc_servicer_functions`. `grpc_port` is the port for gRPC
proxies to listen to. It defaults to 9000. `grpc_servicer_functions` is a list of import
paths for gRPC `add_servicer_to_server` functions to add to a gRPC proxy. It also
serves as the flag to determine whether to start gRPC server. The default is an empty
list, meaning no gRPC server is started.

::::{tab-set}

:::{tab-item} CLI
```bash
ray start --head
serve start \
  --grpc-port 9000 \
  --grpc-servicer-functions user_defined_protos_pb2_grpc.add_UserDefinedServiceServicer_to_server \
  --grpc-servicer-functions user_defined_protos_pb2_grpc.add_ImageClassificationServiceServicer_to_server

```
:::

:::{tab-item} Python API
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_start_grpc_proxy__
:end-before: __end_start_grpc_proxy__
:language: python
```
:::

:::{tab-item} Serve config file
```yaml
# config.yaml
grpc_options:
  port: 9000
  grpc_servicer_functions:
    - user_defined_protos_pb2_grpc.add_UserDefinedServiceServicer_to_server
    - user_defined_protos_pb2_grpc.add_ImageClassificationServiceServicer_to_server

applications:
  - name: app1
    route_prefix: /app1
    import_path: test_deployment_v2:g
    runtime_env: {}

  - name: app2
    route_prefix: /app2
    import_path: test_deployment_v2:g2
    runtime_env: {}

```

```bash
# Start Serve with above config file.
serve run config.yaml

```

:::

::::

(deploy-serve-grpc-applications)=
## Deploy gRPC applications
gRPC applications in Serve works similarly to HTTP applications. The only difference is
that the input and output of the methods need to match with what's defined in the `.proto`
file and that the method of the application needs to be an exact match (case sensitive)
with the predefined RPC methods. For example, if we want to deploy `UserDefinedService`
with `__call__` method, the method name needs to be `__call__`, the input type needs to
be `UserDefinedMessage`, and the output type needs to be `UserDefinedResponse`. Serve
passes the protobuf object into the method and expects the protobuf object back
from the method. 

Example deployment:
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_grpc_deployment__
:end-before: __end_grpc_deployment__
:language: python
```

Deploy the application:
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_deploy_grpc_app__
:end-before: __end_deploy_grpc_app__
:language: python
```

:::{note}
`route_prefix` is still a required field as of Ray 2.7.0 due to a shared code path with
HTTP. Future releases will make it optional for gRPC.
:::


(send-serve-grpc-proxy-request)=
## Send gRPC requests to serve deployments
Sending a gRPC request to a Serve deployment is similar to sending a gRPC request to
any other gRPC server. Create a gRPC channel and stub, then call the RPC
method on the stub with the appropriate input. The output is the protobuf object
that your Serve application returns.

Sending a gRPC request:
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_send_grpc_requests__
:end-before: __end_send_grpc_requests__
:language: python
```

Read more about gRPC clients in Python: [https://grpc.io/docs/languages/python/basics/#client](https://grpc.io/docs/languages/python/basics/#client)


(serve-grpc-proxy-health-checks)=
## Check proxy health
Similar to HTTP `/-/routes` and `/-/healthz` endpoints, Serve also provides gRPC
service method to be used in health check. 
- `/ray.serve.RayServeAPIService/ListApplications` is used to list all applications
  deployed in Serve. 
- `/ray.serve.RayServeAPIService/Healthz` is used to check the health of the proxy.
  It returns `OK` status and "success" message if the proxy is healthy.

The service method and protobuf are defined as below:
```proto
message ListApplicationsRequest {}

message ListApplicationsResponse {
  repeated string application_names = 1;
}

message HealthzRequest {}

message HealthzResponse {
  string message = 1;
}

service RayServeAPIService {
  rpc ListApplications(ListApplicationsRequest) returns (ListApplicationsResponse);
  rpc Healthz(HealthzRequest) returns (HealthzResponse);
}
```

You can call the service method with the following code:
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_health_check__
:end-before: __end_health_check__
:language: python
```

:::{note}
Serve provides the `RayServeAPIServiceStub` stub, and `HealthzRequest` and
`ListApplicationsRequest` protobufs for you to use. You don't need to generate them
from the proto file. They are available for your reference.
:::

(serve-grpc-metadata)=
## Work with gRPC metadata
Just like HTTP headers, gRPC also supports metadata to pass request related information.
You can pass metadata to Serve's gRPC proxy and Serve knows how to parse and use
them. Serve also passes trailing metadata back to the client.

List of Serve accepted metadata keys:
- `application`: The name of the Serve application to route to. If not passed and only
  one application is deployed, serve routes to the only deployed app automatically.
- `request_id`: The request ID to track the request.
- `multiplexed_model_id`: The model ID to do model multiplexing.

List of Serve returned trailing metadata keys:
- `request_id`: The request ID to track the request.

Example of using metadata:
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_metadata__   
:end-before: __end_metadata__
:language: python
```

(serve-grpc-proxy-more-examples)=
## Use streaming and model composition
gRPC proxy remains at feature parity with HTTP proxy. Here are more examples of using
gRPC proxy for getting streaming response as well as doing model composition.

### Streaming
The `Steaming` method is deployed with the app named "app1" above. The following code
gets a streaming response.
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_streaming__   
:end-before: __end_streaming__
:language: python
```

### Model composition
Assuming we have the below deployments. `ImageDownloader` and `DataPreprocessor` are two
separate steps to download and process the image before PyTorch can run inference.
The `ImageClassifier` deployment initializes the model, calls both
`ImageDownloader` and `DataPreprocessor`, and feed into the resnet model to get the
classes and probabilities of the given image.

```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_model_composition_deployment__   
:end-before: __end_model_composition_deployment__
:language: python
```

We can deploy the application with the following code:
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_model_composition_deploy__   
:end-before: __end_model_composition_deploy__
:language: python
```

The client code to call the application looks like the following:
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_model_composition_client__   
:end-before: __end_model_composition_client__
:language: python
```

:::{note}
At this point, two applications are running on Serve, "app1" and "app2". If more
than one application is running, you need to pass `application` to the
metadata so Serve knows which application to route to.
:::


(serve-grpc-proxy-error-handling)=
## Handle errors
Similar to any other gRPC server, request throws a `grpc.RpcError` when the response
code is not "OK". Put your request code in a try-except block and handle
the error accordingly.
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_error_handle__   
:end-before: __end_error_handle__
:language: python
```

Serve uses the following gRPC error codes:
- `NOT_FOUND`: When multiple applications are deployed to Serve and the application is
  not passed in metadata or passed but no matching application.
- `UNAVAILABLE`: Only on the health check methods when the proxy is in draining state.
  When the health check is throwing `UNAVAILABLE`, it means the health check failed on
  this node and you should no longer route to this node.
- `DEADLINE_EXCEEDED`: The request took longer than the timeout setting and got cancelled.
- `INTERNAL`: Other unhandled errors during the request.

(serve-grpc-proxy-grpc-context)=
## Use gRPC context
Serve provides a [gRPC context object](https://grpc.github.io/grpc/python/grpc.html#grpc.ServicerContext)
to the deployment replica to get information
about the request as well as setting response metadata such as code and details.
If the handler function is defined with a `grpc_context` argument, Serve will pass a
[RayServegRPCContext](../api/doc/ray.serve.grpc_util.RayServegRPCContext.rst) object
in for each request. Below is an example of how to set a custom status code,
details, and trailing metadata.

```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_grpc_context_define_app__
:end-before: __end_grpc_context_define_app__
:language: python
```

The client code is defined like the following to get those attributes.
```{literalinclude} ../doc_code/grpc_proxy/grpc_guide.py
:start-after: __begin_grpc_context_client__
:end-before: __end_grpc_context_client__
:language: python
```

:::{note}
If the handler raises an unhandled exception, Serve will return an `INTERNAL` error code
with the stacktrace in the details, regardless of what code and details
are set in the `RayServegRPCContext` object.
:::


(serve-api)=
# Ray Serve API

## Python API

(core-apis)=

```{eval-rst}
.. module:: ray
```

### Writing Applications

<!---
NOTE: `serve.deployment` and `serve.Deployment` have an autosummary-generated filename collision due to case insensitivity.
This is fixed by added custom filename mappings in `source/conf.py` (look for "autosummary_filename_map").
--->

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: doc/
   :template: autosummary/class_without_init_args.rst

   serve.Deployment
   serve.Application
```

#### Deployment Decorators

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: doc/

   serve.deployment
      :noindex:
   serve.ingress
   serve.batch
   serve.multiplexed
```

#### Deployment Handles

:::{note}
The deprecated `RayServeHandle` and `RayServeSyncHandle` APIs have been fully removed as of Ray 2.10.
See the [model composition guide](serve-model-composition) for how to update code to use the {mod}`DeploymentHandle <ray.serve.handle.DeploymentHandle>` API instead.
:::

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: doc/
   :template: autosummary/class_without_init_args.rst

   serve.handle.DeploymentHandle
   serve.handle.DeploymentResponse
   serve.handle.DeploymentResponseGenerator
```

### Running Applications

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: doc/

   serve.start
   serve.run
   serve.delete
   serve.status
   serve.shutdown
```

### Configurations

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: doc/

   serve.config.ProxyLocation
   serve.config.gRPCOptions
   serve.config.HTTPOptions
   serve.config.AutoscalingConfig
```

#### Advanced APIs

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: doc/

   serve.get_replica_context
   serve.context.ReplicaContext
   serve.get_multiplexed_model_id
   serve.get_app_handle
   serve.get_deployment_handle
   serve.grpc_util.RayServegRPCContext
   serve.exceptions.BackPressureError
   serve.exceptions.RayServeException
```

(serve-cli)=

## Command Line Interface (CLI)

```{eval-rst}
.. click:: ray.serve.scripts:cli
   :prog: serve
   :nested: full
```

(serve-rest-api)=

## Serve REST API

The Serve REST API is exposed at the same port as the Ray Dashboard. The Dashboard port is `8265` by default. This port can be changed using the `--dashboard-port` argument when running `ray start`. All example requests in this section use the default port.

### `PUT "/api/serve/applications/"`

Declaratively deploys a list of Serve applications. If Serve is already running on the Ray cluster, removes all applications not listed in the new config. If Serve is not running on the Ray cluster, starts Serve. See [multi-app config schema](serve-rest-api-config-schema) for the request's JSON schema.

**Example Request**:

```http
PUT /api/serve/applications/ HTTP/1.1
Host: http://localhost:8265/
Accept: application/json
Content-Type: application/json

{
    "applications": [
        {
            "name": "text_app",
            "route_prefix": "/",
            "import_path": "text_ml:app",
            "runtime_env": {
                "working_dir": "https://github.com/ray-project/serve_config_examples/archive/HEAD.zip"
            },
            "deployments": [
                {"name": "Translator", "user_config": {"language": "french"}},
                {"name": "Summarizer"},
            ]
        },
    ]
}
```



**Example Response**


```http
HTTP/1.1 200 OK
Content-Type: application/json
```

### `GET "/api/serve/applications/"`

Gets cluster-level info and comprehensive details on all Serve applications deployed on the Ray cluster. See [metadata schema](serve-rest-api-response-schema) for the response's JSON schema.

```http
GET /api/serve/applications/ HTTP/1.1
Host: http://localhost:8265/
Accept: application/json
```

**Example Response (abridged JSON)**:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
    "controller_info": {
        "node_id": "cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec",
        "node_ip": "10.0.29.214",
        "actor_id": "1d214b7bdf07446ea0ed9d7001000000",
        "actor_name": "SERVE_CONTROLLER_ACTOR",
        "worker_id": "adf416ae436a806ca302d4712e0df163245aba7ab835b0e0f4d85819",
        "log_file_path": "/serve/controller_29778.log"
    },
    "proxy_location": "EveryNode",
    "http_options": {
        "host": "0.0.0.0",
        "port": 8000,
        "root_path": "",
        "request_timeout_s": null,
        "keep_alive_timeout_s": 5
    },
    "grpc_options": {
        "port": 9000,
        "grpc_servicer_functions": []
    },
    "proxies": {
        "cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec": {
            "node_id": "cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec",
            "node_ip": "10.0.29.214",
            "actor_id": "b7a16b8342e1ced620ae638901000000",
            "actor_name": "SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec",
            "worker_id": "206b7fe05b65fac7fdceec3c9af1da5bee82b0e1dbb97f8bf732d530",
            "log_file_path": "/serve/http_proxy_10.0.29.214.log",
            "status": "HEALTHY"
        }
    },
    "deploy_mode": "MULTI_APP",
    "applications": {
        "app1": {
            "name": "app1",
            "route_prefix": "/",
            "docs_path": null,
            "status": "RUNNING",
            "message": "",
            "last_deployed_time_s": 1694042836.1912267,
            "deployed_app_config": {
                "name": "app1",
                "route_prefix": "/",
                "import_path": "src.text-test:app",
                "deployments": [
                    {
                        "name": "Translator",
                        "num_replicas": 1,
                        "user_config": {
                            "language": "german"
                        }
                    }
                ]
            },
            "deployments": {
                "Translator": {
                    "name": "Translator",
                    "status": "HEALTHY",
                    "message": "",
                    "deployment_config": {
                        "name": "Translator",
                        "num_replicas": 1,
                        "max_ongoing_requests": 100,
                        "user_config": {
                            "language": "german"
                        },
                        "graceful_shutdown_wait_loop_s": 2.0,
                        "graceful_shutdown_timeout_s": 20.0,
                        "health_check_period_s": 10.0,
                        "health_check_timeout_s": 30.0,
                        "ray_actor_options": {
                            "runtime_env": {
                                "env_vars": {}
                            },
                            "num_cpus": 1.0
                        },
                        "is_driver_deployment": false
                    },
                    "replicas": [
                        {
                            "node_id": "cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec",
                            "node_ip": "10.0.29.214",
                            "actor_id": "4bb8479ad0c9e9087fee651901000000",
                            "actor_name": "SERVE_REPLICA::app1#Translator#oMhRlb",
                            "worker_id": "1624afa1822b62108ead72443ce72ef3c0f280f3075b89dd5c5d5e5f",
                            "log_file_path": "/serve/deployment_Translator_app1#Translator#oMhRlb.log",
                            "replica_id": "app1#Translator#oMhRlb",
                            "state": "RUNNING",
                            "pid": 29892,
                            "start_time_s": 1694042840.577496
                        }
                    ]
                },
                "Summarizer": {
                    "name": "Summarizer",
                    "status": "HEALTHY",
                    "message": "",
                    "deployment_config": {
                        "name": "Summarizer",
                        "num_replicas": 1,
                        "max_ongoing_requests": 100,
                        "user_config": null,
                        "graceful_shutdown_wait_loop_s": 2.0,
                        "graceful_shutdown_timeout_s": 20.0,
                        "health_check_period_s": 10.0,
                        "health_check_timeout_s": 30.0,
                        "ray_actor_options": {
                            "runtime_env": {},
                            "num_cpus": 1.0
                        },
                        "is_driver_deployment": false
                    },
                    "replicas": [
                        {
                            "node_id": "cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec",
                            "node_ip": "10.0.29.214",
                            "actor_id": "7118ae807cffc1c99ad5ad2701000000",
                            "actor_name": "SERVE_REPLICA::app1#Summarizer#cwiPXg",
                            "worker_id": "12de2ac83c18ce4a61a443a1f3308294caf5a586f9aa320b29deed92",
                            "log_file_path": "/serve/deployment_Summarizer_app1#Summarizer#cwiPXg.log",
                            "replica_id": "app1#Summarizer#cwiPXg",
                            "state": "RUNNING",
                            "pid": 29893,
                            "start_time_s": 1694042840.5789504
                        }
                    ]
                }
            }
        }
    }
}
```

### `DELETE "/api/serve/applications/"`

Shuts down Serve and all applications running on the Ray cluster. Has no effect if Serve is not running on the Ray cluster.

**Example Request**:

```http
DELETE /api/serve/applications/ HTTP/1.1
Host: http://localhost:8265/
Accept: application/json
```

**Example Response**

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

(serve-rest-api-config-schema)=
## Config Schemas

```{eval-rst}
.. currentmodule:: ray.serve
```


```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: doc/

   schema.ServeDeploySchema
   schema.gRPCOptionsSchema
   schema.HTTPOptionsSchema
   schema.ServeApplicationSchema
   schema.DeploymentSchema
   schema.RayActorOptionsSchema
```

(serve-rest-api-response-schema)=
## Response Schemas

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: doc/

   schema.ServeInstanceDetails
   schema.ApplicationDetails
   schema.DeploymentDetails
   schema.ReplicaDetails
```

## Observability

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: doc/

   metrics.Counter
   metrics.Histogram
   metrics.Gauge
   schema.LoggingConfig
```


(serve-handling-dependencies)=
# Handle Dependencies

(serve-runtime-env)=
## Add a runtime environment

The import path (e.g., `text_ml:app`) must be importable by Serve at runtime.
When running locally, this path might be in your current working directory.
However, when running on a cluster you also need to make sure the path is importable.
Build the code into the cluster's container image (see [Cluster Configuration](kuberay-config) for more details) or use a `runtime_env` with a [remote URI](remote-uris) that hosts the code in remote storage.

For an example, see the [Text ML Models application on GitHub](https://github.com/ray-project/serve_config_examples/blob/master/text_ml.py). You can use this config file to deploy the text summarization and translation application to your own Ray cluster even if you don't have the code locally:

```yaml
import_path: text_ml:app

runtime_env:
    working_dir: "https://github.com/ray-project/serve_config_examples/archive/HEAD.zip"
    pip:
      - torch
      - transformers
```

:::{note}
You can also package a deployment graph into a standalone Python package that you can import using a [PYTHONPATH](https://docs.python.org/3.10/using/cmdline.html#envvar-PYTHONPATH) to provide location independence on your local machine. However, the best practice is to use a `runtime_env`, to ensure consistency across all machines in your cluster.
:::

## Dependencies per deployment

Ray Serve also supports serving deployments with different (and possibly conflicting)
Python dependencies.  For example, you can simultaneously serve one deployment
that uses legacy Tensorflow 1 and another that uses Tensorflow 2.

This is supported on Mac OS and Linux using Ray's {ref}`runtime-environments` feature.
As with all other Ray actor options, pass the runtime environment in via `ray_actor_options` in
your deployment.  Be sure to first run `pip install "ray[default]"` to ensure the
Runtime Environments feature is installed.

Example:

```{literalinclude} ../doc_code/varying_deps.py
:language: python
```

:::{tip}
Avoid dynamically installing packages that install from source: these can be slow and
use up all resources while installing, leading to problems with the Ray cluster.  Consider
precompiling such packages in a private repository or Docker image.
:::

The dependencies required in the deployment may be different than
the dependencies installed in the driver program (the one running Serve API
calls). In this case, you should use a delayed import within the class to avoid
importing unavailable packages in the driver.  This applies even when not
using runtime environments.

Example:

```{literalinclude} ../doc_code/delayed_import.py
:language: python
```


(serve-best-practices)=

# Best practices in production

This section helps you:

* Understand best practices when operating Serve in production
* Learn more about managing Serve with the Serve CLI
* Configure your HTTP requests when querying Serve

## CLI best practices

This section summarizes the best practices for deploying to production using the Serve CLI:

* Use `serve run` to manually test and improve your Serve application locally.
* Use `serve build` to create a Serve config file for your Serve application.
    * For development, put your Serve application's code in a remote repository and manually configure the `working_dir` or `py_modules` fields in your Serve config file's `runtime_env` to point to that repository.
    * For production, put your Serve application's code in a custom Docker image instead of a `runtime_env`. See [this tutorial](serve-custom-docker-images) to learn how to create custom Docker images and deploy them on KubeRay.
* Use `serve status` to track your Serve application's health and deployment progress. See [the monitoring guide](serve-in-production-inspecting) for more info.
* Use `serve config` to check the latest config that your Serve application received. This is its goal state. See [the monitoring guide](serve-in-production-inspecting) for more info.
* Make lightweight configuration updates (e.g., `num_replicas` or `user_config` changes) by modifying your Serve config file and redeploying it with `serve deploy`.

(serve-best-practices-http-requests)=

## Client-side HTTP requests

Most examples in these docs use straightforward `get` or `post` requests using Python's `requests` library, such as:

```{literalinclude} ../doc_code/requests_best_practices.py
:start-after: __prototype_code_start__
:end-before: __prototype_code_end__
:language: python
```

This pattern is useful for prototyping, but it isn't sufficient for production. In production, HTTP requests should use:

* Retries: Requests may occasionally fail due to transient issues (e.g., slow network, node failure, power outage, spike in traffic, etc.). Retry failed requests a handful of times to account for these issues.
* Exponential backoff: To avoid bombarding the Serve application with retries during a transient error, apply an exponential backoff on failure. Each retry should wait exponentially longer than the previous one before running. For example, the first retry may wait 0.1s after a failure, and subsequent retries wait 0.4s (4 x 0.1), 1.6s, 6.4s, 25.6s, etc. after the failure.
* Timeouts: Add a timeout to each retry to prevent requests from hanging. The timeout should be longer than the application's latency to give your application enough time to process requests. Additionally, set an [end-to-end timeout](serve-performance-e2e-timeout) in the Serve application, so slow requests don't bottleneck replicas.

```{literalinclude} ../doc_code/requests_best_practices.py
:start-after: __production_code_start__
:end-before: __production_code_end__
:language: python
```

## Load shedding

When a request is sent to a cluster, it's first received by the Serve proxy, which then forwards it to a replica for handling using a {mod}`DeploymentHandle <ray.serve.handle.DeploymentHandle>`.
Replicas can handle up to a configurable number of requests at a time. Configure the number using the `max_ongoing_requests` option.
If all replicas are busy and cannot accept more requests, the request is queued in the {mod}`DeploymentHandle <ray.serve.handle.DeploymentHandle>` until one becomes available.

Under heavy load, {mod}`DeploymentHandle <ray.serve.handle.DeploymentHandle>` queues can grow and cause high tail latency and excessive load on the system.
To avoid instability, it's often preferable to intentionally reject some requests to avoid these queues growing indefinitely.
This technique is called "load shedding," and it allows the system to gracefully handle excessive load without spiking tail latencies or overloading components to the point of failure.

You can configure load shedding for your Serve deployments using the `max_queued_requests` parameter to the {mod}`@serve.deployment <ray.serve.deployment>` decorator.
This controls the maximum number of requests that each {mod}`DeploymentHandle <ray.serve.handle.DeploymentHandle>`, including the Serve proxy, will queue.
Once the limit is reached, enqueueing any new requests immediately raises a {mod}`BackPressureError <ray.serve.exceptions.BackPressureError>`.
HTTP requests will return a `503` status code (service unavailable).

The following example defines a deployment that emulates slow request handling and has `max_concurrent_queries` and `max_queued_requests` configured.

```{literalinclude} ../doc_code/load_shedding.py
:start-after: __example_deployment_start__
:end-before: __example_deployment_end__
:language: python
```

To test the behavior, send HTTP requests in parallel to emulate multiple clients.
Serve accepts `max_concurrent_queries` and `max_queued_requests` requests, and rejects further requests with a `503`, or service unavailable, status.

```{literalinclude} ../doc_code/load_shedding.py
:start-after: __client_test_start__
:end-before: __client_test_end__
:language: python
```

```bash
2024-02-28 11:12:22,287 INFO worker.py:1744 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
(ProxyActor pid=21011) INFO 2024-02-28 11:12:24,088 proxy 127.0.0.1 proxy.py:1140 - Proxy actor 15b7c620e64c8c69fb45559001000000 starting on node ebc04d744a722577f3a049da12c9f83d9ba6a4d100e888e5fcfa19d9.
(ProxyActor pid=21011) INFO 2024-02-28 11:12:24,089 proxy 127.0.0.1 proxy.py:1357 - Starting HTTP server on node: ebc04d744a722577f3a049da12c9f83d9ba6a4d100e888e5fcfa19d9 listening on port 8000
(ProxyActor pid=21011) INFO:     Started server process [21011]
(ServeController pid=21008) INFO 2024-02-28 11:12:24,199 controller 21008 deployment_state.py:1614 - Deploying new version of deployment SlowDeployment in application 'default'. Setting initial target number of replicas to 1.
(ServeController pid=21008) INFO 2024-02-28 11:12:24,300 controller 21008 deployment_state.py:1924 - Adding 1 replica to deployment SlowDeployment in application 'default'.
(ProxyActor pid=21011) WARNING 2024-02-28 11:12:27,141 proxy 127.0.0.1 544437ef-f53a-4991-bb37-0cda0b05cb6a / router.py:96 - Request dropped due to backpressure (num_queued_requests=2, max_queued_requests=2).
(ProxyActor pid=21011) WARNING 2024-02-28 11:12:27,142 proxy 127.0.0.1 44dcebdc-5c07-4a92-b948-7843443d19cc / router.py:96 - Request dropped due to backpressure (num_queued_requests=2, max_queued_requests=2).
(ProxyActor pid=21011) WARNING 2024-02-28 11:12:27,143 proxy 127.0.0.1 83b444ae-e9d6-4ac6-84b7-f127c48f6ba7 / router.py:96 - Request dropped due to backpressure (num_queued_requests=2, max_queued_requests=2).
(ProxyActor pid=21011) WARNING 2024-02-28 11:12:27,144 proxy 127.0.0.1 f92b47c2-6bff-4a0d-8e5b-126d948748ea / router.py:96 - Request dropped due to backpressure (num_queued_requests=2, max_queued_requests=2).
(ProxyActor pid=21011) WARNING 2024-02-28 11:12:27,145 proxy 127.0.0.1 cde44bcc-f3e7-4652-b487-f3f2077752aa / router.py:96 - Request dropped due to backpressure (num_queued_requests=2, max_queued_requests=2).
(ServeReplica:default:SlowDeployment pid=21013) INFO 2024-02-28 11:12:28,168 default_SlowDeployment 8ey9y40a e3b77013-7dc8-437b-bd52-b4839d215212 / replica.py:373 - __CALL__ OK 2007.7ms
(ServeReplica:default:SlowDeployment pid=21013) INFO 2024-02-28 11:12:30,175 default_SlowDeployment 8ey9y40a 601e7b0d-1cd3-426d-9318-43c2c4a57a53 / replica.py:373 - __CALL__ OK 4013.5ms
(ServeReplica:default:SlowDeployment pid=21013) INFO 2024-02-28 11:12:32,183 default_SlowDeployment 8ey9y40a 0655fa12-0b44-4196-8fc5-23d31ae6fcb9 / replica.py:373 - __CALL__ OK 3987.9ms
(ServeReplica:default:SlowDeployment pid=21013) INFO 2024-02-28 11:12:34,188 default_SlowDeployment 8ey9y40a c49dee09-8de1-4e7a-8c2f-8ce3f6d8ef34 / replica.py:373 - __CALL__ OK 3960.8ms
Request finished with status code 200.
Request finished with status code 200.
Request finished with status code 200.
Request finished with status code 200.
```


(serve-custom-docker-images)=

# Custom Docker Images

This section helps you:

* Extend the official Ray Docker images with your own dependencies
* Package your Serve application in a custom Docker image instead of a `runtime_env`
* Use custom Docker images with KubeRay

To follow this tutorial, make sure to install [Docker Desktop](https://docs.docker.com/engine/install/) and create a [Dockerhub](https://hub.docker.com/) account where you can host custom Docker images.

## Working example

Create a Python file called `fake.py` and save the following Serve application to it:

```{literalinclude} ../doc_code/fake_email_creator.py
:start-after: __fake_start__
:end-before: __fake_end__
:language: python
```

This app creates and returns a fake email address. It relies on the [Faker package](https://github.com/joke2k/faker) to create the fake email address. Install the `Faker` package locally to run it:

```console
% pip install Faker==18.13.0

...

% serve run fake:app

...

# In another terminal window:
% curl localhost:8000
john24@example.org
```

This tutorial explains how to package and serve this code inside a custom Docker image.

## Extending the Ray Docker image

The [rayproject](https://hub.docker.com/u/rayproject) organization maintains Docker images with dependencies needed to run Ray. In fact, the [rayproject/ray](https://hub.docker.com/r/rayproject/ray) and [rayproject/ray-ml](https://hub.docker.com/r/rayproject/ray-ml) repos host Docker images for this doc. For instance, [this RayService config](https://github.com/ray-project/kuberay/blob/release-0.6/ray-operator/config/samples/ray_v1alpha1_rayservice.yaml) uses the [rayproject/ray:2.5.0](https://hub.docker.com/layers/rayproject/ray/2.5.0/images/sha256-cb53dcc21af8f913978fd2a3fc57c812f87d99e0b40db6a42ccd6f43eca11281) image hosted by `rayproject/ray`.

You can extend these images and add your own dependencies to them by using them as a base layer in a Dockerfile. For instance, the working example application uses Ray 2.5.0 and Faker 18.13.0. You can create a Dockerfile that extends the `rayproject/ray:2.5.0` by adding the Faker package:

```dockerfile
# File name: Dockerfile
FROM rayproject/ray:2.5.0

RUN pip install Faker==18.13.0
```

In general, the `rayproject/ray` images contain only the dependencies needed to import Ray and the Ray libraries. The `rayproject/ray-ml` images contain additional dependencies (e.g., PyTorch, HuggingFace, etc.) that are useful for machine learning. You can extend images from either of these repos to build your custom images.

Then, you can build this image and push it to your Dockerhub account, so it can be pulled in the future:

```console
% docker build . -t your_dockerhub_username/custom_image_name:latest

...

% docker image push your_dockerhub_username/custom_image_name:latest

...
```

Make sure to replace `your_dockerhub_username` with your DockerHub user name and the `custom_image_name` with the name you want for your image. `latest` is this image's version. If you don't specify a version when you pull the image, then Docker automatically pulls the `latest` version of the package. You can also replace `latest` with a specific version if you prefer.

## Adding your Serve application to the Docker image

During development, it's useful to package your Serve application into a zip file and pull it into your Ray cluster using `runtime_envs`. During production, it's more stable to put the Serve application in the Docker image instead of the `runtime_env` since new nodes won't need to dynamically pull and install the Serve application code before running it.

Use the [WORKDIR](https://docs.docker.com/engine/reference/builder/#workdir) and [COPY](https://docs.docker.com/engine/reference/builder/#copy) commands inside the Dockerfile to install the example Serve application code in your image:

```dockerfile
# File name: Dockerfile
FROM rayproject/ray:2.5.0

RUN pip install Faker==18.13.0

# Set the working dir for the container to /serve_app
WORKDIR /serve_app

# Copies the local `fake.py` file into the WORKDIR
COPY fake.py /serve_app/fake.py
```

KubeRay starts Ray with the `ray start` command inside the `WORKDIR` directory. All the Ray Serve actors are then able to import any dependencies in the directory. By `COPY`ing the Serve file into the `WORKDIR`, the Serve deployments have access to the Serve code without needing a `runtime_env.`

For your applications, you can also add any other dependencies needed for your Serve app to the `WORKDIR` directory.

Build and push this image to Dockerhub. Use the same version as before to overwrite the image stored at that version.

## Using custom Docker images in KubeRay

Run these custom Docker images in KubeRay by adding them to the RayService config. Make the following changes:

1. Set the `rayVersion` in the `rayClusterConfig` to the Ray version used in your custom Docker image.
2. Set the `ray-head` container's `image` to the custom image's name on Dockerhub.
3. Set the `ray-worker` container's `image` to the custom image's name on Dockerhub.
4. Update the  `serveConfigV2` field to remove any `runtime_env` dependencies that are in the container.

A pre-built version of this image is available at [shrekrisanyscale/serve-fake-email-example](https://hub.docker.com/r/shrekrisanyscale/serve-fake-email-example). Try it out by running this RayService config:

```{literalinclude} ../doc_code/fake_email_creator.yaml
:start-after: __fake_config_start__
:end-before: __fake_config_end__
:language: yaml
```


(serve-in-production)=

# Production Guide

```{toctree}
:hidden:

config
kubernetes
docker
fault-tolerance
handling-dependencies
best-practices
```


The recommended way to run Ray Serve in production is on Kubernetes using the [KubeRay](kuberay-quickstart) [RayService](kuberay-rayservice-quickstart) custom resource.
The RayService custom resource automatically handles important production requirements such as health checking, status reporting, failure recovery, and upgrades.
If you're not running on Kubernetes, you can also run Ray Serve on a Ray cluster directly using the Serve CLI.

This section will walk you through a quickstart of how to generate a Serve config file and deploy it using the Serve CLI.
For more details, you can check out the other pages in the production guide:
- Understand the [Serve config file format](serve-in-production-config-file).
- Understand how to [deploy on Kubernetes using KubeRay](serve-in-production-kubernetes).
- Understand how to [monitor running Serve applications](serve-monitoring).

For deploying on VMs instead of Kubernetes, see [Deploy on VM](serve-in-production-deploying).

(serve-in-production-example)=

## Working example: Text summarization and translation application

Throughout the production guide, we will use the following Serve application as a working example.
The application takes in a string of text in English, then summarizes and translates it into French (default), German, or Romanian.

```{literalinclude} ../doc_code/production_guide/text_ml.py
:language: python
:start-after: __example_start__
:end-before: __example_end__
```

Save this code locally in `text_ml.py`.
In development, we would likely use the `serve run` command to iteratively run, develop, and repeat (see the [Development Workflow](serve-dev-workflow) for more information).
When we're ready to go to production, we will generate a structured [config file](serve-in-production-config-file) that acts as the single source of truth for the application.

This config file can be generated using `serve build`:
```
$ serve build text_ml:app -o serve_config.yaml
```

The generated version of this file contains an `import_path`, `runtime_env`, and configuration options for each deployment in the application.
The application needs the `torch` and `transformers` packages, so modify the `runtime_env` field of the generated config to include these two pip packages. Save this config locally in `serve_config.yaml`.

```yaml
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env:
    pip:
      - torch
      - transformers
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: french
  - name: Summarizer
    num_replicas: 1
```

You can use `serve deploy` to deploy the application to a local Ray cluster and `serve status` to get the status at runtime:

```console
# Start a local Ray cluster.
ray start --head

# Deploy the Text ML application to the local Ray cluster.
serve deploy serve_config.yaml
2022-08-16 12:51:22,043 SUCC scripts.py:180 --
Sent deploy request successfully!
 * Use `serve status` to check deployments' statuses.
 * Use `serve config` to see the running app's config.

$ serve status
proxies:
  cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec: HEALTHY
applications:
  default:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1694041157.2211847
    deployments:
      Translator:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
      Summarizer:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
```

Test the application using Python `requests`:

```{literalinclude} ../doc_code/production_guide/text_ml.py
:language: python
:start-after: __start_client__
:end-before: __end_client__
```

To update the application, modify the config file and use `serve deploy` again.

## Next Steps

For a deeper dive into how to deploy, update, and monitor Serve applications, see the following pages:
- Learn the details of the [Serve config file format](serve-in-production-config-file).
- Learn how to [deploy on Kubernetes using KubeRay](serve-in-production-kubernetes).
- Learn how to [build custom Docker images](serve-custom-docker-images) to use with KubeRay.
- Learn how to [monitor running Serve applications](serve-monitoring).

[KubeRay]: kuberay-index
[RayService]: kuberay-rayservice-quickstart


(serve-e2e-ft)=
# Add End-to-End Fault Tolerance

This section helps you:

* Provide additional fault tolerance for your Serve application
* Understand Serve's recovery procedures
* Simulate system errors in your Serve application

:::{admonition} Relevant Guides
:class: seealso
This section discusses concepts from:
* Serve's [architecture guide](serve-architecture)
* Serve's [Kubernetes production guide](serve-in-production-kubernetes)
:::

(serve-e2e-ft-guide)=
## Guide: end-to-end fault tolerance for your Serve app

Serve provides some [fault tolerance](serve-ft-detail) features out of the box. You can provide end-to-end fault tolerance by tuning these features and running Serve on top of [KubeRay].

### Replica health-checking

By default, the Serve controller periodically health-checks each Serve deployment replica and restarts it on failure.

You can define custom application-level health-checks and adjust their frequency and timeout.
To define a custom health-check, add a `check_health` method to your deployment class.
This method should take no arguments and return no result, and it should raise an exception if Ray Serve considers the replica unhealthy.
If the health-check fails, the Serve controller logs the exception, kills the unhealthy replica(s), and restarts them.
You can also use the deployment options to customize how frequently Serve runs the health-check and the timeout after which Serve marks a replica unhealthy.

```{literalinclude} ../doc_code/fault_tolerance/replica_health_check.py
:start-after: __health_check_start__
:end-before: __health_check_end__
:language: python
```

### Worker node recovery

:::{admonition} KubeRay Required
:class: caution, dropdown
You **must** deploy your Serve application with [KubeRay] to use this feature.

See Serve's [Kubernetes production guide](serve-in-production-kubernetes) to learn how you can deploy your app with KubeRay.
:::

By default, Serve can recover from certain failures, such as unhealthy actors. When [Serve runs on Kubernetes](serve-in-production-kubernetes) with [KubeRay], it can also recover from some cluster-level failures, such as dead workers or head nodes.

When a worker node fails, the actors running on it also fail. Serve detects that the actors have failed, and it attempts to respawn the actors on the remaining, healthy nodes. Meanwhile, KubeRay detects that the node itself has failed, so it attempts to restart the worker pod on another running node, and it also brings up a new healthy node to replace it. Once the node comes up, if the pod is still pending, it can be restarted on that node. Similarly, Serve can also respawn any pending actors on that node as well. The deployment replicas running on healthy nodes can continue serving traffic throughout the recovery period.

(serve-e2e-ft-guide-gcs)=
### Head node recovery: Ray GCS fault tolerance

:::{admonition} KubeRay Required
:class: caution, dropdown
You **must** deploy your Serve application with [KubeRay] to use this feature.

See Serve's [Kubernetes production guide](serve-in-production-kubernetes) to learn how you can deploy your app with KubeRay.
:::

In this section, you'll learn how to add fault tolerance to Ray's Global Control Store (GCS), which allows your Serve application to serve traffic even when the head node crashes.

By default, the Ray head node is a single point of failure: if it crashes, the entire Ray cluster crashes and you must restart it. When running on Kubernetes, the `RayService` controller health-checks the Ray cluster and restarts it if this occurs, but this introduces some downtime.

Starting with Ray 2.0+, KubeRay supports [Global Control Store (GCS) fault tolerance](kuberay-gcs-ft), preventing the Ray cluster from crashing if the head node goes down.
While the head node is recovering, Serve applications can still handle traffic with worker nodes but you can't update or recover from other failures like Actors or Worker nodes crashing.
Once the GCS recovers, the cluster returns to normal behavior.

You can enable GCS fault tolerance on KubeRay by adding an external Redis server and modifying your `RayService` Kubernetes object with the following steps:

#### Step 1: Add external Redis server

GCS fault tolerance requires an external Redis database. You can choose to host your own Redis database, or you can use one through a third-party vendor. Use a highly available Redis database for resiliency.

**For development purposes**, you can also host a small Redis database on the same Kubernetes cluster as your Ray cluster. For example, you can add a 1-node Redis cluster by prepending these three Redis objects to your Kubernetes YAML:

(one-node-redis-example)=
```YAML
kind: ConfigMap
apiVersion: v1
metadata:
  name: redis-config
  labels:
    app: redis
data:
  redis.conf: |-
    port 6379
    bind 0.0.0.0
    protected-mode no
    requirepass 5241590000000000
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  labels:
    app: redis
spec:
  type: ClusterIP
  ports:
    - name: redis
      port: 6379
  selector:
    app: redis
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  labels:
    app: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:5.0.8
          command:
            - "sh"
            - "-c"
            - "redis-server /usr/local/etc/redis/redis.conf"
          ports:
            - containerPort: 6379
          volumeMounts:
            - name: config
              mountPath: /usr/local/etc/redis/redis.conf
              subPath: redis.conf
      volumes:
        - name: config
          configMap:
            name: redis-config
---
```

**This configuration is NOT production-ready**, but it's useful for development and testing. When you move to production, it's highly recommended that you replace this 1-node Redis cluster with a highly available Redis cluster.

#### Step 2: Add Redis info to RayService

After adding the Redis objects, you also need to modify the `RayService` configuration.

First, you need to update your `RayService` metadata's annotations:

::::{tab-set}

:::{tab-item} Vanilla Config
```yaml
...
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
  name: rayservice-sample
spec:
...
```
:::

:::{tab-item} Fault Tolerant Config
:selected:
```yaml
...
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
  name: rayservice-sample
  annotations:
    ray.io/ft-enabled: "true"
    ray.io/external-storage-namespace: "my-raycluster-storage-namespace"
spec:
...
```
:::

::::

The annotations are:
* `ray.io/ft-enabled` REQUIRED: Enables GCS fault tolerance when true
* `ray.io/external-storage-namespace` OPTIONAL: Sets the [external storage namespace]

Next, you need to add the `RAY_REDIS_ADDRESS` environment variable to the `headGroupSpec`:

::::{tab-set}

:::{tab-item} Vanilla Config

```yaml
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
    ...
spec:
    ...
    rayClusterConfig:
        headGroupSpec:
            ...
            template:
                ...
                spec:
                    ...
                    env:
                        ...
```

:::

:::{tab-item} Fault Tolerant Config
:selected:

```yaml
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
    ...
spec:
    ...
    rayClusterConfig:
        headGroupSpec:
            ...
            template:
                ...
                spec:
                    ...
                    env:
                        ...
                        - name: RAY_REDIS_ADDRESS
                          value: redis:6379
```
:::

::::

`RAY_REDIS_ADDRESS`'s value should be your Redis database's `redis://` address. It should contain your Redis database's host and port. An [example Redis address](https://www.iana.org/assignments/uri-schemes/prov/rediss) is `redis://user:secret@localhost:6379/0?foo=bar&qux=baz`.

In the example above, the Redis deployment name (`redis`) is the host within the Kubernetes cluster, and the Redis port is `6379`. The example is compatible with the previous section's [example config](one-node-redis-example).

After you apply the Redis objects along with your updated `RayService`, your Ray cluster can recover from head node crashes without restarting all the workers!

:::{seealso}
Check out the KubeRay guide on [GCS fault tolerance](kuberay-gcs-ft) to learn more about how Serve leverages the external Redis cluster to provide head node fault tolerance.
:::

### Spreading replicas across nodes

One way to improve the availability of your Serve application is to spread deployment replicas across multiple nodes so that you still have enough running
replicas to serve traffic even after a certain number of node failures.

By default, Serve soft spreads all deployment replicas but it has a few limitations:

* The spread is soft and best-effort with no guarantee that the it's perfectly even.

* Serve tries to spread replicas among the existing nodes if possible instead of launching new nodes.
For example, if you have a big enough single node cluster, Serve schedules all replicas on that single node assuming
it has enough resources. However, that node becomes the single point of failure.

You can change the spread behavior of your deployment with the `max_replicas_per_node`
[deployment option](../../serve/api/doc/ray.serve.deployment_decorator.rst), which hard limits the number of replicas of a given deployment that can run on a single node.
If you set it to 1 then you're effectively strict spreading the deployment replicas. If you don't set it then there's no hard spread constraint and Serve uses the default soft spread mentioned in the preceding paragraph. `max_replicas_per_node` option is per deployment and only affects the spread of replicas within a deployment. There's no spread between replicas of different deployments.

The following code example shows how to set `max_replicas_per_node` deployment option:

```{testcode}
import ray
from ray import serve

@serve.deployment(max_replicas_per_node=1)
class Deployment1:
  def __call__(self, request):
    return "hello"

@serve.deployment(max_replicas_per_node=2)
class Deployment2:
  def __call__(self, request):
    return "world"
```

This example has two Serve deployments with different `max_replicas_per_node`: `Deployment1` can have at most one replica on each node and `Deployment2` can have at most two replicas on each node. If you schedule two replicas of `Deployment1` and two replicas of `Deployment2`, Serve runs a cluster with at least two nodes, each running one replica of `Deployment1`. The two replicas of `Deployment2` may run on either a single node or across two nodes because either satisfies the `max_replicas_per_node` constraint.

(serve-e2e-ft-behavior)=
## Serve's recovery procedures

This section explains how Serve recovers from system failures. It uses the following Serve application and config as a working example.

::::{tab-set}

:::{tab-item} Python Code
```{literalinclude} ../doc_code/fault_tolerance/sleepy_pid.py
:start-after: __start__
:end-before: __end__
:language: python
```
:::

:::{tab-item} Kubernetes Config
```{literalinclude} ../doc_code/fault_tolerance/k8s_config.yaml
:language: yaml
```
:::

::::

Follow the [KubeRay quickstart guide](kuberay-quickstart) to:
* Install `kubectl` and `Helm`
* Prepare a Kubernetes cluster
* Deploy a KubeRay operator

Then, [deploy the Serve application](serve-deploy-app-on-kuberay) above:

```console
$ kubectl apply -f config.yaml
```

### Worker node failure

You can simulate a worker node failure in the working example. First, take a look at the nodes and pods running in your Kubernetes cluster:

```console
$ kubectl get nodes

NAME                                        STATUS   ROLES    AGE     VERSION
gke-serve-demo-default-pool-ed597cce-nvm2   Ready    <none>   3d22h   v1.22.12-gke.1200
gke-serve-demo-default-pool-ed597cce-m888   Ready    <none>   3d22h   v1.22.12-gke.1200
gke-serve-demo-default-pool-ed597cce-pu2q   Ready    <none>   3d22h   v1.22.12-gke.1200

$ kubectl get pods -o wide

NAME                                                      READY   STATUS    RESTARTS        AGE    IP           NODE                                        NOMINATED NODE   READINESS GATES
ervice-sample-raycluster-thwmr-worker-small-group-bdv6q   1/1     Running   0               3m3s   10.68.2.62   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
ervice-sample-raycluster-thwmr-worker-small-group-pztzk   1/1     Running   0               3m3s   10.68.2.61   gke-serve-demo-default-pool-ed597cce-m888   <none>           <none>
rayservice-sample-raycluster-thwmr-head-28mdh             1/1     Running   1 (2m55s ago)   3m3s   10.68.0.45   gke-serve-demo-default-pool-ed597cce-pu2q   <none>           <none>
redis-75c8b8b65d-4qgfz                                    1/1     Running   0               3m3s   10.68.2.60   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
```

Open a separate terminal window and port-forward to one of the worker nodes:

```console
$ kubectl port-forward ervice-sample-raycluster-thwmr-worker-small-group-bdv6q 8000
Forwarding from 127.0.0.1:8000 -> 8000
Forwarding from [::1]:8000 -> 8000
```

While the `port-forward` is running, you can query the application in another terminal window:

```console
$ curl localhost:8000
418
```

The output is the process ID of the deployment replica that handled the request. The application launches 6 deployment replicas, so if you run the query multiple times, you should see different process IDs:

```console
$ curl localhost:8000
418
$ curl localhost:8000
256
$ curl localhost:8000
385
```

Now you can simulate worker failures. You have two options: kill a worker pod or kill a worker node. Let's start with the worker pod. Make sure to kill the pod that you're **not** port-forwarding to, so you can continue querying the living worker while the other one relaunches.

```console
$ kubectl delete pod ervice-sample-raycluster-thwmr-worker-small-group-pztzk
pod "ervice-sample-raycluster-thwmr-worker-small-group-pztzk" deleted

$ curl localhost:8000
6318
```

While the pod crashes and recovers, the live pod can continue serving traffic!

:::{tip}
Killing a node and waiting for it to recover usually takes longer than killing a pod and waiting for it to recover. For this type of debugging, it's quicker to simulate failures by killing at the pod level rather than at the node level.
:::

You can similarly kill a worker node and see that the other nodes can continue serving traffic:

```console
$ kubectl get pods -o wide

NAME                                                      READY   STATUS    RESTARTS      AGE     IP           NODE                                        NOMINATED NODE   READINESS GATES
ervice-sample-raycluster-thwmr-worker-small-group-bdv6q   1/1     Running   0             65m     10.68.2.62   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
ervice-sample-raycluster-thwmr-worker-small-group-mznwq   1/1     Running   0             5m46s   10.68.1.3    gke-serve-demo-default-pool-ed597cce-m888   <none>           <none>
rayservice-sample-raycluster-thwmr-head-28mdh             1/1     Running   1 (65m ago)   65m     10.68.0.45   gke-serve-demo-default-pool-ed597cce-pu2q   <none>           <none>
redis-75c8b8b65d-4qgfz                                    1/1     Running   0             65m     10.68.2.60   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>

$ kubectl delete node gke-serve-demo-default-pool-ed597cce-m888
node "gke-serve-demo-default-pool-ed597cce-m888" deleted

$ curl localhost:8000
385
```

### Head node failure

You can simulate a head node failure by either killing the head pod or the head node. First, take a look at the running pods in your cluster:

```console
$ kubectl get pods -o wide

NAME                                                      READY   STATUS    RESTARTS      AGE     IP           NODE                                        NOMINATED NODE   READINESS GATES
ervice-sample-raycluster-thwmr-worker-small-group-6f2pk   1/1     Running   0             6m59s   10.68.2.64   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
ervice-sample-raycluster-thwmr-worker-small-group-bdv6q   1/1     Running   0             79m     10.68.2.62   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
rayservice-sample-raycluster-thwmr-head-28mdh             1/1     Running   1 (79m ago)   79m     10.68.0.45   gke-serve-demo-default-pool-ed597cce-pu2q   <none>           <none>
redis-75c8b8b65d-4qgfz                                    1/1     Running   0             79m     10.68.2.60   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
```

Port-forward to one of your worker pods. Make sure this pod is on a separate node from the head node, so you can kill the head node without crashing the worker:

```console
$ kubectl port-forward ervice-sample-raycluster-thwmr-worker-small-group-bdv6q
Forwarding from 127.0.0.1:8000 -> 8000
Forwarding from [::1]:8000 -> 8000
```

In a separate terminal, you can make requests to the Serve application:

```console
$ curl localhost:8000
418
```

You can kill the head pod to simulate killing the Ray head node:

```console
$ kubectl delete pod rayservice-sample-raycluster-thwmr-head-28mdh
pod "rayservice-sample-raycluster-thwmr-head-28mdh" deleted

$ curl localhost:8000
```

If you have configured [GCS fault tolerance](serve-e2e-ft-guide-gcs) on your cluster, your worker pod can continue serving traffic without restarting when the head pod crashes and recovers. Without GCS fault tolerance, KubeRay restarts all worker pods when the head pod crashes, so you'll need to wait for the workers to restart and the deployments to reinitialize before you can port-forward and send more requests.

### Serve controller failure

You can simulate a Serve controller failure by manually killing the Serve actor.

If you're running KubeRay, `exec` into one of your pods:

```console
$ kubectl get pods

NAME                                                      READY   STATUS    RESTARTS   AGE
ervice-sample-raycluster-mx5x6-worker-small-group-hfhnw   1/1     Running   0          118m
ervice-sample-raycluster-mx5x6-worker-small-group-nwcpb   1/1     Running   0          118m
rayservice-sample-raycluster-mx5x6-head-bqjhw             1/1     Running   0          118m
redis-75c8b8b65d-4qgfz                                    1/1     Running   0          3h36m

$ kubectl exec -it rayservice-sample-raycluster-mx5x6-head-bqjhw -- bash
ray@rayservice-sample-raycluster-mx5x6-head-bqjhw:~$
```

You can use the [Ray State API](state-api-cli-ref) to inspect your Serve app:

```console
$ ray summary actors

======== Actors Summary: 2022-10-04 21:06:33.678706 ========
Stats:
------------------------------------
total_actors: 10


Table (group by class):
------------------------------------
    CLASS_NAME              STATE_COUNTS
0   ProxyActor          ALIVE: 3
1   ServeReplica:SleepyPid  ALIVE: 6
2   ServeController         ALIVE: 1

$ ray list actors --filter "class_name=ServeController"

======== List: 2022-10-04 21:09:14.915881 ========
Stats:
------------------------------
Total: 1

Table:
------------------------------
    ACTOR_ID                          CLASS_NAME       STATE    NAME                      PID
 0  70a718c973c2ce9471d318f701000000  ServeController  ALIVE    SERVE_CONTROLLER_ACTOR  48570
```

You can then kill the Serve controller via the Python interpreter. Note that you'll need to use the `NAME` from the `ray list actor` output to get a handle to the Serve controller.

```console
$ python

>>> import ray
>>> controller_handle = ray.get_actor("SERVE_CONTROLLER_ACTOR", namespace="serve")
>>> ray.kill(controller_handle, no_restart=True)
>>> exit()
```

You can use the Ray State API to check the controller's status:

```console
$ ray list actors --filter "class_name=ServeController"

======== List: 2022-10-04 21:36:37.157754 ========
Stats:
------------------------------
Total: 2

Table:
------------------------------
    ACTOR_ID                          CLASS_NAME       STATE    NAME                      PID
 0  3281133ee86534e3b707190b01000000  ServeController  ALIVE    SERVE_CONTROLLER_ACTOR  49914
 1  70a718c973c2ce9471d318f701000000  ServeController  DEAD     SERVE_CONTROLLER_ACTOR  48570
```

You should still be able to query your deployments while the controller is recovering:

```
# If you're running KubeRay, you
# can do this from inside the pod:

$ python

>>> import requests
>>> requests.get("http://localhost:8000").json()
347
```

:::{note}
While the controller is dead, replica health-checking and deployment autoscaling will not work. They'll continue working once the controller recovers.
:::

### Deployment replica failure

You can simulate replica failures by manually killing deployment replicas. If you're running KubeRay, make sure to `exec` into a Ray pod before running these commands.

```console
$ ray summary actors

======== Actors Summary: 2022-10-04 21:40:36.454488 ========
Stats:
------------------------------------
total_actors: 11


Table (group by class):
------------------------------------
    CLASS_NAME              STATE_COUNTS
0   ProxyActor          ALIVE: 3
1   ServeController         ALIVE: 1
2   ServeReplica:SleepyPid  ALIVE: 6

$ ray list actors --filter "class_name=ServeReplica:SleepyPid"

======== List: 2022-10-04 21:41:32.151864 ========
Stats:
------------------------------
Total: 6

Table:
------------------------------
    ACTOR_ID                          CLASS_NAME              STATE    NAME                               PID
 0  39e08b172e66a5d22b2b4cf401000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#RlRptP    203
 1  55d59bcb791a1f9353cd34e301000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#BnoOtj    348
 2  8c34e675edf7b6695461d13501000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#SakmRM    283
 3  a95405318047c5528b7483e701000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#rUigUh    347
 4  c531188fede3ebfc868b73a001000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#gbpoFe    383
 5  de8dfa16839443f940fe725f01000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#PHvdJW    176
```

You can use the `NAME` from the `ray list actor` output to get a handle to one of the replicas:

```console
$ python

>>> import ray
>>> replica_handle = ray.get_actor("SERVE_REPLICA::SleepyPid#RlRptP", namespace="serve")
>>> ray.kill(replica_handle, no_restart=True)
>>> exit()
```

While the replica is restarted, the other replicas can continue processing requests. Eventually the replica restarts and continues serving requests:

```console
$ python

>>> import requests
>>> requests.get("http://localhost:8000").json()
383
```

### HTTPProxy failure

You can simulate HTTPProxy failures by manually killing deployment replicas. If you're running KubeRay, make sure to `exec` into a Ray pod before running these commands.

```console
$ ray summary actors

======== Actors Summary: 2022-10-04 21:51:55.903800 ========
Stats:
------------------------------------
total_actors: 12


Table (group by class):
------------------------------------
    CLASS_NAME              STATE_COUNTS
0   ProxyActor          ALIVE: 3
1   ServeController         ALIVE: 1
2   ServeReplica:SleepyPid  ALIVE: 6

$ ray list actors --filter "class_name=ProxyActor"

======== List: 2022-10-04 21:52:39.853758 ========
Stats:
------------------------------
Total: 3

Table:
------------------------------
    ACTOR_ID                          CLASS_NAME      STATE    NAME                                                                                                 PID
 0  283fc11beebb6149deb608eb01000000  ProxyActor  ALIVE    SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-91f9a685e662313a0075efcb7fd894249a5bdae7ee88837bea7985a0    101
 1  2b010ce28baeff5cb6cb161e01000000  ProxyActor  ALIVE    SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-cc262f3dba544a49ea617d5611789b5613f8fe8c86018ef23c0131eb    133
 2  7abce9dd241b089c1172e9ca01000000  ProxyActor  ALIVE    SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-7589773fc62e08c2679847aee9416805bbbf260bee25331fa3389c4f    267
```

You can use the `NAME` from the `ray list actor` output to get a handle to one of the replicas:

```console
$ python

>>> import ray
>>> proxy_handle = ray.get_actor("SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-91f9a685e662313a0075efcb7fd894249a5bdae7ee88837bea7985a0", namespace="serve")
>>> ray.kill(proxy_handle, no_restart=False)
>>> exit()
```

While the proxy is restarted, the other proxies can continue accepting requests. Eventually the proxy restarts and continues accepting requests. You can use the `ray list actor` command to see when the proxy restarts:

```console
$ ray list actors --filter "class_name=ProxyActor"

======== List: 2022-10-04 21:58:41.193966 ========
Stats:
------------------------------
Total: 3

Table:
------------------------------
    ACTOR_ID                          CLASS_NAME      STATE    NAME                                                                                                 PID
 0  283fc11beebb6149deb608eb01000000  ProxyActor  ALIVE     SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-91f9a685e662313a0075efcb7fd894249a5bdae7ee88837bea7985a0  57317
 1  2b010ce28baeff5cb6cb161e01000000  ProxyActor  ALIVE    SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-cc262f3dba544a49ea617d5611789b5613f8fe8c86018ef23c0131eb    133
 2  7abce9dd241b089c1172e9ca01000000  ProxyActor  ALIVE    SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-7589773fc62e08c2679847aee9416805bbbf260bee25331fa3389c4f    267
```

Note that the PID for the first ProxyActor has changed, indicating that it restarted.

[KubeRay]: kuberay-index
[external storage namespace]: kuberay-external-storage-namespace


(serve-in-production-config-file)=

# Serve Config Files 

This section should help you:

- Understand the Serve config file format.
- Learn how to deploy and update your applications in production using the Serve config.
- Learn how to generate a config file for a list of Serve applications.

The Serve config is the recommended way to deploy and update your applications in production. It allows you to fully configure everything related to Serve, including system-level components like the proxy and application-level options like individual deployment parameters (recall how to [configure Serve deployments](serve-configure-deployment)). One major benefit is you can dynamically update individual deployment parameters by modifying the Serve config, without needing to redeploy or restart your application.

:::{tip}
If you are deploying Serve on a VM, you can use the Serve config with the [serve deploy](serve-in-production-deploying) CLI command. If you are deploying Serve on Kubernetes, you can embed the Serve config in a [RayService](serve-in-production-kubernetes) custom resource in Kubernetes to 
:::

The Serve config is a YAML file with the following format:

```yaml
proxy_location: ...

http_options: 
  host: ...
  port: ...
  request_timeout_s: ...
  keep_alive_timeout_s: ...

grpc_options:
  port: ...
  grpc_servicer_functions: ...

logging_config:
  log_level: ...
  logs_dir: ...
  encoding: ...
  enable_access_log: ...

applications:
- name: ...
  route_prefix: ...
  import_path: ...
  runtime_env: ... 
  deployments:
  - name: ...
    num_replicas: ...
    ...
  - name:
    ...
```

The file contains `proxy_location`, `http_options`, `grpc_options`, `logging_config` and `applications`.

The `proxy_location` field configures where to run proxies to handle traffic to the cluster. You can set `proxy_location` to the following values:
- EveryNode (default): Run a proxy on every node in the cluster that has at least one replica actor.
- HeadOnly: Only run a single proxy on the head node.
- Disabled: Don't run proxies at all. Set this value if you are only making calls to your applications using deployment handles.

The `http_options` are as follows. Note that the HTTP config is global to your Ray cluster, and you can't update it during runtime.

- **`host`**: The host IP address for Serve's HTTP proxies. This is optional and can be omitted. By default, the `host` is set to `0.0.0.0` to expose your deployments publicly. If you're using Kubernetes, you must set `host` to `0.0.0.0` to expose your deployments outside the cluster.
- **`port`**: The port for Serve's HTTP proxies. This parameter is optional and can be omitted. By default, the port is set to `8000`. 
- **`request_timeout_s`**: Allows you to set the end-to-end timeout for a request before terminating and retrying at another replica. By default, the Serve HTTP proxy retries up to `10` times when a response is not received due to failures (for example, network disconnect, request timeout, etc.) By default, there is no request timeout. 
- **`keep_alive_timeout_s`**: Allows you to set the keep alive timeout for the HTTP proxy. For more details, see [here](serve-http-guide-keep-alive-timeout)

The `grpc_options` are as follows. Note that the gRPC config is global to your Ray cluster, and you can't update it during runtime.
- **`port`**: The port that the gRPC proxies listen on. These are optional settings and can be omitted. By default, the port is
  set to `9000`.
- **`grpc_servicer_functions`**: List of import paths for gRPC `add_servicer_to_server` functions to add to Serve's gRPC proxy. The servicer functions need to be importable from the context of where Serve is running. This defaults to an empty list, which means the gRPC server isn't started.

The `logging_config` is global config, you can configure controller & proxy & replica logs. Note that you can also set application and deployment level logging config, which will take precedence over the global config. See logging config API [here](../../serve/api/doc/ray.serve.schema.LoggingConfig.rst) for more details.

These are the fields per application:

- **`name`**: The names for each application that are auto-generated by `serve build`. The name of each application must be unique. 
- **`route_prefix`**: An application can be called via HTTP at the specified route prefix. It defaults to `/`. The route prefix for each application must be unique.
- **`import_path`**: The path to your top-level Serve deployment (or the same path passed to `serve run`). The most minimal config file consists of only an `import_path`.
- **`runtime_env`**: Defines the environment that the application runs in. Use this parameter to package application dependencies such as `pip` packages (see {ref}`Runtime Environments <runtime-environments>` for supported fields). The `import_path` must be available _within_ the `runtime_env` if it's specified. The Serve config's `runtime_env` can only use [remote URIs](remote-uris) in its `working_dir` and `py_modules`; it can't use local zip files or directories. [More details on runtime env](serve-runtime-env).
- **`deployments (optional)`**: A list of deployment options that allows you to override the `@serve.deployment` settings specified in the deployment graph code. Each entry in this list must include the deployment `name`, which must match one in the code. If this section is omitted, Serve launches all deployments in the graph with the parameters specified in the code. See how to [configure serve deployment options](serve-configure-deployment).
- **`args`**: Arguments that are passed to the [application builder](serve-app-builder-guide).

Below is a config for the [`Text ML Model` example](serve-in-production-example) that follows the format explained above:

```yaml
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env:
    pip:
      - torch
      - transformers
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: french
  - name: Summarizer
    num_replicas: 1
```

The file uses the same `text_ml:app` import path that was used with `serve run`, and has two entries in the `deployments` list for the translation and summarization deployments. Both entries contain a `name` setting and some other configuration options such as `num_replicas`.

:::{tip}
Each individual entry in the `deployments` list is optional. In the example config file above, you could omit the `Summarizer`, including its `name` and `num_replicas`, and the file would still be valid. When you deploy the file, the `Summarizer` deployment is still deployed, using the configurations set in the `@serve.deployment` decorator from the application's code.
:::

## Auto-generate the Serve config using `serve build`

You can use a utility to auto-generate this config file from the code. The `serve build` command takes an import path to your application, and it generates a config file containing all the deployments and their parameters in the application code. Tweak these parameters to manage your deployments in production.

```console
$ ls
text_ml.py

$ serve build text_ml:app -o serve_config.yaml

$ ls
text_ml.py
serve_config.yaml
```

(production-config-yaml)=
The `serve_config.yaml` file contains:

```yaml
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

grpc_options:
  port: 9000
  grpc_servicer_functions: []

logging_config:
  encoding: TEXT
  log_level: INFO
  logs_dir: null
  enable_access_log: true

applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env: {}
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: french
  - name: Summarizer
```

Note that the `runtime_env` field will always be empty when using `serve build` and must be set manually. In this case, if `torch` and `transformers` are not installed globally, you should include these two pip packages in the `runtime_env`.

Additionally, `serve build` includes the default HTTP and gPRC options in its
autogenerated files. You can modify these parameters.

(serve-user-config)=
## Dynamically change parameters without restarting replicas (`user_config`)

You can use the `user_config` field to supply a structured configuration for your deployment. You can pass arbitrary JSON serializable objects to the YAML configuration. Serve then applies it to all running and future deployment replicas. The application of user configuration *doesn't* restart the replica. This deployment continuity means that you can use this field to dynamically:
- adjust model weights and versions without restarting the cluster.
- adjust traffic splitting percentage for your model composition graph.
- configure any feature flag, A/B tests, and hyper-parameters for your deployments.

To enable the `user_config` feature, implement a `reconfigure` method that takes a JSON-serializable object (e.g., a Dictionary, List, or String) as its only argument:

```python
@serve.deployment
class Model:
    def reconfigure(self, config: Dict[str, Any]):
        self.threshold = config["threshold"]
```

If you set the `user_config` when you create the deployment (that is, in the decorator or the Serve config file), Ray Serve calls this `reconfigure` method right after the deployment's `__init__` method, and passes the `user_config` in as an argument. You can also trigger the `reconfigure` method by updating your Serve config file with a new `user_config` and reapplying it to the Ray cluster. See [In-place Updates](serve-inplace-updates) for more information.

The corresponding YAML snippet is:

```yaml
...
deployments:
    - name: Model
      user_config:
        threshold: 1.5
```





(serve-in-production-kubernetes)=

# Deploy on Kubernetes

This section should help you:

- understand how to install and use the [KubeRay] operator.
- understand how to deploy a Ray Serve application using a [RayService].
- understand how to monitor and update your application.

Deploying Ray Serve on Kubernetes provides the scalable compute of Ray Serve and operational benefits of Kubernetes.
This combination also allows you to integrate with existing applications that may be running on Kubernetes. When running on Kubernetes, use the [RayService] controller from [KubeRay].

> NOTE: [Anyscale](https://www.anyscale.com/get-started) is a managed Ray solution that provides high-availability, high-performance autoscaling, multi-cloud clusters, spot instance support, and more out of the box.

A [RayService] CR encapsulates a multi-node Ray Cluster and a Serve application that runs on top of it into a single Kubernetes manifest.
Deploying, upgrading, and getting the status of the application can be done using standard `kubectl` commands.
This section walks through how to deploy, monitor, and upgrade the [Text ML example](serve-in-production-example) on Kubernetes.

(serve-installing-kuberay-operator)=

## Installing the KubeRay operator

Follow the [KubeRay quickstart guide](kuberay-quickstart) to:
* Install `kubectl` and `Helm`
* Prepare a Kubernetes cluster
* Deploy a KubeRay operator

## Setting up a RayService custom resource (CR)
Once the KubeRay controller is running, manage your Ray Serve application by creating and updating a `RayService` CR ([example](https://github.com/ray-project/kuberay/blob/5b1a5a11f5df76db2d66ed332ff0802dc3bbff76/ray-operator/config/samples/ray-service.text-ml.yaml)).

Under the `spec` section in the `RayService` CR, set the following fields:

**`serveConfigV2`**: Represents the configuration that Ray Serve uses to deploy the application. Using `serve build` to print the Serve configuration and copy-paste it directly into your [Kubernetes config](serve-in-production-kubernetes) and `RayService` CR.

**`rayClusterConfig`**: Populate this field with the contents of the `spec` field from the `RayCluster` CR YAML file. Refer to [KubeRay configuration](kuberay-config) for more details.

:::{tip}
To enhance the reliability of your application, particularly when dealing with large dependencies that may require a significant amount of time to download, consider including the dependencies in your image's Dockerfile, so the dependencies are available as soon as the pods start.
:::

(serve-deploy-app-on-kuberay)=
## Deploying a Serve application

When the `RayService` is created, the `KubeRay` controller first creates a Ray cluster using the provided configuration.
Then, once the cluster is running, it deploys the Serve application to the cluster using the [REST API](serve-in-production-deploying).
The controller also creates a Kubernetes Service that can be used to route traffic to the Serve application.

To see an example, deploy the [Text ML example](serve-in-production-example).
The Serve config for the example is embedded into [this sample `RayService` CR](https://github.com/ray-project/kuberay/blob/5b1a5a11f5df76db2d66ed332ff0802dc3bbff76/ray-operator/config/samples/ray-service.text-ml.yaml).
Save this CR locally to a file named `ray-service.text-ml.yaml`:

:::{note}
- The example `RayService` uses very low `numCpus` values for demonstration purposes. In production, provide more resources to the Serve application.
Learn more about how to configure KubeRay clusters [here](kuberay-config).
- If you have dependencies that must be installed during deployment, you can add them to the `runtime_env` in the Deployment code. Learn more [here](serve-handling-dependencies)
:::

```console
$ curl -o ray-service.text-ml.yaml https://raw.githubusercontent.com/ray-project/kuberay/5b1a5a11f5df76db2d66ed332ff0802dc3bbff76/ray-operator/config/samples/ray-service.text-ml.yaml
```

To deploy the example, we simply `kubectl apply` the CR.
This creates the underlying Ray cluster, consisting of a head and worker node pod (see [Ray Clusters Key Concepts](../../cluster/key-concepts.rst) for more details on Ray clusters), as well as the service that can be used to query our application:

```console
$ kubectl apply -f ray-service.text-ml.yaml

$ kubectl get rayservices
NAME                AGE
rayservice-sample   7s

$ kubectl get pods
NAME                                                      READY   STATUS    RESTARTS   AGE
ervice-sample-raycluster-454c4-worker-small-group-b6mmg   1/1     Running   0          XXs
kuberay-operator-7fbdbf8c89-4lrnr                         1/1     Running   0          XXs
rayservice-sample-raycluster-454c4-head-krk9d             1/1     Running   0          XXs

$ kubectl get services

rayservice-sample-head-svc                         ClusterIP   ...        8080/TCP,6379/TCP,8265/TCP,10001/TCP,8000/TCP,52365/TCP   XXs
rayservice-sample-raycluster-454c4-dashboard-svc   ClusterIP   ...        52365/TCP                                                 XXs
rayservice-sample-raycluster-454c4-head-svc        ClusterIP   ...        8000/TCP,52365/TCP,8080/TCP,6379/TCP,8265/TCP,10001/TCP   XXs
rayservice-sample-serve-svc                        ClusterIP   ...        8000/TCP                                                  XXs
```

Note that the `rayservice-sample-serve-svc` above is the one that can be used to send queries to the Serve application -- this will be used in the next section.

## Querying the application

Once the `RayService` is running, we can query it over HTTP using the service created by the KubeRay controller.
This service can be queried directly from inside the cluster, but to access it from your laptop you'll need to configure a [Kubernetes ingress](kuberay-networking) or use port forwarding as below:

```console
$ kubectl port-forward service/rayservice-sample-serve-svc 8000
$ curl -X POST -H "Content-Type: application/json" localhost:8000/summarize_translate -d '"It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief"'
c'était le meilleur des temps, c'était le pire des temps .
```

(serve-getting-status-kubernetes)=
## Getting the status of the application

As the `RayService` is running, the `KubeRay` controller continually monitors it and writes relevant status updates to the CR.
You can view the status of the application using `kubectl describe`.
This includes the status of the cluster, events such as health check failures or restarts, and the application-level statuses reported by [`serve status`](serve-in-production-inspecting).

```console
$ kubectl get rayservices
NAME                AGE
rayservice-sample   7s

$ kubectl describe rayservice rayservice-sample
...
Status:
  Active Service Status:
    Application Statuses:
      text_ml_app:
        Health Last Update Time:  2023-09-07T01:21:30Z
        Last Update Time:         2023-09-07T01:21:30Z
        Serve Deployment Statuses:
          text_ml_app_Summarizer:
            Health Last Update Time:  2023-09-07T01:21:30Z
            Last Update Time:         2023-09-07T01:21:30Z
            Status:                   HEALTHY
          text_ml_app_Translator:
            Health Last Update Time:  2023-09-07T01:21:30Z
            Last Update Time:         2023-09-07T01:21:30Z
            Status:                   HEALTHY
        Status:                       RUNNING
    Dashboard Status:
      Health Last Update Time:  2023-09-07T01:21:30Z
      Is Healthy:               true
      Last Update Time:         2023-09-07T01:21:30Z
    Ray Cluster Name:           rayservice-sample-raycluster-kkd2p
    Ray Cluster Status:
      Head:
  Observed Generation:  1
  Pending Service Status:
    Dashboard Status:
    Ray Cluster Status:
      Head:
  Service Status:  Running
Events:
  Type    Reason   Age                      From                   Message
  ----    ------   ----                     ----                   -------
  Normal  Running  2m15s (x29791 over 16h)  rayservice-controller  The Serve applicaton is now running and healthy.
```

## Updating the application

To update the `RayService`, modify the manifest and apply it use `kubectl apply`.
There are two types of updates that can occur:
- *Application-level updates*: when only the Serve config options are changed, the update is applied _in-place_ on the same Ray cluster. This enables [lightweight updates](serve-in-production-lightweight-update) such as scaling a deployment up or down or modifying autoscaling parameters.
- *Cluster-level updates*: when the `RayCluster` config options are changed, such as updating the container image for the cluster, it may result in a cluster-level update. In this case, a new cluster is started, and the application is deployed to it. Once the new cluster is ready, the Kubernetes service is updated to point to the new cluster and the previous cluster is terminated. There should not be any downtime for the application, but note that this requires the Kubernetes cluster to be large enough to schedule both Ray clusters.

### Example: Serve config update

In the Text ML example above, change the language of the Translator in the Serve config to German:

```yaml
  - name: Translator
    num_replicas: 1
    user_config:
      language: german
```

Now to update the application we apply the modified manifest:

```console
$ kubectl apply -f ray-service.text-ml.yaml

$ kubectl describe rayservice rayservice-sample
...
  Serve Deployment Statuses:
    text_ml_app_Translator:
      Health Last Update Time:  2023-09-07T18:21:36Z
      Last Update Time:         2023-09-07T18:21:36Z
      Status:                   UPDATING
...
```

Query the application to see a different translation in German:

```console
$ curl -X POST -H "Content-Type: application/json" localhost:8000/summarize_translate -d '"It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief"'
Es war die beste Zeit, es war die schlimmste Zeit .
```

### Updating the RayCluster config

The process of updating the RayCluster config is the same as updating the Serve config.
For example, we can update the number of worker nodes to 2 in the manifest:

```console
workerGroupSpecs:
  # the number of pods in the worker group.
  - replicas: 2
```

```console
$ kubectl apply -f ray-service.text-ml.yaml

$ kubectl describe rayservice rayservice-sample
...
  pendingServiceStatus:
    appStatus: {}
    dashboardStatus:
      healthLastUpdateTime: "2022-07-18T21:54:53Z"
      lastUpdateTime: "2022-07-18T21:54:54Z"
    rayClusterName: rayservice-sample-raycluster-bshfr
    rayClusterStatus: {}
...
```

In the status, you can see that the `RayService` is preparing a pending cluster.
After the pending cluster is healthy, it becomes the active cluster and the previous cluster is terminated.

## Autoscaling
You can configure autoscaling for your Serve application by setting the autoscaling field in the Serve config. Learn more about the configuration options in the [Serve Autoscaling Guide](serve-autoscaling).

To enable autoscaling in a KubeRay Cluster, you need to set `enableInTreeAutoscaling` to True. Additionally, there are other options available to configure the autoscaling behavior. For further details, please refer to the documentation [here](serve-autoscaling).


:::{note}
In most use cases, it is recommended to enable Kubernetes autoscaling to fully utilize the resources in your cluster. If you are using GKE, you can utilize the AutoPilot Kubernetes cluster. For instructions, see [Create an Autopilot Cluster](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-an-autopilot-cluster). For EKS, you can enable Kubernetes cluster autoscaling by utilizing the Cluster Autoscaler. For detailed information, see [Cluster Autoscaler on AWS](https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/cloudprovider/aws/README.md). To understand the relationship between Kubernetes autoscaling and Ray autoscaling, see [Ray Autoscaler with Kubernetes Cluster Autoscaler](kuberay-autoscaler-with-ray-autoscaler).
:::

## Load balancer
Set up ingress to expose your Serve application with a load balancer. See [this configuration](https://github.com/ray-project/kuberay/blob/v1.0.0/ray-operator/config/samples/ray-service-alb-ingress.yaml)

:::{note}
- Ray Serve runs HTTP proxy on every node, allowing you to use `/-/routes` as the endpoint for node health checks.
- Ray Serve uses port 8000 as the default HTTP proxy traffic port. You can change the port by setting `http_options` in the Serve config. Learn more details [here](serve-multi-application).
:::

## Monitoring
Monitor your Serve application using the Ray Dashboard.
- Learn more about how to configure and manage Dashboard [here](observability-configure-manage-dashboard).
- Learn about the Ray Serve Dashboard [here](serve-monitoring).
- Learn how to set up [Prometheus](prometheus-setup) and [Grafana](grafana) for Dashboard.
- Learn about the [Ray Serve logs](serve-logging) and how to [persistent logs](kuberay-logging) on Kubernetes.

:::{note}
- To troubleshoot application deployment failures in Serve, you can check the KubeRay operator logs by running `kubectl logs -f <kuberay-operator-pod-name>` (e.g., `kubectl logs -f kuberay-operator-7447d85d58-lv7pf`). The KubeRay operator logs contain information about the Serve application deployment event and Serve application health checks.
- You can also check the controller log and deployment log, which are located under `/tmp/ray/session_latest/logs/serve/` in both the head node pod and worker node pod. These logs contain information about specific deployment failure reasons and autoscaling events.
:::

## Next Steps

See [Add End-to-End Fault Tolerance](serve-e2e-ft) to learn more about Serve's failure conditions and how to guard against them.

[KubeRay]: kuberay-quickstart
[RayService]: kuberay-rayservice-quickstart


