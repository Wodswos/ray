.. _namespaces-guide:

Using Namespaces
================

A namespace is a logical grouping of jobs and named actors. When an actor is
named, its name must be unique within the namespace.

In order to set your applications namespace, it should be specified when you
first connect to the cluster.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/namespaces.py
          :language: python
          :start-after: __init_namespace_start__
          :end-before: __init_namespace_end__

    .. tab-item:: Java

        .. code-block:: java

          System.setProperty("ray.job.namespace", "hello"); // set it before Ray.init()
          Ray.init();

    .. tab-item:: C++

        .. code-block:: c++

          ray::RayConfig config;
          config.ray_namespace = "hello";
          ray::Init(config);

Please refer to `Driver Options <configure.html#driver-options>`__ for ways of configuring a Java application.

Named actors are only accessible within their namespaces.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/namespaces.py
          :language: python
          :start-after: __actor_namespace_start__
          :end-before: __actor_namespace_end__

    .. tab-item:: Java

        .. code-block:: java

            // `ray start --head` has been run to launch a local cluster.

            // Job 1 creates two actors, "orange" and "purple" in the "colors" namespace.
            System.setProperty("ray.address", "localhost:10001");
            System.setProperty("ray.job.namespace", "colors");
            try {
                Ray.init();
                Ray.actor(Actor::new).setName("orange").remote();
                Ray.actor(Actor::new).setName("purple").remote();
            } finally {
                Ray.shutdown();
            }

            // Job 2 is now connecting to a different namespace.
            System.setProperty("ray.address", "localhost:10001");
            System.setProperty("ray.job.namespace", "fruits");
            try {
                Ray.init();
                // This fails because "orange" was defined in the "colors" namespace.
                Ray.getActor("orange").isPresent(); // return false
                // This succceeds because the name "orange" is unused in this namespace.
                Ray.actor(Actor::new).setName("orange").remote();
                Ray.actor(Actor::new).setName("watermelon").remote();
            } finally {
                Ray.shutdown();
            }

            // Job 3 connects to the original "colors" namespace.
            System.setProperty("ray.address", "localhost:10001");
            System.setProperty("ray.job.namespace", "colors");
            try {
                Ray.init();
                // This fails because "watermelon" was in the fruits namespace.
                Ray.getActor("watermelon").isPresent(); // return false
                // This returns the "orange" actor we created in the first job, not the second.
                Ray.getActor("orange").isPresent(); // return true
            } finally {
                Ray.shutdown();
            }

    .. tab-item:: C++

        .. code-block:: c++

            // `ray start --head` has been run to launch a local cluster.

            // Job 1 creates two actors, "orange" and "purple" in the "colors" namespace.
            ray::RayConfig config;
            config.ray_namespace = "colors";
            ray::Init(config);
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("orange").Remote();
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("purple").Remote();
            ray::Shutdown();

            // Job 2 is now connecting to a different namespace.
            ray::RayConfig config;
            config.ray_namespace = "fruits";
            ray::Init(config);
            // This fails because "orange" was defined in the "colors" namespace.
            ray::GetActor<Counter>("orange"); // return nullptr;
            // This succeeds because the name "orange" is unused in this namespace.
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("orange").Remote();
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("watermelon").Remote();
            ray::Shutdown();

            // Job 3 connects to the original "colors" namespace.
            ray::RayConfig config;
            config.ray_namespace = "colors";
            ray::Init(config);
            // This fails because "watermelon" was in the fruits namespace.
            ray::GetActor<Counter>("watermelon"); // return nullptr;
            // This returns the "orange" actor we created in the first job, not the second.
            ray::GetActor<Counter>("orange");
            ray::Shutdown();

Specifying namespace for named actors
-------------------------------------

You can specify a namespace for a named actor while creating it. The created actor belongs to
the specified namespace, no matter what namespace of the current job is.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/namespaces.py
          :language: python
          :start-after: __specify_actor_namespace_start__
          :end-before: __specify_actor_namespace_end__


    .. tab-item:: Java

        .. code-block:: java

            // `ray start --head` has been run to launch a local cluster.

            System.setProperty("ray.address", "localhost:10001");
            try {
                Ray.init();
                // Create an actor with specified namespace.
                Ray.actor(Actor::new).setName("my_actor", "actor_namespace").remote();
                // It is accessible in its namespace.
                Ray.getActor("my_actor", "actor_namespace").isPresent(); // return true

            } finally {
                Ray.shutdown();
            }

    .. tab-item:: C++

        .. code-block::

            // `ray start --head` has been run to launch a local cluster.
            ray::RayConfig config;
            ray::Init(config);
            // Create an actor with specified namespace.
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("my_actor", "actor_namespace").Remote();
            // It is accessible in its namespace.
            ray::GetActor<Counter>("orange");
            ray::Shutdown();`


Anonymous namespaces
--------------------

When a namespace is not specified, Ray will place your job in an anonymous
namespace. In an anonymous namespace, your job will have its own namespace and
will not have access to actors in other namespaces.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/namespaces.py
          :language: python
          :start-after: __anonymous_namespace_start__
          :end-before: __anonymous_namespace_end__

    .. tab-item:: Java

        .. code-block:: java

            // `ray start --head` has been run to launch a local cluster.

            // Job 1 connects to an anonymous namespace by default.
            System.setProperty("ray.address", "localhost:10001");
            try {
                Ray.init();
                Ray.actor(Actor::new).setName("my_actor").remote();
            } finally {
                Ray.shutdown();
            }

            // Job 2 connects to a _different_ anonymous namespace by default
            System.setProperty("ray.address", "localhost:10001");
            try {
                Ray.init();
                // This succeeds because the second job is in its own namespace.
                Ray.actor(Actor::new).setName("my_actor").remote();
            } finally {
                Ray.shutdown();
            }

    .. tab-item:: C++

        .. code-block:: c++

            // `ray start --head` has been run to launch a local cluster.

            // Job 1 connects to an anonymous namespace by default.
            ray::RayConfig config;
            ray::Init(config);
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("my_actor").Remote();
            ray::Shutdown();

            // Job 2 connects to a _different_ anonymous namespace by default
            ray::RayConfig config;
            ray::Init(config);
            // This succeeds because the second job is in its own namespace.
            ray::Actor(RAY_FUNC(Counter::FactoryCreate)).SetName("my_actor").Remote();
            ray::Shutdown();

.. note::

     Anonymous namespaces are implemented as UUID's. This makes it possible for
     a future job to manually connect to an existing anonymous namespace, but
     it is not recommended.


Getting the current namespace
-----------------------------
You can access to the current namespace using :ref:`runtime_context APIs <runtime-context-apis>`.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/namespaces.py
          :language: python
          :start-after: __get_namespace_start__
          :end-before: __get_namespace_end__


    .. tab-item:: Java

        .. code-block:: java

            System.setProperty("ray.job.namespace", "colors");
            try {
                Ray.init();
                // Will print namespace name "colors".
                System.out.println(Ray.getRuntimeContext().getNamespace());
            } finally {
                Ray.shutdown();
            }

    .. tab-item:: C++

        .. code-block:: c++

            ray::RayConfig config;
            config.ray_namespace = "colors";
            ray::Init(config);
            // Will print namespace name "colors".
            std::cout << ray::GetNamespace() << std::endl;
            ray::Shutdown();


Advanced Topics
===============

This section covers extended topics on how to use Ray.

.. toctree::
    :maxdepth: -1

    tips-for-first-time
    starting-ray
    ray-generator
    namespaces
    cross-language
    using-ray-with-jupyter
    ray-dag
    miscellaneous
    runtime_env_auth
    user-spawn-processes


.. _handling_dependencies:

Environment Dependencies
========================

Your Ray application may have dependencies that exist outside of your Ray script. For example:

* Your Ray script may import/depend on some Python packages.
* Your Ray script may be looking for some specific environment variables to be available.
* Your Ray script may import some files outside of the script.

One frequent problem when running on a cluster is that Ray expects these "dependencies" to exist on each Ray node. If these are not present, you may run into issues such as ``ModuleNotFoundError``, ``FileNotFoundError`` and so on.

To address this problem, you can (1) prepare your dependencies on the cluster in advance (e.g. using a container image) using the Ray :ref:`Cluster Launcher <vm-cluster-quick-start>`, or (2) use Ray's :ref:`runtime environments <runtime-environments>` to install them on the fly.

For production usage or non-changing environments, we recommend installing your dependencies into a container image and specifying the image using the Cluster Launcher.
For dynamic environments (e.g. for development and experimentation), we recommend using runtime environments.


Concepts
--------

- **Ray Application**.  A program including a Ray script that calls ``ray.init()`` and uses Ray tasks or actors.

- **Dependencies**, or **Environment**.  Anything outside of the Ray script that your application needs to run, including files, packages, and environment variables.

- **Files**. Code files, data files or other files that your Ray application needs to run.

- **Packages**. External libraries or executables required by your Ray application, often installed via ``pip`` or ``conda``.

- **Local machine** and **Cluster**.  Usually, you may want to separate the Ray cluster compute machines/pods from the machine/pod that handles and submits the application. You can submit a Ray Job via :ref:`the Ray Job Submission mechanism <jobs-overview>`, or use `ray attach` to connect to a cluster interactively. We call the machine submitting the job your *local machine*.

- **Job**. A :ref:`Ray job <cluster-clients-and-jobs>` is a single application: it is the collection of Ray tasks, objects, and actors that originate from the same script.

.. _using-the-cluster-launcher:

Preparing an environment using the Ray Cluster launcher
-------------------------------------------------------

The first way to set up dependencies is to is to prepare a single environment across the cluster before starting the Ray runtime.

- You can build all your files and dependencies into a container image and specify this in your your :ref:`Cluster YAML Configuration <cluster-config>`.

- You can also install packages using ``setup_commands`` in the Ray Cluster configuration file (:ref:`reference <cluster-configuration-setup-commands>`); these commands will be run as each node joins the cluster.
  Note that for production settings, it is recommended to build any necessary packages into a container image instead.

- You can push local files to the cluster using ``ray rsync_up`` (:ref:`reference<ray-rsync>`).

.. _runtime-environments:

Runtime environments
--------------------

.. note::

    This feature requires a full installation of Ray using ``pip install "ray[default]"``. This feature is available starting with Ray 1.4.0 and is currently supported on macOS and Linux, with beta support on Windows.

The second way to set up dependencies is to install them dynamically while Ray is running.

A **runtime environment** describes the dependencies your Ray application needs to run, including :ref:`files, packages, environment variables, and more <runtime-environments-api-ref>`.
It is installed dynamically on the cluster at runtime and cached for future use (see :ref:`Caching and Garbage Collection <runtime-environments-caching>` for details about the lifecycle).

Runtime environments can be used on top of the prepared environment from :ref:`the Ray Cluster launcher <using-the-cluster-launcher>` if it was used.
For example, you can use the Cluster launcher to install a base set of packages, and then use runtime environments to install additional packages.
In contrast with the base cluster environment, a runtime environment will only be active for Ray processes.  (For example, if using a runtime environment specifying a ``pip`` package ``my_pkg``, the statement ``import my_pkg`` will fail if called outside of a Ray task, actor, or job.)

Runtime environments also allow you to set dependencies per-task, per-actor, and per-job on a long-running Ray cluster.

.. testcode::
  :hide:

  import ray
  ray.shutdown()

.. testcode::

    import ray

    runtime_env = {"pip": ["emoji"]}

    ray.init(runtime_env=runtime_env)

    @ray.remote
    def f():
      import emoji
      return emoji.emojize('Python is :thumbs_up:')

    print(ray.get(f.remote()))

.. testoutput::

    Python is 👍

A runtime environment can be described by a Python `dict`:

.. literalinclude:: /ray-core/doc_code/runtime_env_example.py
   :language: python
   :start-after: __runtime_env_pip_def_start__
   :end-before: __runtime_env_pip_def_end__

Alternatively, you can use :class:`ray.runtime_env.RuntimeEnv <ray.runtime_env.RuntimeEnv>`:

.. literalinclude:: /ray-core/doc_code/runtime_env_example.py
   :language: python
   :start-after: __strong_typed_api_runtime_env_pip_def_start__
   :end-before: __strong_typed_api_runtime_env_pip_def_end__

For more examples, jump to the :ref:`API Reference <runtime-environments-api-ref>`.


There are two primary scopes for which you can specify a runtime environment:

* :ref:`Per-Job <rte-per-job>`, and
* :ref:`Per-Task/Actor, within a job <rte-per-task-actor>`.

.. _rte-per-job:

Specifying a Runtime Environment Per-Job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can specify a runtime environment for your whole job, whether running a script directly on the cluster, using the :ref:`Ray Jobs API <jobs-overview>`, or submitting a :ref:`KubeRay RayJob <kuberay-rayjob-quickstart>`:

.. literalinclude:: /ray-core/doc_code/runtime_env_example.py
   :language: python
   :start-after: __ray_init_start__
   :end-before: __ray_init_end__

.. testcode::
    :skipif: True

    # Option 2: Using Ray Jobs API (Python SDK)
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient("http://<head-node-ip>:8265")
    job_id = client.submit_job(
        entrypoint="python my_ray_script.py",
        runtime_env=runtime_env,
    )

.. code-block:: bash

    # Option 3: Using Ray Jobs API (CLI). (Note: can use --runtime-env to pass a YAML file instead of an inline JSON string.)
    $ ray job submit --address="http://<head-node-ip>:8265" --runtime-env-json='{"working_dir": "/data/my_files", "pip": ["emoji"]}' -- python my_ray_script.py

.. code-block:: yaml

    # Option 4: Using KubeRay RayJob. You can specify the runtime environment in the RayJob YAML manifest.
    # [...]
    spec:
      runtimeEnvYAML: |
        pip:
          - requests==2.26.0
          - pendulum==2.1.2
        env_vars:
          KEY: "VALUE"

.. warning::

    Specifying the ``runtime_env`` argument in the ``submit_job`` or ``ray job submit`` call ensures the runtime environment is installed on the cluster before the entrypoint script is run.

    If ``runtime_env`` is specified from ``ray.init(runtime_env=...)``, the runtime env is only applied to all children Tasks and Actors, not the entrypoint script (Driver) itself.

    If ``runtime_env`` is specified by both ``ray job submit`` and ``ray.init``, the runtime environments are merged. See :ref:`Runtime Environment Specified by Both Job and Driver <runtime-environments-job-conflict>` for more details.

.. note::

  There are two options for when to install the runtime environment:

  1. As soon as the job starts (i.e., as soon as ``ray.init()`` is called), the dependencies are eagerly downloaded and installed.
  2. The dependencies are installed only when a task is invoked or an actor is created.

  The default is option 1. To change the behavior to option 2, add ``"eager_install": False`` to the ``config`` of ``runtime_env``.

.. _rte-per-task-actor:

Specifying a Runtime Environment Per-Task or Per-Actor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can specify different runtime environments per-actor or per-task using ``.options()`` or the ``@ray.remote`` decorator:

.. literalinclude:: /ray-core/doc_code/runtime_env_example.py
   :language: python
   :start-after: __per_task_per_actor_start__
   :end-before: __per_task_per_actor_end__

This allows you to have actors and tasks running in their own environments, independent of the surrounding environment. (The surrounding environment could be the job's runtime environment, or the system environment of the cluster.)

.. warning::

  Ray does not guarantee compatibility between tasks and actors with conflicting runtime environments.
  For example, if an actor whose runtime environment contains a ``pip`` package tries to communicate with an actor with a different version of that package, it can lead to unexpected behavior such as unpickling errors.

Common Workflows
^^^^^^^^^^^^^^^^

This section describes some common use cases for runtime environments. These use cases are not mutually exclusive; all of the options described below can be combined in a single runtime environment.

.. _workflow-local-files:

Using Local Files
"""""""""""""""""

Your Ray application might depend on source files or data files.
For a development workflow, these might live on your local machine, but when it comes time to run things at scale, you will need to get them to your remote cluster.

The following simple example explains how to get your local files on the cluster.

.. testcode::
  :hide:

  import ray
  ray.shutdown()

.. testcode::

  import os
  import ray

  os.makedirs("/tmp/runtime_env_working_dir", exist_ok=True)
  with open("/tmp/runtime_env_working_dir/hello.txt", "w") as hello_file:
    hello_file.write("Hello World!")

  # Specify a runtime environment for the entire Ray job
  ray.init(runtime_env={"working_dir": "/tmp/runtime_env_working_dir"})

  # Create a Ray task, which inherits the above runtime env.
  @ray.remote
  def f():
      # The function will have its working directory changed to its node's
      # local copy of /tmp/runtime_env_working_dir.
      return open("hello.txt").read()

  print(ray.get(f.remote()))

.. testoutput::

  Hello World!

.. note::
  The example above is written to run on a local machine, but as for all of these examples, it also works when specifying a Ray cluster to connect to
  (e.g., using ``ray.init("ray://123.456.7.89:10001", runtime_env=...)`` or ``ray.init(address="auto", runtime_env=...)``).

The specified local directory will automatically be pushed to the cluster nodes when ``ray.init()`` is called.

You can also specify files via a remote cloud storage URI; see :ref:`remote-uris` for details.

If you specify a `working_dir`, Ray always prepares it first, and it's present in the creation of other runtime environments in the `${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}` environment variable. This sequencing allows `pip` and `conda` to reference local files in the `working_dir` like `requirements.txt` or `environment.yml`. See `pip` and `conda` sections in :ref:`runtime-environments-api-ref` for more details.

Using ``conda`` or ``pip`` packages
"""""""""""""""""""""""""""""""""""

Your Ray application might depend on Python packages (for example, ``pendulum`` or ``requests``) via ``import`` statements.

Ray ordinarily expects all imported packages to be preinstalled on every node of the cluster; in particular, these packages are not automatically shipped from your local machine to the cluster or downloaded from any repository.

However, using runtime environments you can dynamically specify packages to be automatically downloaded and installed in a virtual environment for your Ray job, or for specific Ray tasks or actors.

.. testcode::
  :hide:

  import ray
  ray.shutdown()

.. testcode::

  import ray
  import requests

  # This example runs on a local machine, but you can also do
  # ray.init(address=..., runtime_env=...) to connect to a cluster.
  ray.init(runtime_env={"pip": ["requests"]})

  @ray.remote
  def reqs():
      return requests.get("https://www.ray.io/").status_code

  print(ray.get(reqs.remote()))

.. testoutput::

  200


You may also specify your ``pip`` dependencies either via a Python list or a local ``requirements.txt`` file.
Consider specifying a ``requirements.txt`` file when your ``pip install`` command requires options such as ``--extra-index-url`` or ``--find-links``; see `<https://pip.pypa.io/en/stable/reference/requirements-file-format/#>`_ for details.
Alternatively, you can specify a ``conda`` environment, either as a Python dictionary or via a local ``environment.yml`` file.  This conda environment can include ``pip`` packages.
For details, head to the :ref:`API Reference <runtime-environments-api-ref>`.

.. warning::

  Since the packages in the ``runtime_env`` are installed at runtime, be cautious when specifying ``conda`` or ``pip`` packages whose installations involve building from source, as this can be slow.

.. note::

  When using the ``"pip"`` field, the specified packages will be installed "on top of" the base environment using ``virtualenv``, so existing packages on your cluster will still be importable.  By contrast, when using the ``conda`` field, your Ray tasks and actors will run in an isolated environment.  The ``conda`` and ``pip`` fields cannot both be used in a single ``runtime_env``.

.. note::

  The ``ray[default]`` package itself will automatically be installed in the environment.  For the ``conda`` field only, if you are using any other Ray libraries (for example, Ray Serve), then you will need to specify the library in the runtime environment (e.g. ``runtime_env = {"conda": {"dependencies": ["pytorch", "pip", {"pip": ["requests", "ray[serve]"]}]}}``.)

.. note::

  ``conda`` environments must have the same Python version as the Ray cluster.  Do not list ``ray`` in the ``conda`` dependencies, as it will be automatically installed.

Library Development
"""""""""""""""""""

Suppose you are developing a library ``my_module`` on Ray.

A typical iteration cycle will involve

1. Making some changes to the source code of ``my_module``
2. Running a Ray script to test the changes, perhaps on a distributed cluster.

To ensure your local changes show up across all Ray workers and can be imported properly, use the ``py_modules`` field.

.. testcode::
  :skipif: True

  import ray
  import my_module

  ray.init("ray://123.456.7.89:10001", runtime_env={"py_modules": [my_module]})

  @ray.remote
  def test_my_module():
      # No need to import my_module inside this function.
      my_module.test()

  ray.get(f.remote())

Note: This feature is currently limited to modules that are packages with a single directory containing an ``__init__.py`` file.  For single-file modules, you may use ``working_dir``.

.. _runtime-environments-api-ref:

API Reference
^^^^^^^^^^^^^

The ``runtime_env`` is a Python dictionary or a Python class :class:`ray.runtime_env.RuntimeEnv <ray.runtime_env.RuntimeEnv>` including one or more of the following fields:

- ``working_dir`` (str): Specifies the working directory for the Ray workers. This must either be (1) an local existing directory with total size at most 100 MiB, (2) a local existing zipped file with total unzipped size at most 100 MiB (Note: ``excludes`` has no effect), or (3) a URI to a remotely-stored zip file containing the working directory for your job (no file size limit is enforced by Ray). See :ref:`remote-uris` for details.
  The specified directory will be downloaded to each node on the cluster, and Ray workers will be started in their node's copy of this directory.

  - Examples

    - ``"."  # cwd``

    - ``"/src/my_project"``

    - ``"/src/my_project.zip"``

    - ``"s3://path/to/my_dir.zip"``

  Note: Setting a local directory per-task or per-actor is currently unsupported; it can only be set per-job (i.e., in ``ray.init()``).

  Note: If the local directory contains a ``.gitignore`` file, the files and paths specified there are not uploaded to the cluster.  You can disable this by setting the environment variable `RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1` on the machine doing the uploading.

- ``py_modules`` (List[str|module]): Specifies Python modules to be available for import in the Ray workers.  (For more ways to specify packages, see also the ``pip`` and ``conda`` fields below.)
  Each entry must be either (1) a path to a local directory, (2) a URI to a remote zip or wheel file (see :ref:`remote-uris` for details), (3) a Python module object, or (4) a path to a local `.whl` file.

  - Examples of entries in the list:

    - ``"."``

    - ``"/local_dependency/my_module"``

    - ``"s3://bucket/my_module.zip"``

    - ``my_module # Assumes my_module has already been imported, e.g. via 'import my_module'``

    - ``my_module.whl``

    - ``"s3://bucket/my_module.whl"``

  The modules will be downloaded to each node on the cluster.

  Note: Setting options (1), (3) and (4) per-task or per-actor is currently unsupported, it can only be set per-job (i.e., in ``ray.init()``).

  Note: For option (1), if the local directory contains a ``.gitignore`` file, the files and paths specified there are not uploaded to the cluster.  You can disable this by setting the environment variable `RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1` on the machine doing the uploading.

  Note: This feature is currently limited to modules that are packages with a single directory containing an ``__init__.py`` file.  For single-file modules, you may use ``working_dir``.

- ``excludes`` (List[str]): When used with ``working_dir`` or ``py_modules``, specifies a list of files or paths to exclude from being uploaded to the cluster.
  This field uses the pattern-matching syntax used by ``.gitignore`` files: see `<https://git-scm.com/docs/gitignore>`_ for details.
  Note: In accordance with ``.gitignore`` syntax, if there is a separator (``/``) at the beginning or middle (or both) of the pattern, then the pattern is interpreted relative to the level of the ``working_dir``.
  In particular, you shouldn't use absolute paths (e.g. `/Users/my_working_dir/subdir/`) with `excludes`; rather, you should use the relative path `/subdir/` (written here with a leading `/` to match only the top-level `subdir` directory, rather than all directories named `subdir` at all levels.)

  - Example: ``{"working_dir": "/Users/my_working_dir/", "excludes": ["my_file.txt", "/subdir/, "path/to/dir", "*.log"]}``

- ``pip`` (dict | List[str] | str): Either (1) a list of pip `requirements specifiers <https://pip.pypa.io/en/stable/cli/pip_install/#requirement-specifiers>`_, (2) a string containing the path to a local pip
  `“requirements.txt” <https://pip.pypa.io/en/stable/user_guide/#requirements-files>`_ file, or (3) a python dictionary that has three fields: (a) ``packages`` (required, List[str]): a list of pip packages,
  (b) ``pip_check`` (optional, bool): whether to enable `pip check <https://pip.pypa.io/en/stable/cli/pip_check/>`_ at the end of pip install, defaults to ``False``.
  (c) ``pip_version`` (optional, str): the version of pip; Ray will spell the package name "pip" in front of the ``pip_version`` to form the final requirement string.
  The syntax of a requirement specifier is defined in full in `PEP 508 <https://www.python.org/dev/peps/pep-0508/>`_.
  This will be installed in the Ray workers at runtime.  Packages in the preinstalled cluster environment will still be available.
  To use a library like Ray Serve or Ray Tune, you will need to include ``"ray[serve]"`` or ``"ray[tune]"`` here.
  The Ray version must match that of the cluster.

  - Example: ``["requests==1.0.0", "aiohttp", "ray[serve]"]``

  - Example: ``"./requirements.txt"``

  - Example: ``{"packages":["tensorflow", "requests"], "pip_check": False, "pip_version": "==22.0.2;python_version=='3.8.11'"}``

  When specifying a path to a ``requirements.txt`` file, the file must be present on your local machine and it must be a valid absolute path or relative filepath relative to your local current working directory, *not* relative to the `working_dir` specified in the `runtime_env`.
  Furthermore, referencing local files *within* a `requirements.txt` file isn't directly supported (e.g., ``-r ./my-laptop/more-requirements.txt``, ``./my-pkg.whl``). Instead, use the `${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}` environment variable in the creation process. For example, use `-r ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/my-laptop/more-requirements.txt` or `${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/my-pkg.whl` to reference local files, while ensuring they're in the `working_dir`.

- ``conda`` (dict | str): Either (1) a dict representing the conda environment YAML, (2) a string containing the path to a local
  `conda “environment.yml” <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually>`_ file,
  or (3) the name of a local conda environment already installed on each node in your cluster (e.g., ``"pytorch_p36"``).
  In the first two cases, the Ray and Python dependencies will be automatically injected into the environment to ensure compatibility, so there is no need to manually include them.
  The Python and Ray version must match that of the cluster, so you likely should not specify them manually.
  Note that the ``conda`` and ``pip`` keys of ``runtime_env`` cannot both be specified at the same time---to use them together, please use ``conda`` and add your pip dependencies in the ``"pip"`` field in your conda ``environment.yaml``.

  - Example: ``{"dependencies": ["pytorch", "torchvision", "pip", {"pip": ["pendulum"]}]}``

  - Example: ``"./environment.yml"``

  - Example: ``"pytorch_p36"``

  When specifying a path to a ``environment.yml`` file, the file must be present on your local machine and it must be a valid absolute path or a relative filepath relative to your local current working directory, *not* relative to the `working_dir` specified in the `runtime_env`.
  Furthermore, referencing local files *within* a `environment.yml` file isn't directly supported (e.g., ``-r ./my-laptop/more-requirements.txt``, ``./my-pkg.whl``). Instead, use the `${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}` environment variable in the creation process. For example, use `-r ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/my-laptop/more-requirements.txt` or `${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/my-pkg.whl` to reference local files, while ensuring they're in the `working_dir`.

- ``env_vars`` (Dict[str, str]): Environment variables to set.  Environment variables already set on the cluster will still be visible to the Ray workers; so there is
  no need to include ``os.environ`` or similar in the ``env_vars`` field.
  By default, these environment variables override the same name environment variables on the cluster.
  You can also reference existing environment variables using ${ENV_VAR} to achieve the appending behavior.
  If the environment variable doesn't exist, it becomes an empty string `""`.

  - Example: ``{"OMP_NUM_THREADS": "32", "TF_WARNINGS": "none"}``

  - Example: ``{"LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}:/home/admin/my_lib"}``

  - Non-existant variable example: ``{"ENV_VAR_NOT_EXIST": "${ENV_VAR_NOT_EXIST}:/home/admin/my_lib"}`` -> ``ENV_VAR_NOT_EXIST=":/home/admin/my_lib"``.

- ``nsight`` (Union[str, Dict[str, str]]): specifies the config for the Nsight System Profiler. The value is either (1) "default", which refers to the `default config <https://github.com/ray-project/ray/blob/master/python/ray/_private/runtime_env/nsight.py#L20>`_, or (2) a dict of Nsight System Profiler options and their values.
  See :ref:`here <profiling-nsight-profiler>` for more details on setup and usage.

  - Example: ``"default"``

  - Example: ``{"stop-on-exit": "true", "t": "cuda,cublas,cudnn", "ftrace": ""}``

- ``container`` (dict): Require a given (Docker) image, and the worker process will run in a container with this image.
  The `worker_path` is the default_worker.py path. It is required only if ray installation directory in the container is different from raylet host.
  The `run_options` list spec is `here <https://docs.docker.com/engine/reference/run/>`__.

  - Example: ``{"image": "anyscale/ray-ml:nightly-py38-cpu", "worker_path": "/root/python/ray/workers/default_worker.py", "run_options": ["--cap-drop SYS_ADMIN","--log-level=debug"]}``

  Note: ``container`` is experimental now. If you have some requirements or run into any problems, raise issues in `github <https://github.com/ray-project/ray/issues>`__.

- ``config`` (dict | :class:`ray.runtime_env.RuntimeEnvConfig <ray.runtime_env.RuntimeEnvConfig>`): config for runtime environment. Either a dict or a RuntimeEnvConfig.
  Fields:
  (1) setup_timeout_seconds, the timeout of runtime environment creation, timeout is in seconds.

  - Example: ``{"setup_timeout_seconds": 10}``

  - Example: ``RuntimeEnvConfig(setup_timeout_seconds=10)``

  (2) ``eager_install`` (bool): Indicates whether to install the runtime environment on the cluster at ``ray.init()`` time, before the workers are leased. This flag is set to ``True`` by default.
  If set to ``False``, the runtime environment will be only installed when the first task is invoked or when the first actor is created.
  Currently, specifying this option per-actor or per-task is not supported.

  - Example: ``{"eager_install": False}``

  - Example: ``RuntimeEnvConfig(eager_install=False)``

.. _runtime-environments-caching:

Caching and Garbage Collection
""""""""""""""""""""""""""""""
Runtime environment resources on each node (such as conda environments, pip packages, or downloaded ``working_dir`` or ``py_modules`` directories) will be cached on the cluster to enable quick reuse across different runtime environments within a job.  Each field (``working_dir``, ``py_modules``, etc.) has its own cache whose size defaults to 10 GB.  To change this default, you may set the environment variable ``RAY_RUNTIME_ENV_<field>_CACHE_SIZE_GB`` on each node in your cluster before starting Ray e.g. ``export RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB=1.5``.

When the cache size limit is exceeded, resources not currently used by any Actor, Task or Job are deleted.

.. _runtime-environments-job-conflict:

Runtime Environment Specified by Both Job and Driver
""""""""""""""""""""""""""""""""""""""""""""""""""""

When running an entrypoint script (Driver), the runtime environment can be specified via `ray.init(runtime_env=...)` or `ray job submit --runtime-env` (See :ref:`Specifying a Runtime Environment Per-Job <rte-per-job>` for more details).

- If the runtime environment is specified by ``ray job submit --runtime-env=...``, the runtime environments are applied to the entrypoint script (Driver) and all the tasks and actors created from it.
- If the runtime environment is specified by ``ray.init(runtime_env=...)``, the runtime environments are applied to all the tasks and actors, but not the entrypoint script (Driver) itself.

Since ``ray job submit`` submits a Driver (that calls ``ray.init``), sometimes runtime environments are specified by both of them. When both the Ray Job and Driver specify runtime environments, their runtime environments are merged if there's no conflict.
It means the driver script uses the runtime environment specified by `ray job submit`, and all the tasks and actors are going to use the merged runtime environment.
Ray raises an exception if the runtime environments conflict.

* The ``runtime_env["env_vars"]`` of `ray job submit --runtime-env=...` is merged with the ``runtime_env["env_vars"]`` of `ray.init(runtime_env=...)`.
  Note that each individual env_var keys are merged.
  If the environment variables conflict, Ray raises an exception.
* Every other field in the ``runtime_env`` will be merged. If any key conflicts, it raises an exception.

Example:

.. testcode::

  # `ray job submit --runtime_env=...`
  {"pip": ["requests", "chess"],
  "env_vars": {"A": "a", "B": "b"}}

  # ray.init(runtime_env=...)
  {"env_vars": {"C": "c"}}

  # Driver's actual `runtime_env` (merged with Job's)
  {"pip": ["requests", "chess"],
  "env_vars": {"A": "a", "B": "b", "C": "c"}}

Conflict Example:

.. testcode::

  # Example 1, env_vars conflicts
  # `ray job submit --runtime_env=...`
  {"pip": ["requests", "chess"],
  "env_vars": {"C": "a", "B": "b"}}

  # ray.init(runtime_env=...)
  {"env_vars": {"C": "c"}}

  # Ray raises an exception because the "C" env var conflicts.

  # Example 2, other field (e.g., pip) conflicts
  # `ray job submit --runtime_env=...`
  {"pip": ["requests", "chess"]}

  # ray.init(runtime_env=...)
  {"pip": ["torch"]}

  # Ray raises an exception because "pip" conflicts.

You can set an environment variable `RAY_OVERRIDE_JOB_RUNTIME_ENV=1`
to avoid raising an exception upon a conflict. In this case, the runtime environments
are inherited in the same way as :ref:`Driver and Task and Actor both specify
runtime environments <runtime-environments-inheritance>`, where ``ray job submit``
is a parent and ``ray.init`` is a child.

.. _runtime-environments-inheritance:

Inheritance
"""""""""""

.. _runtime-env-driver-to-task-inheritance:

The runtime environment is inheritable, so it applies to all Tasks and Actors within a Job and all child Tasks and Actors of a Task or Actor once set, unless it is overridden.

If an Actor or Task specifies a new ``runtime_env``, it overrides the parent’s ``runtime_env`` (i.e., the parent Actor's or Task's ``runtime_env``, or the Job's ``runtime_env`` if Actor or Task doesn't have a parent) as follows:

* The ``runtime_env["env_vars"]`` field will be merged with the ``runtime_env["env_vars"]`` field of the parent.
  This allows for environment variables set in the parent's runtime environment to be automatically propagated to the child, even if new environment variables are set in the child's runtime environment.
* Every other field in the ``runtime_env`` will be *overridden* by the child, not merged.  For example, if ``runtime_env["py_modules"]`` is specified, it will replace the ``runtime_env["py_modules"]`` field of the parent.

Example:

.. testcode::

  # Parent's `runtime_env`
  {"pip": ["requests", "chess"],
  "env_vars": {"A": "a", "B": "b"}}

  # Child's specified `runtime_env`
  {"pip": ["torch", "ray[serve]"],
  "env_vars": {"B": "new", "C": "c"}}

  # Child's actual `runtime_env` (merged with parent's)
  {"pip": ["torch", "ray[serve]"],
  "env_vars": {"A": "a", "B": "new", "C": "c"}}

.. _runtime-env-faq:

Frequently Asked Questions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Are environments installed on every node?
"""""""""""""""""""""""""""""""""""""""""

If a runtime environment is specified in ``ray.init(runtime_env=...)``, then the environment will be installed on every node.  See :ref:`Per-Job <rte-per-job>` for more details.
(Note, by default the runtime environment will be installed eagerly on every node in the cluster. If you want to lazily install the runtime environment on demand, set the ``eager_install`` option to false: ``ray.init(runtime_env={..., "config": {"eager_install": False}}``.)

When is the environment installed?
""""""""""""""""""""""""""""""""""

When specified per-job, the environment is installed when you call ``ray.init()`` (unless ``"eager_install": False`` is set).
When specified per-task or per-actor, the environment is installed when the task is invoked or the actor is instantiated (i.e. when you call ``my_task.remote()`` or ``my_actor.remote()``.)
See :ref:`Per-Job <rte-per-job>` :ref:`Per-Task/Actor, within a job <rte-per-task-actor>` for more details.

Where are the environments cached?
""""""""""""""""""""""""""""""""""

Any local files downloaded by the environments are cached at ``/tmp/ray/session_latest/runtime_resources``.

How long does it take to install or to load from cache?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

The install time usually mostly consists of the time it takes to run ``pip install`` or ``conda create`` / ``conda activate``, or to upload/download a ``working_dir``, depending on which ``runtime_env`` options you're using.
This could take seconds or minutes.

On the other hand, loading a runtime environment from the cache should be nearly as fast as the ordinary Ray worker startup time, which is on the order of a few seconds. A new Ray worker is started for every Ray actor or task that requires a new runtime environment.
(Note that loading a cached ``conda`` environment could still be slow, since the ``conda activate`` command sometimes takes a few seconds.)

You can set ``setup_timeout_seconds`` config to avoid the installation hanging for a long time. If the installation is not finished within this time, your tasks or actors will fail to start.

What is the relationship between runtime environments and Docker?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

They can be used independently or together.
A container image can be specified in the :ref:`Cluster Launcher <vm-cluster-quick-start>` for large or static dependencies, and runtime environments can be specified per-job or per-task/actor for more dynamic use cases.
The runtime environment will inherit packages, files, and environment variables from the container image.

My ``runtime_env`` was installed, but when I log into the node I can't import the packages.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The runtime environment is only active for the Ray worker processes; it does not install any packages "globally" on the node.

.. _remote-uris:

Remote URIs
-----------

The ``working_dir`` and ``py_modules`` arguments in the ``runtime_env`` dictionary can specify either local path(s) or remote URI(s).

A local path must be a directory path. The directory's contents will be directly accessed as the ``working_dir`` or a ``py_module``.
A remote URI must be a link directly to a zip file or a wheel file (only for ``py_module``). **The zip file must contain only a single top-level directory.**
The contents of this directory will be directly accessed as the ``working_dir`` or a ``py_module``.

For example, suppose you want to use the contents in your local ``/some_path/example_dir`` directory as your ``working_dir``.
If you want to specify this directory as a local path, your ``runtime_env`` dictionary should contain:

.. testcode::
  :skipif: True

  runtime_env = {..., "working_dir": "/some_path/example_dir", ...}

Suppose instead you want to host your files in your ``/some_path/example_dir`` directory remotely and provide a remote URI.
You would need to first compress the ``example_dir`` directory into a zip file.

There should be no other files or directories at the top level of the zip file, other than ``example_dir``.
You can use the following command in the Terminal to do this:

.. code-block:: bash

    cd /some_path
    zip -r zip_file_name.zip example_dir

Note that this command must be run from the *parent directory* of the desired ``working_dir`` to ensure that the resulting zip file contains a single top-level directory.
In general, the zip file's name and the top-level directory's name can be anything.
The top-level directory's contents will be used as the ``working_dir`` (or ``py_module``).

You can check that the zip file contains a single top-level directory by running the following command in the Terminal:

.. code-block:: bash

  zipinfo -1 zip_file_name.zip
  # example_dir/
  # example_dir/my_file_1.txt
  # example_dir/subdir/my_file_2.txt

Suppose you upload the compressed ``example_dir`` directory to AWS S3 at the S3 URI ``s3://example_bucket/example.zip``.
Your ``runtime_env`` dictionary should contain:

.. testcode::
  :skipif: True

  runtime_env = {..., "working_dir": "s3://example_bucket/example.zip", ...}

.. warning::

  Check for hidden files and metadata directories in zipped dependencies.
  You can inspect a zip file's contents by running the ``zipinfo -1 zip_file_name.zip`` command in the Terminal.
  Some zipping methods can cause hidden files or metadata directories to appear in the zip file at the top level.
  To avoid this, use the ``zip -r`` command directly on the directory you want to compress from its parent's directory. For example, if you have a directory structure such as: ``a/b`` and you what to compress ``b``, issue the ``zip -r b`` command from the directory ``a.``
  If Ray detects more than a single directory at the top level, it will use the entire zip file instead of the top-level directory, which may lead to unexpected behavior.

Currently, three types of remote URIs are supported for hosting ``working_dir`` and ``py_modules`` packages:

- ``HTTPS``: ``HTTPS`` refers to URLs that start with ``https``.
  These are particularly useful because remote Git providers (e.g. GitHub, Bitbucket, GitLab, etc.) use ``https`` URLs as download links for repository archives.
  This allows you to host your dependencies on remote Git providers, push updates to them, and specify which dependency versions (i.e. commits) your jobs should use.
  To use packages via ``HTTPS`` URIs, you must have the ``smart_open`` library (you can install it using ``pip install smart_open``).

  - Example:

    - ``runtime_env = {"working_dir": "https://github.com/example_username/example_respository/archive/HEAD.zip"}``

- ``S3``: ``S3`` refers to URIs starting with ``s3://`` that point to compressed packages stored in `AWS S3 <https://aws.amazon.com/s3/>`_.
  To use packages via ``S3`` URIs, you must have the ``smart_open`` and ``boto3`` libraries (you can install them using ``pip install smart_open`` and ``pip install boto3``).
  Ray does not explicitly pass in any credentials to ``boto3`` for authentication.
  ``boto3`` will use your environment variables, shared credentials file, and/or AWS config file to authenticate access.
  See the `AWS boto3 documentation <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html>`_ to learn how to configure these.

  - Example:

    - ``runtime_env = {"working_dir": "s3://example_bucket/example_file.zip"}``

- ``GS``: ``GS`` refers to URIs starting with ``gs://`` that point to compressed packages stored in `Google Cloud Storage <https://cloud.google.com/storage>`_.
  To use packages via ``GS`` URIs, you must have the ``smart_open`` and ``google-cloud-storage`` libraries (you can install them using ``pip install smart_open`` and ``pip install google-cloud-storage``).
  Ray does not explicitly pass in any credentials to the ``google-cloud-storage``'s ``Client`` object.
  ``google-cloud-storage`` will use your local service account key(s) and environment variables by default.
  Follow the steps on Google Cloud Storage's `Getting started with authentication <https://cloud.google.com/docs/authentication/getting-started>`_ guide to set up your credentials, which allow Ray to access your remote package.

  - Example:

    - ``runtime_env = {"working_dir": "gs://example_bucket/example_file.zip"}``

Note that the ``smart_open``, ``boto3``, and ``google-cloud-storage`` packages are not installed by default, and it is not sufficient to specify them in the ``pip`` section of your ``runtime_env``.
The relevant packages must already be installed on all nodes of the cluster when Ray starts.

Hosting a Dependency on a Remote Git Provider: Step-by-Step Guide
-----------------------------------------------------------------

You can store your dependencies in repositories on a remote Git provider (e.g. GitHub, Bitbucket, GitLab, etc.), and you can periodically push changes to keep them updated.
In this section, you will learn how to store a dependency on GitHub and use it in your runtime environment.

.. note::
  These steps will also be useful if you use another large, remote Git provider (e.g. BitBucket, GitLab, etc.).
  For simplicity, this section refers to GitHub alone, but you can follow along on your provider.

First, create a repository on GitHub to store your ``working_dir`` contents or your ``py_module`` dependency.
By default, when you download a zip file of your repository, the zip file will already contain a single top-level directory that holds the repository contents,
so you can directly upload your ``working_dir`` contents or your ``py_module`` dependency to the GitHub repository.

Once you have uploaded your ``working_dir`` contents or your ``py_module`` dependency, you need the HTTPS URL of the repository zip file, so you can specify it in your ``runtime_env`` dictionary.

You have two options to get the HTTPS URL.

Option 1: Download Zip (quicker to implement, but not recommended for production environments)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first option is to use the remote Git provider's "Download Zip" feature, which provides an HTTPS link that zips and downloads your repository.
This is quick, but it is **not recommended** because it only allows you to download a zip file of a repository branch's latest commit.
To find a GitHub URL, navigate to your repository on `GitHub <https://github.com/>`_, choose a branch, and click on the green "Code" drop down button:

.. figure:: images/ray_repo.png
   :width: 500px

This will drop down a menu that provides three options: "Clone" which provides HTTPS/SSH links to clone the repository,
"Open with GitHub Desktop", and "Download ZIP."
Right-click on "Download Zip."
This will open a pop-up near your cursor. Select "Copy Link Address":

.. figure:: images/download_zip_url.png
   :width: 300px

Now your HTTPS link is copied to your clipboard. You can paste it into your ``runtime_env`` dictionary.

.. warning::

  Using the HTTPS URL from your Git provider's "Download as Zip" feature is not recommended if the URL always points to the latest commit.
  For instance, using this method on GitHub generates a link that always points to the latest commit on the chosen branch.

  By specifying this link in the ``runtime_env`` dictionary, your Ray Cluster always uses the chosen branch's latest commit.
  This creates a consistency risk: if you push an update to your remote Git repository while your cluster's nodes are pulling the repository's contents,
  some nodes may pull the version of your package just before you pushed, and some nodes may pull the version just after.
  For consistency, it is better to specify a particular commit, so all the nodes use the same package.
  See "Option 2: Manually Create URL" to create a URL pointing to a specific commit.

Option 2: Manually Create URL (slower to implement, but recommended for production environments)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second option is to manually create this URL by pattern-matching your specific use case with one of the following examples.
**This is recommended** because it provides finer-grained control over which repository branch and commit to use when generating your dependency zip file.
These options prevent consistency issues on Ray Clusters (see the warning above for more info).
To create the URL, pick a URL template below that fits your use case, and fill in all parameters in brackets (e.g. [username], [repository], etc.) with the specific values from your repository.
For instance, suppose your GitHub username is ``example_user``, the repository's name is ``example_repository``, and the desired commit hash is ``abcdefg``.
If ``example_repository`` is public and you want to retrieve the ``abcdefg`` commit (which matches the first example use case), the URL would be:

.. testcode::

    runtime_env = {"working_dir": ("https://github.com"
                                   "/example_user/example_repository/archive/abcdefg.zip")}

Here is a list of different use cases and corresponding URLs:

- Example: Retrieve package from a specific commit hash on a public GitHub repository

.. testcode::

    runtime_env = {"working_dir": ("https://github.com"
                                   "/[username]/[repository]/archive/[commit hash].zip")}

- Example: Retrieve package from a private GitHub repository using a Personal Access Token **during development**. **For production** see :ref:`this document <runtime-env-auth>` to learn how to authenticate private dependencies safely.

.. testcode::

    runtime_env = {"working_dir": ("https://[username]:[personal access token]@github.com"
                                   "/[username]/[private repository]/archive/[commit hash].zip")}

- Example: Retrieve package from a public GitHub repository's latest commit

.. testcode::

    runtime_env = {"working_dir": ("https://github.com"
                                   "/[username]/[repository]/archive/HEAD.zip")}

- Example: Retrieve package from a specific commit hash on a public Bitbucket repository

.. testcode::

    runtime_env = {"working_dir": ("https://bitbucket.org"
                                   "/[owner]/[repository]/get/[commit hash].tar.gz")}

.. tip::

  It is recommended to specify a particular commit instead of always using the latest commit.
  This prevents consistency issues on a multi-node Ray Cluster.
  See the warning below "Option 1: Download Zip" for more info.

Once you have specified the URL in your ``runtime_env`` dictionary, you can pass the dictionary
into a ``ray.init()`` or ``.options()`` call. Congratulations! You have now hosted a ``runtime_env`` dependency
remotely on GitHub!


Debugging
---------
If runtime_env cannot be set up (e.g., network issues, download failures, etc.), Ray will fail to schedule tasks/actors
that require the runtime_env. If you call ``ray.get``, it will raise ``RuntimeEnvSetupError`` with
the error message in detail.

.. testcode::

    import ray
    import time

    @ray.remote
    def f():
        pass

    @ray.remote
    class A:
        def f(self):
            pass

    start = time.time()
    bad_env = {"conda": {"dependencies": ["this_doesnt_exist"]}}

    # [Tasks] will raise `RuntimeEnvSetupError`.
    try:
      ray.get(f.options(runtime_env=bad_env).remote())
    except ray.exceptions.RuntimeEnvSetupError:
      print("Task fails with RuntimeEnvSetupError")

    # [Actors] will raise `RuntimeEnvSetupError`.
    a = A.options(runtime_env=bad_env).remote()
    try:
      ray.get(a.f.remote())
    except ray.exceptions.RuntimeEnvSetupError:
      print("Actor fails with RuntimeEnvSetupError")

.. testoutput::

  Task fails with RuntimeEnvSetupError
  Actor fails with RuntimeEnvSetupError


Full logs can always be found in the file ``runtime_env_setup-[job_id].log`` for per-actor, per-task and per-job environments, or in
``runtime_env_setup-ray_client_server_[port].log`` for per-job environments when using Ray Client.

You can also enable ``runtime_env`` debugging log streaming by setting an environment variable ``RAY_RUNTIME_ENV_LOG_TO_DRIVER_ENABLED=1`` on each node before starting Ray, for example using ``setup_commands`` in the Ray Cluster configuration file (:ref:`reference <cluster-configuration-setup-commands>`).
This will print the full ``runtime_env`` setup log messages to the driver (the script that calls ``ray.init()``).

Example log output:

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

  ray.init(runtime_env={"pip": ["requests"]})

.. testoutput::
    :options: +MOCK

    (pid=runtime_env) 2022-02-28 14:12:33,653       INFO pip.py:188 -- Creating virtualenv at /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv, current python dir /Users/user/anaconda3/envs/ray-py38
    (pid=runtime_env) 2022-02-28 14:12:33,653       INFO utils.py:76 -- Run cmd[1] ['/Users/user/anaconda3/envs/ray-py38/bin/python', '-m', 'virtualenv', '--app-data', '/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv_app_data', '--reset-app-data', '--no-periodic-update', '--system-site-packages', '--no-download', '/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv']
    (pid=runtime_env) 2022-02-28 14:12:34,267       INFO utils.py:97 -- Output of cmd[1]: created virtual environment CPython3.8.11.final.0-64 in 473ms
    (pid=runtime_env)   creator CPython3Posix(dest=/private/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv, clear=False, no_vcs_ignore=False, global=True)
    (pid=runtime_env)   seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/private/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv_app_data)
    (pid=runtime_env)     added seed packages: pip==22.0.3, setuptools==60.6.0, wheel==0.37.1
    (pid=runtime_env)   activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator
    (pid=runtime_env)
    (pid=runtime_env) 2022-02-28 14:12:34,268       INFO utils.py:76 -- Run cmd[2] ['/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv/bin/python', '-c', 'import ray; print(ray.__version__, ray.__path__[0])']
    (pid=runtime_env) 2022-02-28 14:12:35,118       INFO utils.py:97 -- Output of cmd[2]: 3.0.0.dev0 /Users/user/ray/python/ray
    (pid=runtime_env)
    (pid=runtime_env) 2022-02-28 14:12:35,120       INFO pip.py:236 -- Installing python requirements to /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv
    (pid=runtime_env) 2022-02-28 14:12:35,122       INFO utils.py:76 -- Run cmd[3] ['/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv/bin/python', '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '-r', '/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt']
    (pid=runtime_env) 2022-02-28 14:12:38,000       INFO utils.py:97 -- Output of cmd[3]: Requirement already satisfied: requests in /Users/user/anaconda3/envs/ray-py38/lib/python3.8/site-packages (from -r /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt (line 1)) (2.26.0)
    (pid=runtime_env) Requirement already satisfied: idna<4,>=2.5 in /Users/user/anaconda3/envs/ray-py38/lib/python3.8/site-packages (from requests->-r /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt (line 1)) (3.2)
    (pid=runtime_env) Requirement already satisfied: certifi>=2017.4.17 in /Users/user/anaconda3/envs/ray-py38/lib/python3.8/site-packages (from requests->-r /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt (line 1)) (2021.10.8)
    (pid=runtime_env) Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/user/anaconda3/envs/ray-py38/lib/python3.8/site-packages (from requests->-r /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt (line 1)) (1.26.7)
    (pid=runtime_env) Requirement already satisfied: charset-normalizer~=2.0.0 in /Users/user/anaconda3/envs/ray-py38/lib/python3.8/site-packages (from requests->-r /tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/requirements.txt (line 1)) (2.0.6)
    (pid=runtime_env)
    (pid=runtime_env) 2022-02-28 14:12:38,001       INFO utils.py:76 -- Run cmd[4] ['/tmp/ray/session_2022-02-28_14-12-29_909064_87908/runtime_resources/pip/0cc818a054853c3841171109300436cad4dcf594/virtualenv/bin/python', '-c', 'import ray; print(ray.__version__, ray.__path__[0])']
    (pid=runtime_env) 2022-02-28 14:12:38,804       INFO utils.py:97 -- Output of cmd[4]: 3.0.0.dev0 /Users/user/ray/python/ray

See :ref:`Logging Directory Structure <logging-directory-structure>` for more details.


Tips for first-time users
=========================

Ray provides a highly flexible, yet minimalist and easy to use API.
On this page, we describe several tips that can help first-time Ray users to avoid some
common mistakes that can significantly hurt the performance of their programs.
For an in-depth treatment of advanced design patterns, please read :ref:`core design patterns <core-patterns>`.

.. list-table:: The core Ray API we use in this document.
   :header-rows: 1

   * - API
     - Description
   * - ``ray.init()``
     - Initialize Ray context.
   * - ``@ray.remote``
     - | Function or class decorator specifying that the function will be
       | executed as a task or the class as an actor in a different process.
   * - ``.remote()``
     - | Postfix to every remote function, remote class declaration, or
       | invocation of a remote class method.
       | Remote operations are asynchronous.
   * - ``ray.put()``
     - | Store object in object store, and return its ID.
       | This ID can be used to pass object as an argument
       | to any remote function or method call.
       | This is a synchronous operation.
   * - ``ray.get()``
     - | Return an object or list of objects from the object ID
       | or list of object IDs.
       | This is a synchronous (i.e., blocking) operation.
   * - ``ray.wait()``
     - | From a list of object IDs, returns
       | (1) the list of IDs of the objects that are ready, and
       | (2) the list of IDs of the objects that are not ready yet.
       | By default, it returns one ready object ID at a time.


All the results reported in this page were obtained on a 13-inch MacBook Pro with a 2.7 GHz Core i7 CPU and 16GB of RAM.
While ``ray.init()`` automatically detects the number of cores when it runs on a single machine,
to reduce the variability of the results you observe on your machine when running the code below,
here we specify num_cpus = 4, i.e., a machine with 4 CPUs.

Since each task requests by default one CPU, this setting allows us to execute up to four tasks in parallel.
As a result, our Ray system consists of one driver executing the program,
and up to four workers running remote tasks or actors.

.. _tip-delay-get:

Tip 1: Delay ray.get()
----------------------

With Ray, the invocation of every remote operation (e.g., task, actor method) is asynchronous. This means that the operation immediately returns a promise/future, which is essentially an identifier (ID) of the operation’s result. This is key to achieving parallelism, as it allows the driver program to launch multiple operations in parallel. To get the actual results, the programmer needs to call ``ray.get()`` on the IDs of the results. This call blocks until the results are available. As a side effect, this operation also blocks the driver program from invoking other operations, which can hurt parallelism.

Unfortunately, it is quite natural for a new Ray user to inadvertently use ``ray.get()``. To illustrate this point, consider the following simple Python code which calls the ``do_some_work()`` function four times, where each invocation takes around 1 sec:

.. testcode::

    import ray
    import time

    def do_some_work(x):
        time.sleep(1) # Replace this with work you need to do.
        return x

    start = time.time()
    results = [do_some_work(x) for x in range(4)]
    print("duration =", time.time() - start)
    print("results =", results)


The output of a program execution is below. As expected, the program takes around 4 seconds:

.. testoutput::
    :options: +MOCK

    duration = 4.0149290561676025
    results = [0, 1, 2, 3]

Now, let’s parallelize the above program with Ray. Some first-time users will do this by just making the function remote, i.e.,

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import time
    import ray

    ray.init(num_cpus=4) # Specify this system has 4 CPUs.

    @ray.remote
    def do_some_work(x):
        time.sleep(1) # Replace this with work you need to do.
        return x

    start = time.time()
    results = [do_some_work.remote(x) for x in range(4)]
    print("duration =", time.time() - start)
    print("results =", results)

However, when executing the above program one gets:

.. testoutput::
    :options: +MOCK

    duration = 0.0003619194030761719
    results = [ObjectRef(df5a1a828c9685d3ffffffff0100000001000000), ObjectRef(cb230a572350ff44ffffffff0100000001000000), ObjectRef(7bbd90284b71e599ffffffff0100000001000000), ObjectRef(bd37d2621480fc7dffffffff0100000001000000)]

When looking at this output, two things jump out. First, the program finishes immediately, i.e., in less than 1 ms. Second, instead of the expected results (i.e., [0, 1, 2, 3]), we get a bunch of identifiers. Recall that remote operations are asynchronous and they return futures (i.e., object IDs) instead of the results themselves. This is exactly what we see here. We measure only the time it takes to invoke the tasks, not their running times, and we get the IDs of the results corresponding to the four tasks.

To get the actual results,  we need to use ray.get(), and here the first instinct is to just call ``ray.get()`` on the remote operation invocation, i.e., replace line 12 with:

.. testcode::

    results = [ray.get(do_some_work.remote(x)) for x in range(4)]

By re-running the program after this change we get:

.. testoutput::
    :options: +MOCK

    duration = 4.018050909042358
    results =  [0, 1, 2, 3]

So now the results are correct, but it still takes 4 seconds, so no speedup! What’s going on? The observant reader will already have the answer: ``ray.get()`` is blocking so calling it after each remote operation means that we wait for that operation to complete, which essentially means that we execute one operation at a time, hence no parallelism!

To enable parallelism, we need to call ``ray.get()`` after invoking all tasks. We can easily do so in our example by replacing line 12 with:

.. testcode::

    results = ray.get([do_some_work.remote(x) for x in range(4)])

By re-running the program after this change we now get:

.. testoutput::
    :options: +MOCK

    duration = 1.0064549446105957
    results =  [0, 1, 2, 3]

So finally, success! Our Ray program now runs in just 1 second which means that all invocations of ``do_some_work()`` are running in parallel.

In summary, always keep in mind that ``ray.get()`` is a blocking operation, and thus if called eagerly it can hurt the parallelism. Instead, you should try to write your program such that ``ray.get()`` is called as late as possible.

Tip 2: Avoid tiny tasks
-----------------------

When a first-time developer wants to parallelize their code with Ray, the natural instinct is to make every function or class remote. Unfortunately, this can lead to undesirable consequences; if the tasks are very small, the Ray program can take longer than the equivalent Python program.

Let’s consider again the above examples, but this time we make the tasks much shorter (i.e, each takes just 0.1ms), and dramatically increase the number of task invocations to 100,000.

.. testcode::

    import time

    def tiny_work(x):
        time.sleep(0.0001) # Replace this with work you need to do.
        return x

    start = time.time()
    results = [tiny_work(x) for x in range(100000)]
    print("duration =", time.time() - start)

By running this program we get:

.. testoutput::
    :options: +MOCK

    duration = 13.36544418334961

This result should be expected since the lower bound of executing 100,000 tasks that take 0.1ms each is 10s, to which we need to add other overheads such as function calls, etc.

Let’s now parallelize this code using Ray, by making every invocation of ``tiny_work()`` remote:

.. testcode::

    import time
    import ray

    @ray.remote
    def tiny_work(x):
        time.sleep(0.0001) # Replace this with work you need to do.
        return x

    start = time.time()
    result_ids = [tiny_work.remote(x) for x in range(100000)]
    results = ray.get(result_ids)
    print("duration =", time.time() - start)

The result of running this code is:

.. testoutput::
    :options: +MOCK

    duration = 27.46447515487671

Surprisingly, not only Ray didn’t improve the execution time, but the Ray program is actually slower than the sequential program! What’s going on? Well, the issue here is that every task invocation has a non-trivial overhead (e.g., scheduling, inter-process communication, updating the system state) and this overhead dominates the actual time it takes to execute the task.

One way to speed up this program is to make the remote tasks larger in order to amortize the invocation overhead. Here is one possible solution where we aggregate 1000 ``tiny_work()`` function calls in a single bigger remote function:

.. testcode::

    import time
    import ray

    def tiny_work(x):
        time.sleep(0.0001) # replace this is with work you need to do
        return x

    @ray.remote
    def mega_work(start, end):
        return [tiny_work(x) for x in range(start, end)]

    start = time.time()
    result_ids = []
    [result_ids.append(mega_work.remote(x*1000, (x+1)*1000)) for x in range(100)]
    results = ray.get(result_ids)
    print("duration =", time.time() - start)

Now, if we run the above program we get:

.. testoutput::
    :options: +MOCK

    duration = 3.2539820671081543

This is approximately one fourth of the sequential execution, in line with our expectations (recall, we can run four tasks in parallel). Of course, the natural question is how large is large enough for a task to amortize the remote invocation overhead. One way to find this is to run the following simple program to estimate the per-task invocation overhead:

.. testcode::

    @ray.remote
    def no_work(x):
        return x

    start = time.time()
    num_calls = 1000
    [ray.get(no_work.remote(x)) for x in range(num_calls)]
    print("per task overhead (ms) =", (time.time() - start)*1000/num_calls)

Running the above program on a 2018 MacBook Pro notebook shows:

.. testoutput::
    :options: +MOCK

    per task overhead (ms) = 0.4739549160003662

In other words, it takes almost half a millisecond to execute an empty task. This suggests that we will need to make sure a task takes at least a few milliseconds to amortize the invocation overhead. One caveat is that the per-task overhead will vary from machine to machine, and between tasks that run on the same machine versus remotely. This being said, making sure that tasks take at least a few milliseconds is a good rule of thumb when developing Ray programs.

Tip 3: Avoid passing same object repeatedly to remote tasks
-----------------------------------------------------------

When we pass a large object as an argument to a remote function, Ray calls ``ray.put()`` under the hood to store that object in the local object store. This can significantly improve the performance of a remote task invocation when the remote task is executed locally, as all local tasks share the object store.

However, there are cases when automatically calling ``ray.put()`` on a task invocation leads to performance issues. One example is passing the same large object as an argument repeatedly, as illustrated by the program below:

.. testcode::

    import time
    import numpy as np
    import ray

    @ray.remote
    def no_work(a):
        return

    start = time.time()
    a = np.zeros((5000, 5000))
    result_ids = [no_work.remote(a) for x in range(10)]
    results = ray.get(result_ids)
    print("duration =", time.time() - start)

This program outputs:

.. testoutput::
    :options: +MOCK

    duration = 1.0837509632110596


This running time is quite large for a program that calls just 10 remote tasks that do nothing. The reason for this unexpected high running time is that each time we invoke ``no_work(a)``, Ray calls ``ray.put(a)`` which results in copying array ``a`` to the object store. Since array ``a`` has 2.5 million entries, copying it takes a non-trivial time.

To avoid copying array ``a`` every time ``no_work()`` is invoked, one simple solution is to explicitly call ``ray.put(a)``, and then pass ``a``’s ID to ``no_work()``, as illustrated below:

.. testcode::
    :hide:

    import ray
    ray.shutdown()

.. testcode::

    import time
    import numpy as np
    import ray

    ray.init(num_cpus=4)

    @ray.remote
    def no_work(a):
        return

    start = time.time()
    a_id = ray.put(np.zeros((5000, 5000)))
    result_ids = [no_work.remote(a_id) for x in range(10)]
    results = ray.get(result_ids)
    print("duration =", time.time() - start)

Running this program takes only:

.. testoutput::
    :options: +MOCK

    duration = 0.132796049118042

This is 7 times faster than the original program which is to be expected since the main overhead of invoking ``no_work(a)`` was copying the array ``a`` to the object store, which now happens only once.

Arguably a more important advantage of avoiding multiple copies of the same object to the object store is that it precludes the object store filling up prematurely and incur the cost of object eviction.


Tip 4: Pipeline data processing
-------------------------------

If we use ``ray.get()`` on the results of multiple tasks we will have to wait until the last one of these tasks finishes. This can be an issue if tasks take widely different amounts of time.

To illustrate this issue, consider the following example where we run four ``do_some_work()`` tasks in parallel, with each task taking a time uniformly distributed between 0 and 4 seconds. Next, assume the results of these tasks are processed by ``process_results()``, which takes 1 sec per result. The expected running time is then (1) the time it takes to execute the slowest of the ``do_some_work()`` tasks, plus (2) 4 seconds which is the time it takes to execute ``process_results()``.

.. testcode::

    import time
    import random
    import ray

    @ray.remote
    def do_some_work(x):
        time.sleep(random.uniform(0, 4)) # Replace this with work you need to do.
        return x

    def process_results(results):
        sum = 0
        for x in results:
            time.sleep(1) # Replace this with some processing code.
            sum += x
        return sum

    start = time.time()
    data_list = ray.get([do_some_work.remote(x) for x in range(4)])
    sum = process_results(data_list)
    print("duration =", time.time() - start, "\nresult = ", sum)

The output of the program shows that it takes close to 8 sec to run:

.. testoutput::
    :options: +MOCK

    duration = 7.82636022567749
    result =  6

Waiting for the last task to finish when the others tasks might have finished much earlier unnecessarily increases the program running time. A better solution would be to process the data as soon it becomes available.
Fortunately, Ray allows you to do exactly this by calling ``ray.wait()`` on a list of object IDs. Without specifying any other parameters, this function returns as soon as an object in its argument list is ready. This call has two returns: (1) the ID of the ready object, and (2) the list containing the IDs of the objects not ready yet. The modified program is below. Note that one change we need to do is to replace ``process_results()`` with ``process_incremental()`` that processes one result at a time.

.. testcode::

    import time
    import random
    import ray

    @ray.remote
    def do_some_work(x):
        time.sleep(random.uniform(0, 4)) # Replace this with work you need to do.
        return x

    def process_incremental(sum, result):
        time.sleep(1) # Replace this with some processing code.
        return sum + result

    start = time.time()
    result_ids = [do_some_work.remote(x) for x in range(4)]
    sum = 0
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        sum = process_incremental(sum, ray.get(done_id[0]))
    print("duration =", time.time() - start, "\nresult = ", sum)

This program now takes just a bit over 4.8sec, a significant improvement:

.. testoutput::
    :options: +MOCK

    duration = 4.852453231811523
    result =  6

To aid the intuition, Figure 1 shows the execution timeline in both cases: when using ``ray.get()`` to wait for all results to become available before processing them, and using ``ray.wait()`` to start processing the results as soon as they become available.

.. figure:: /images/pipeline.png

    Figure 1: (a) Execution timeline when  using ray.get() to wait for all results from ``do_some_work()`` tasks before calling ``process_results()``. (b) Execution timeline when using ``ray.wait()`` to process results as soon as they become available.


:orphan:

.. _accelerator_types:

Accelerator Types
=================

Ray supports the following accelerator types:

.. literalinclude:: ../../../python/ray/util/accelerators/accelerators.py
    :language: python

.. _cross_language:

Cross-Language Programming
==========================

This page will show you how to use Ray's cross-language programming feature.

Setup the driver
-----------------

We need to set :ref:`code_search_path` in your driver.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/cross_language.py
            :language: python
            :start-after: __crosslang_init_start__
            :end-before: __crosslang_init_end__

    .. tab-item:: Java

        .. code-block:: bash

            java -classpath <classpath> \
                -Dray.address=<address> \
                -Dray.job.code-search-path=/path/to/code/ \
                <classname> <args>

You may want to include multiple directories to load both Python and Java code for workers, if they are placed in different directories.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/cross_language.py
            :language: python
            :start-after: __crosslang_multidir_start__
            :end-before: __crosslang_multidir_end__

    .. tab-item:: Java

        .. code-block:: bash

            java -classpath <classpath> \
                -Dray.address=<address> \
                -Dray.job.code-search-path=/path/to/jars:/path/to/pys \
                <classname> <args>

Python calling Java
-------------------

Suppose we have a Java static method and a Java class as follows:

.. code-block:: java

  package io.ray.demo;

  public class Math {

    public static int add(int a, int b) {
      return a + b;
    }
  }

.. code-block:: java

  package io.ray.demo;

  // A regular Java class.
  public class Counter {

    private int value = 0;

    public int increment() {
      this.value += 1;
      return this.value;
    }
  }

Then, in Python, we can call the above Java remote function, or create an actor
from the above Java class.

.. literalinclude:: ./doc_code/cross_language.py
  :language: python
  :start-after: __python_call_java_start__
  :end-before: __python_call_java_end__

Java calling Python
-------------------

Suppose we have a Python module as follows:

.. literalinclude:: ./doc_code/cross_language.py
  :language: python
  :start-after: __python_module_start__
  :end-before: __python_module_end__

.. note::

  * The function or class should be decorated by `@ray.remote`.

Then, in Java, we can call the above Python remote function, or create an actor
from the above Python class.

.. code-block:: java

  package io.ray.demo;

  import io.ray.api.ObjectRef;
  import io.ray.api.PyActorHandle;
  import io.ray.api.Ray;
  import io.ray.api.function.PyActorClass;
  import io.ray.api.function.PyActorMethod;
  import io.ray.api.function.PyFunction;
  import org.testng.Assert;

  public class JavaCallPythonDemo {

    public static void main(String[] args) {
      // Set the code-search-path to the directory of your `ray_demo.py` file.
      System.setProperty("ray.job.code-search-path", "/path/to/the_dir/");
      Ray.init();

      // Define a Python class.
      PyActorClass actorClass = PyActorClass.of(
          "ray_demo", "Counter");

      // Create a Python actor and call actor method.
      PyActorHandle actor = Ray.actor(actorClass).remote();
      ObjectRef objRef1 = actor.task(
          PyActorMethod.of("increment", int.class)).remote();
      Assert.assertEquals(objRef1.get(), 1);
      ObjectRef objRef2 = actor.task(
          PyActorMethod.of("increment", int.class)).remote();
      Assert.assertEquals(objRef2.get(), 2);

      // Call the Python remote function.
      ObjectRef objRef3 = Ray.task(PyFunction.of(
          "ray_demo", "add", int.class), 1, 2).remote();
      Assert.assertEquals(objRef3.get(), 3);

      Ray.shutdown();
    }
  }

Cross-language data serialization
---------------------------------

The arguments and return values of ray call can be serialized & deserialized
automatically if their types are the following:

  - Primitive data types
      ===========   =======  =======
      MessagePack   Python   Java
      ===========   =======  =======
      nil           None     null
      bool          bool     Boolean
      int           int      Short / Integer / Long / BigInteger
      float         float    Float / Double
      str           str      String
      bin           bytes    byte[]
      ===========   =======  =======

  - Basic container types
      ===========   =======  =======
      MessagePack   Python   Java
      ===========   =======  =======
      array         list     Array
      ===========   =======  =======

  - Ray builtin types
      - ActorHandle

.. note::

  * Be aware of float / double precision between Python and Java. If Java is using a
    float type to receive the input argument, the double precision Python data
    will be reduced to float precision in Java.
  * BigInteger can support a max value of 2^64-1, please refer to:
    https://github.com/msgpack/msgpack/blob/master/spec.md#int-format-family.
    If the value is larger than 2^64-1, then sending the value to Python will raise an exception.

The following example shows how to pass these types as parameters and how to
return these types.

You can write a Python function which returns the input data:

.. literalinclude:: ./doc_code/cross_language.py
  :language: python
  :start-after: __serialization_start__
  :end-before: __serialization_end__

Then you can transfer the object from Java to Python, and back from Python
to Java:

.. code-block:: java

  package io.ray.demo;

  import io.ray.api.ObjectRef;
  import io.ray.api.Ray;
  import io.ray.api.function.PyFunction;
  import java.math.BigInteger;
  import org.testng.Assert;

  public class SerializationDemo {

    public static void main(String[] args) {
      Ray.init();

      Object[] inputs = new Object[]{
          true,  // Boolean
          Byte.MAX_VALUE,  // Byte
          Short.MAX_VALUE,  // Short
          Integer.MAX_VALUE,  // Integer
          Long.MAX_VALUE,  // Long
          BigInteger.valueOf(Long.MAX_VALUE),  // BigInteger
          "Hello World!",  // String
          1.234f,  // Float
          1.234,  // Double
          "example binary".getBytes()};  // byte[]
      for (Object o : inputs) {
        ObjectRef res = Ray.task(
            PyFunction.of("ray_serialization", "py_return_input", o.getClass()),
            o).remote();
        Assert.assertEquals(res.get(), o);
      }

      Ray.shutdown();
    }
  }

Cross-language exception stacks
-------------------------------

Suppose we have a Java package as follows:

.. code-block:: java

  package io.ray.demo;

  import io.ray.api.ObjectRef;
  import io.ray.api.Ray;
  import io.ray.api.function.PyFunction;

  public class MyRayClass {

    public static int raiseExceptionFromPython() {
      PyFunction<Integer> raiseException = PyFunction.of(
          "ray_exception", "raise_exception", Integer.class);
      ObjectRef<Integer> refObj = Ray.task(raiseException).remote();
      return refObj.get();
    }
  }

and a Python module as follows:

.. literalinclude:: ./doc_code/cross_language.py
  :language: python
  :start-after: __raise_exception_start__
  :end-before: __raise_exception_end__

Then, run the following code:

.. literalinclude:: ./doc_code/cross_language.py
  :language: python
  :start-after: __raise_exception_demo_start__
  :end-before: __raise_exception_demo_end__

The exception stack will be:

.. code-block:: text

  Traceback (most recent call last):
    File "ray_exception_demo.py", line 9, in <module>
      ray.get(obj_ref)  # <-- raise exception from here.
    File "ray/python/ray/_private/client_mode_hook.py", line 105, in wrapper
      return func(*args, **kwargs)
    File "ray/python/ray/_private/worker.py", line 2247, in get
      raise value
  ray.exceptions.CrossLanguageError: An exception raised from JAVA:
  io.ray.api.exception.RayTaskException: (pid=61894, ip=172.17.0.2) Error executing task c8ef45ccd0112571ffffffffffffffffffffffff01000000
          at io.ray.runtime.task.TaskExecutor.execute(TaskExecutor.java:186)
          at io.ray.runtime.RayNativeRuntime.nativeRunTaskExecutor(Native Method)
          at io.ray.runtime.RayNativeRuntime.run(RayNativeRuntime.java:231)
          at io.ray.runtime.runner.worker.DefaultWorker.main(DefaultWorker.java:15)
  Caused by: io.ray.api.exception.CrossLanguageException: An exception raised from PYTHON:
  ray.exceptions.RayTaskError: ray::raise_exception() (pid=62041, ip=172.17.0.2)
    File "ray_exception.py", line 7, in raise_exception
      1 / 0
  ZeroDivisionError: division by zero


.. _ray-remote-classes:
.. _actor-guide:

Actors
======

Actors extend the Ray API from functions (tasks) to classes.
An actor is essentially a stateful worker (or a service). When a new actor is
instantiated, a new worker is created, and methods of the actor are scheduled on
that specific worker and can access and mutate the state of that worker.

.. tab-set::

    .. tab-item:: Python

        The ``ray.remote`` decorator indicates that instances of the ``Counter`` class will be actors. Each actor runs in its own Python process.

        .. testcode::

          import ray

          @ray.remote
          class Counter:
              def __init__(self):
                  self.value = 0

              def increment(self):
                  self.value += 1
                  return self.value

              def get_counter(self):
                  return self.value

          # Create an actor from this class.
          counter = Counter.remote()

    .. tab-item:: Java

        ``Ray.actor`` is used to create actors from regular Java classes.

        .. code-block:: java

          // A regular Java class.
          public class Counter {

            private int value = 0;

            public int increment() {
              this.value += 1;
              return this.value;
            }
          }

          // Create an actor from this class.
          // `Ray.actor` takes a factory method that can produce
          // a `Counter` object. Here, we pass `Counter`'s constructor
          // as the argument.
          ActorHandle<Counter> counter = Ray.actor(Counter::new).remote();

    .. tab-item:: C++

        ``ray::Actor`` is used to create actors from regular C++ classes.

        .. code-block:: c++

          // A regular C++ class.
          class Counter {

          private:
              int value = 0;

          public:
            int Increment() {
              value += 1;
              return value;
            }
          };

          // Factory function of Counter class.
          static Counter *CreateCounter() {
              return new Counter();
          };

          RAY_REMOTE(&Counter::Increment, CreateCounter);

          // Create an actor from this class.
          // `ray::Actor` takes a factory method that can produce
          // a `Counter` object. Here, we pass `Counter`'s factory function
          // as the argument.
          auto counter = ray::Actor(CreateCounter).Remote();



Use `ray list actors` from :ref:`State API <state-api-overview-ref>` to see actors states:

.. code-block:: bash

  # This API is only available when you install Ray with `pip install "ray[default]"`.
  ray list actors

.. code-block:: bash

  ======== List: 2023-05-25 10:10:50.095099 ========
  Stats:
  ------------------------------
  Total: 1
  
  Table:
  ------------------------------
      ACTOR_ID                          CLASS_NAME    STATE      JOB_ID  NAME    NODE_ID                                                     PID  RAY_NAMESPACE
   0  9e783840250840f87328c9f201000000  Counter       ALIVE    01000000          13a475571662b784b4522847692893a823c78f1d3fd8fd32a2624923  38906  ef9de910-64fb-4575-8eb5-50573faa3ddf


Specifying required resources
-----------------------------

.. _actor-resource-guide:

You can specify resource requirements in actors too (see :ref:`resource-requirements` for more details.)

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            # Specify required resources for an actor.
            @ray.remote(num_cpus=2, num_gpus=0.5)
            class Actor:
                pass

    .. tab-item:: Java

        .. code-block:: java

            // Specify required resources for an actor.
            Ray.actor(Counter::new).setResource("CPU", 2.0).setResource("GPU", 0.5).remote();

    .. tab-item:: C++

        .. code-block:: c++

            // Specify required resources for an actor.
            ray::Actor(CreateCounter).SetResource("CPU", 2.0).SetResource("GPU", 0.5).Remote();


Calling the actor
-----------------

We can interact with the actor by calling its methods with the ``remote``
operator. We can then call ``get`` on the object ref to retrieve the actual
value.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            # Call the actor.
            obj_ref = counter.increment.remote()
            print(ray.get(obj_ref))

        .. testoutput::

            1

    .. tab-item:: Java

        .. code-block:: java

            // Call the actor.
            ObjectRef<Integer> objectRef = counter.task(&Counter::increment).remote();
            Assert.assertTrue(objectRef.get() == 1);

    .. tab-item:: C++

        .. code-block:: c++

            // Call the actor.
            auto object_ref = counter.Task(&Counter::increment).Remote();
            assert(*object_ref.Get() == 1);

Methods called on different actors can execute in parallel, and methods called on the same actor are executed serially in the order that they are called. Methods on the same actor will share state with one another, as shown below.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            # Create ten Counter actors.
            counters = [Counter.remote() for _ in range(10)]

            # Increment each Counter once and get the results. These tasks all happen in
            # parallel.
            results = ray.get([c.increment.remote() for c in counters])
            print(results)

            # Increment the first Counter five times. These tasks are executed serially
            # and share state.
            results = ray.get([counters[0].increment.remote() for _ in range(5)])
            print(results)

        .. testoutput::

            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            [2, 3, 4, 5, 6]

    .. tab-item:: Java

        .. code-block:: java

            // Create ten Counter actors.
            List<ActorHandle<Counter>> counters = new ArrayList<>();
            for (int i = 0; i < 10; i++) {
                counters.add(Ray.actor(Counter::new).remote());
            }

            // Increment each Counter once and get the results. These tasks all happen in
            // parallel.
            List<ObjectRef<Integer>> objectRefs = new ArrayList<>();
            for (ActorHandle<Counter> counterActor : counters) {
                objectRefs.add(counterActor.task(Counter::increment).remote());
            }
            // prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            System.out.println(Ray.get(objectRefs));

            // Increment the first Counter five times. These tasks are executed serially
            // and share state.
            objectRefs = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                objectRefs.add(counters.get(0).task(Counter::increment).remote());
            }
            // prints [2, 3, 4, 5, 6]
            System.out.println(Ray.get(objectRefs));

    .. tab-item:: C++

        .. code-block:: c++

            // Create ten Counter actors.
            std::vector<ray::ActorHandle<Counter>> counters;
            for (int i = 0; i < 10; i++) {
                counters.emplace_back(ray::Actor(CreateCounter).Remote());
            }

            // Increment each Counter once and get the results. These tasks all happen in
            // parallel.
            std::vector<ray::ObjectRef<int>> object_refs;
            for (ray::ActorHandle<Counter> counter_actor : counters) {
                object_refs.emplace_back(counter_actor.Task(&Counter::Increment).Remote());
            }
            // prints 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            auto results = ray::Get(object_refs);
            for (const auto &result : results) {
                std::cout << *result;
            }

            // Increment the first Counter five times. These tasks are executed serially
            // and share state.
            object_refs.clear();
            for (int i = 0; i < 5; i++) {
                object_refs.emplace_back(counters[0].Task(&Counter::Increment).Remote());
            }
            // prints 2, 3, 4, 5, 6
            results = ray::Get(object_refs);
            for (const auto &result : results) {
                std::cout << *result;
            }

Passing Around Actor Handles
----------------------------

Actor handles can be passed into other tasks. We can define remote functions (or actor methods) that use actor handles.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import time

            @ray.remote
            def f(counter):
                for _ in range(10):
                    time.sleep(0.1)
                    counter.increment.remote()

    .. tab-item:: Java

        .. code-block:: java

            public static class MyRayApp {

              public static void foo(ActorHandle<Counter> counter) throws InterruptedException {
                for (int i = 0; i < 1000; i++) {
                  TimeUnit.MILLISECONDS.sleep(100);
                  counter.task(Counter::increment).remote();
                }
              }
            }

    .. tab-item:: C++

        .. code-block:: c++

            void Foo(ray::ActorHandle<Counter> counter) {
                for (int i = 0; i < 1000; i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    counter.Task(&Counter::Increment).Remote();
                }
            }

If we instantiate an actor, we can pass the handle around to various tasks.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            counter = Counter.remote()

            # Start some tasks that use the actor.
            [f.remote(counter) for _ in range(3)]

            # Print the counter value.
            for _ in range(10):
                time.sleep(0.1)
                print(ray.get(counter.get_counter.remote()))

        .. testoutput::
            :options: +MOCK

            0
            3
            8
            10
            15
            18
            20
            25
            30
            30

    .. tab-item:: Java

        .. code-block:: java

            ActorHandle<Counter> counter = Ray.actor(Counter::new).remote();

            // Start some tasks that use the actor.
            for (int i = 0; i < 3; i++) {
              Ray.task(MyRayApp::foo, counter).remote();
            }

            // Print the counter value.
            for (int i = 0; i < 10; i++) {
              TimeUnit.SECONDS.sleep(1);
              System.out.println(counter.task(Counter::getCounter).remote().get());
            }

    .. tab-item:: C++

        .. code-block:: c++

            auto counter = ray::Actor(CreateCounter).Remote();

            // Start some tasks that use the actor.
            for (int i = 0; i < 3; i++) {
              ray::Task(Foo).Remote(counter);
            }

            // Print the counter value.
            for (int i = 0; i < 10; i++) {
              std::this_thread::sleep_for(std::chrono::seconds(1));
              std::cout << *counter.Task(&Counter::GetCounter).Remote().Get() << std::endl;
            }



Generators
----------
Ray is compatible with Python generator syntax. See :ref:`Ray Generators <generators>` for more details.

Cancelling Actor Tasks
----------------------

Cancel Actor Tasks by calling :func:`ray.cancel() <ray.cancel>` on the returned `ObjectRef`.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/actors.py
            :language: python
            :start-after: __cancel_start__
            :end-before: __cancel_end__


In Ray, Task cancellation behavior is contingent on the Task's current state:

**Unscheduled Tasks**:
If the Actor Task hasn't been scheduled yet, Ray attempts to cancel the scheduling. 
When successfully cancelled at this stage, invoking ``ray.get(actor_task_ref)`` 
produce a :class:`TaskCancelledError <ray.exceptions.TaskCancelledError>`.

**Running Actor Tasks (Regular Actor, Threaded Actor)**:
For tasks classified as a single-threaded Actor or a multi-threaded Actor,
Ray offers no mechanism for interruption.

**Running Async Actor Tasks**:
For Tasks classified as `async Actors <_async-actors>`, Ray seeks to cancel the associated `asyncio.Task`. 
This cancellation approach aligns with the standards presented in 
`asyncio task cancellation <https://docs.python.org/3/library/asyncio-task.html#task-cancellation>`__.
Note that `asyncio.Task` won't be interrupted in the middle of execution if you don't `await` within the async function.

**Cancellation Guarantee**:
Ray attempts to cancel Tasks on a *best-effort* basis, meaning cancellation isn't always guaranteed.
For example, if the cancellation request doesn't get through to the executor,
the Task might not be cancelled.
You can check if a Task was successfully cancelled using ``ray.get(actor_task_ref)``.

**Recursive Cancellation**:
Ray tracks all child and Actor Tasks. When the``recursive=True`` argument is given,
it cancels all child and Actor Tasks.

Scheduling
----------

For each actor, Ray will choose a node to run it
and the scheduling decision is based on a few factors like
:ref:`the actor's resource requirements <ray-scheduling-resources>`
and :ref:`the specified scheduling strategy <ray-scheduling-strategies>`.
See :ref:`Ray scheduling <ray-scheduling>` for more details.

Fault Tolerance
---------------

By default, Ray actors won't be :ref:`restarted <fault-tolerance-actors>` and
actor tasks won't be retried when actors crash unexpectedly.
You can change this behavior by setting
``max_restarts`` and ``max_task_retries`` options
in :func:`ray.remote() <ray.remote>` and :meth:`.options() <ray.actor.ActorClass.options>`.
See :ref:`Ray fault tolerance <fault-tolerance>` for more details.

FAQ: Actors, Workers and Resources
----------------------------------

What's the difference between a worker and an actor?

Each "Ray worker" is a python process.

Workers are treated differently for tasks and actors. Any "Ray worker" is either 1. used to execute multiple Ray tasks or 2. is started as a dedicated Ray actor.

* **Tasks**: When Ray starts on a machine, a number of Ray workers will be started automatically (1 per CPU by default). They will be used to execute tasks (like a process pool). If you execute 8 tasks with `num_cpus=2`, and total number of CPUs is 16 (`ray.cluster_resources()["CPU"] == 16`), you will end up with 8 of your 16 workers idling.

* **Actor**: A Ray Actor is also a "Ray worker" but is instantiated at runtime (upon `actor_cls.remote()`). All of its methods will run on the same process, using the same resources (designated when defining the Actor). Note that unlike tasks, the python processes that runs Ray Actors are not reused and will be terminated when the Actor is deleted.

To maximally utilize your resources, you want to maximize the time that
your workers are working. You also want to allocate enough cluster resources
so that both all of your needed actors can run and any other tasks you
define can run. This also implies that tasks are scheduled more flexibly,
and that if you don't need the stateful part of an actor, you're mostly
better off using tasks.

Task Events 
-----------

By default, Ray traces the execution of actor tasks, reporting task status events and profiling events
that Ray Dashboard and :ref:`State API <state-api-overview-ref>` use.

You can disable task events for the actor by setting the `enable_task_events` option to `False` in :func:`ray.remote() <ray.remote>` and :meth:`.options() <ray.actor.ActorClass.options>`, which reduces the overhead of task execution, and the amount of data the being sent to the Ray Dashboard.

You can also disable task events for some actor methods by setting the `enable_task_events` option to `False` in :func:`ray.remote() <ray.remote>` and :meth:`.options() <ray.remote_function.RemoteFunction.options>` on the actor method.
Method settings override the actor setting:

.. literalinclude:: doc_code/actors.py
    :language: python
    :start-after: __enable_task_events_start__
    :end-before: __enable_task_events_end__


More about Ray Actors
---------------------

.. toctree::
    :maxdepth: 1

    actors/named-actors.rst
    actors/terminating-actors.rst
    actors/async_api.rst
    actors/concurrency_group_api.rst
    actors/actor-utils.rst
    actors/out-of-band-communication.rst
    actors/task-orders.rst


Starting Ray
============

This page covers how to start Ray on your single machine or cluster of machines.

.. tip:: Be sure to have :ref:`installed Ray <installation>` before following the instructions on this page.


What is the Ray runtime?
------------------------

Ray programs are able to parallelize and distribute by leveraging an underlying *Ray runtime*.
The Ray runtime consists of multiple services/processes started in the background for communication, data transfer, scheduling, and more. The Ray runtime can be started on a laptop, a single server, or multiple servers.

There are three ways of starting the Ray runtime:

* Implicitly via ``ray.init()`` (:ref:`start-ray-init`)
* Explicitly via CLI (:ref:`start-ray-cli`)
* Explicitly via the cluster launcher (:ref:`start-ray-up`)

In all cases, ``ray.init()`` will try to automatically find a Ray instance to
connect to. It checks, in order:
1. The ``RAY_ADDRESS`` OS environment variable.
2. The concrete address passed to ``ray.init(address=<address>)``.
3. If no address is provided, the latest Ray instance that was started on the same machine using ``ray start``.

.. _start-ray-init:

Starting Ray on a single machine
--------------------------------

Calling ``ray.init()`` starts a local Ray instance on your laptop/machine. This laptop/machine becomes the  "head node".

.. note::

  In recent versions of Ray (>=1.5), ``ray.init()`` will automatically be called on the first use of a Ray remote API.

.. tab-set::

    .. tab-item:: Python

        .. testcode::
          :hide:

          import ray
          ray.shutdown()

        .. testcode::

          import ray
          # Other Ray APIs will not work until `ray.init()` is called.
          ray.init()

    .. tab-item:: Java

        .. code-block:: java

            import io.ray.api.Ray;

            public class MyRayApp {

              public static void main(String[] args) {
                // Other Ray APIs will not work until `Ray.init()` is called.
                Ray.init();
                ...
              }
            }

    .. tab-item:: C++

        .. code-block:: c++

            #include <ray/api.h>
            // Other Ray APIs will not work until `ray::Init()` is called.
            ray::Init()

When the process calling ``ray.init()`` terminates, the Ray runtime will also terminate. To explicitly stop or restart Ray, use the shutdown API.

.. tab-set::

    .. tab-item:: Python

        .. testcode::
          :hide:

          ray.shutdown()

        .. testcode::

            import ray
            ray.init()
            ... # ray program
            ray.shutdown()

    .. tab-item:: Java

        .. code-block:: java

            import io.ray.api.Ray;

            public class MyRayApp {

              public static void main(String[] args) {
                Ray.init();
                ... // ray program
                Ray.shutdown();
              }
            }

    .. tab-item:: C++

        .. code-block:: c++

            #include <ray/api.h>
            ray::Init()
            ... // ray program
            ray::Shutdown()

To check if Ray is initialized, use the ``is_initialized`` API.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import ray
            ray.init()
            assert ray.is_initialized()

            ray.shutdown()
            assert not ray.is_initialized()

    .. tab-item:: Java

        .. code-block:: java

            import io.ray.api.Ray;

            public class MyRayApp {

            public static void main(String[] args) {
                    Ray.init();
                    Assert.assertTrue(Ray.isInitialized());
                    Ray.shutdown();
                    Assert.assertFalse(Ray.isInitialized());
                }
            }

    .. tab-item:: C++

        .. code-block:: c++

            #include <ray/api.h>

            int main(int argc, char **argv) {
                ray::Init();
                assert(ray::IsInitialized());

                ray::Shutdown();
                assert(!ray::IsInitialized());
            }

See the `Configuration <configure.html>`__ documentation for the various ways to configure Ray.

.. _start-ray-cli:

Starting Ray via the CLI (``ray start``)
----------------------------------------

Use ``ray start`` from the CLI to start a 1 node ray runtime on a machine. This machine becomes the "head node".

.. code-block:: bash

  $ ray start --head --port=6379

  Local node IP: 192.123.1.123
  2020-09-20 10:38:54,193 INFO services.py:1166 -- View the Ray dashboard at http://localhost:8265

  --------------------
  Ray runtime started.
  --------------------

  ...


You can connect to this Ray instance by starting a driver process on the same node as where you ran ``ray start``.
``ray.init()`` will now automatically connect to the latest Ray instance.

.. tab-set::

    .. tab-item:: Python

      .. testcode::

        import ray
        ray.init()

    .. tab-item:: java

        .. code-block:: java

          import io.ray.api.Ray;

          public class MyRayApp {

            public static void main(String[] args) {
              Ray.init();
              ...
            }
          }

        .. code-block:: bash

          java -classpath <classpath> \
            -Dray.address=<address> \
            <classname> <args>

    .. tab-item:: C++

        .. code-block:: c++

          #include <ray/api.h>

          int main(int argc, char **argv) {
            ray::Init();
            ...
          }

        .. code-block:: bash

          RAY_ADDRESS=<address> ./<binary> <args>


You can connect other nodes to the head node, creating a Ray cluster by also calling ``ray start`` on those nodes. See :ref:`on-prem` for more details. Calling ``ray.init()`` on any of the cluster machines will connect to the same Ray cluster.

.. _start-ray-up:

Launching a Ray cluster (``ray up``)
------------------------------------

Ray clusters can be launched with the :ref:`Cluster Launcher <cluster-index>`.
The ``ray up`` command uses the Ray cluster launcher to start a cluster on the cloud, creating a designated "head node" and worker nodes. Underneath the hood, it automatically calls ``ray start`` to create a Ray cluster.

Your code **only** needs to execute on one machine in the cluster (usually the head node). Read more about :ref:`running programs on a Ray cluster <cluster-index>`.

To connect to the Ray cluster, call ``ray.init`` from one of the machines in the cluster. This will connect to the latest Ray cluster:

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

  ray.init()

Note that the machine calling ``ray up`` will not be considered as part of the Ray cluster, and therefore calling ``ray.init`` on that same machine will not attach to the cluster.

What's next?
------------

Check out our `Deployment section <cluster/index.html>`_ for more information on deploying Ray in different settings, including Kubernetes, YARN, and SLURM.


.. _ray-remote-functions:

Tasks
=====

Ray enables arbitrary functions to be executed asynchronously on separate Python workers. Such functions are called **Ray remote functions** and their asynchronous invocations are called **Ray tasks**. Here is an example.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __tasks_start__
            :end-before: __tasks_end__

        See the :func:`ray.remote<ray.remote>` API for more details.

    .. tab-item:: Java

        .. code-block:: java

          public class MyRayApp {
            // A regular Java static method.
            public static int myFunction() {
              return 1;
            }
          }

          // Invoke the above method as a Ray task.
          // This will immediately return an object ref (a future) and then create
          // a task that will be executed on a worker process.
          ObjectRef<Integer> res = Ray.task(MyRayApp::myFunction).remote();

          // The result can be retrieved with ``ObjectRef::get``.
          Assert.assertTrue(res.get() == 1);

          public class MyRayApp {
            public static int slowFunction() throws InterruptedException {
              TimeUnit.SECONDS.sleep(10);
              return 1;
            }
          }

          // Ray tasks are executed in parallel.
          // All computation is performed in the background, driven by Ray's internal event loop.
          for(int i = 0; i < 4; i++) {
            // This doesn't block.
            Ray.task(MyRayApp::slowFunction).remote();
          }

    .. tab-item:: C++

        .. code-block:: c++

          // A regular C++ function.
          int MyFunction() {
            return 1;
          }
          // Register as a remote function by `RAY_REMOTE`.
          RAY_REMOTE(MyFunction);

          // Invoke the above method as a Ray task.
          // This will immediately return an object ref (a future) and then create
          // a task that will be executed on a worker process.
          auto res = ray::Task(MyFunction).Remote();

          // The result can be retrieved with ``ray::ObjectRef::Get``.
          assert(*res.Get() == 1);

          int SlowFunction() {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            return 1;
          }
          RAY_REMOTE(SlowFunction);

          // Ray tasks are executed in parallel.
          // All computation is performed in the background, driven by Ray's internal event loop.
          for(int i = 0; i < 4; i++) {
            // This doesn't block.
            ray::Task(SlowFunction).Remote();
          a

Use `ray summary tasks` from :ref:`State API <state-api-overview-ref>`  to see running and finished tasks and count:

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray summary tasks


.. code-block:: bash

  ======== Tasks Summary: 2023-05-26 11:09:32.092546 ========
  Stats:
  ------------------------------------
  total_actor_scheduled: 0
  total_actor_tasks: 0
  total_tasks: 5
  
  
  Table (group by func_name):
  ------------------------------------
      FUNC_OR_CLASS_NAME    STATE_COUNTS    TYPE
  0   slow_function         RUNNING: 4      NORMAL_TASK
  1   my_function           FINISHED: 1     NORMAL_TASK

Specifying required resources
-----------------------------

You can specify resource requirements in tasks (see :ref:`resource-requirements` for more details.)

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __resource_start__
            :end-before: __resource_end__

    .. tab-item:: Java

        .. code-block:: java

            // Specify required resources.
            Ray.task(MyRayApp::myFunction).setResource("CPU", 4.0).setResource("GPU", 2.0).remote();

    .. tab-item:: C++

        .. code-block:: c++

            // Specify required resources.
            ray::Task(MyFunction).SetResource("CPU", 4.0).SetResource("GPU", 2.0).Remote();

.. _ray-object-refs:

Passing object refs to Ray tasks
---------------------------------------

In addition to values, `Object refs <objects.html>`__ can also be passed into remote functions. When the task gets executed, inside the function body **the argument will be the underlying value**. For example, take this function:

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __pass_by_ref_start__
            :end-before: __pass_by_ref_end__

    .. tab-item:: Java

        .. code-block:: java

            public class MyRayApp {
                public static int functionWithAnArgument(int value) {
                    return value + 1;
                }
            }

            ObjectRef<Integer> objRef1 = Ray.task(MyRayApp::myFunction).remote();
            Assert.assertTrue(objRef1.get() == 1);

            // You can pass an object ref as an argument to another Ray task.
            ObjectRef<Integer> objRef2 = Ray.task(MyRayApp::functionWithAnArgument, objRef1).remote();
            Assert.assertTrue(objRef2.get() == 2);

    .. tab-item:: C++

        .. code-block:: c++

            static int FunctionWithAnArgument(int value) {
                return value + 1;
            }
            RAY_REMOTE(FunctionWithAnArgument);

            auto obj_ref1 = ray::Task(MyFunction).Remote();
            assert(*obj_ref1.Get() == 1);

            // You can pass an object ref as an argument to another Ray task.
            auto obj_ref2 = ray::Task(FunctionWithAnArgument).Remote(obj_ref1);
            assert(*obj_ref2.Get() == 2);

Note the following behaviors:

  -  As the second task depends on the output of the first task, Ray will not execute the second task until the first task has finished.
  -  If the two tasks are scheduled on different machines, the output of the
     first task (the value corresponding to ``obj_ref1/objRef1``) will be sent over the
     network to the machine where the second task is scheduled.

Waiting for Partial Results
---------------------------

Calling **ray.get** on Ray task results will block until the task finished execution. After launching a number of tasks, you may want to know which ones have
finished executing without blocking on all of them. This could be achieved by :func:`ray.wait() <ray.wait>`. The function
works as follows.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __wait_start__
            :end-before: __wait_end__

    .. tab-item:: Java

      .. code-block:: java

        WaitResult<Integer> waitResult = Ray.wait(objectRefs, /*num_returns=*/0, /*timeoutMs=*/1000);
        System.out.println(waitResult.getReady());  // List of ready objects.
        System.out.println(waitResult.getUnready());  // list of unready objects.

    .. tab-item:: C++

      .. code-block:: c++

        ray::WaitResult<int> wait_result = ray::Wait(object_refs, /*num_objects=*/0, /*timeout_ms=*/1000);

Generators
----------
Ray is compatible with Python generator syntax. See :ref:`Ray Generators <generators>` for more details.

.. _ray-task-returns:

Multiple returns
----------------

By default, a Ray task only returns a single Object Ref. However, you can configure Ray tasks to return multiple Object Refs, by setting the ``num_returns`` option.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __multiple_returns_start__
            :end-before: __multiple_returns_end__

For tasks that return multiple objects, Ray also supports remote generators that allow a task to return one object at a time to reduce memory usage at the worker. Ray also supports an option to set the number of return values dynamically, which can be useful when the task caller does not know how many return values to expect. See the :ref:`user guide <generators>` for more details on use cases.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __generator_start__
            :end-before: __generator_end__

.. _ray-task-cancel:

Cancelling tasks
----------------

Ray tasks can be canceled by calling :func:`ray.cancel() <ray.cancel>` on the returned Object ref.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: doc_code/tasks.py
            :language: python
            :start-after: __cancel_start__
            :end-before: __cancel_end__


Scheduling
----------

For each task, Ray will choose a node to run it
and the scheduling decision is based on a few factors like
:ref:`the task's resource requirements <ray-scheduling-resources>`,
:ref:`the specified scheduling strategy <ray-scheduling-strategies>`
and :ref:`locations of task arguments <ray-scheduling-locality>`.
See :ref:`Ray scheduling <ray-scheduling>` for more details.

Fault Tolerance
---------------

By default, Ray will :ref:`retry <task-retries>` failed tasks
due to system failures and specified application-level failures.
You can change this behavior by setting
``max_retries`` and ``retry_exceptions`` options
in :func:`ray.remote() <ray.remote>` and :meth:`.options() <ray.remote_function.RemoteFunction.options>`.
See :ref:`Ray fault tolerance <fault-tolerance>` for more details.

Task Events
-----------


By default, Ray traces the execution of tasks, reporting task status events and profiling events
that the Ray Dashboard and :ref:`State API <state-api-overview-ref>` use.

You can change this behavior by setting ``enable_task_events`` options in :func:`ray.remote() <ray.remote>` and :meth:`.options() <ray.remote_function.RemoteFunction.options>`
to disable task events, which reduces the overhead of task execution, and the amount of data the task sends to the Ray Dashboard.
Nested tasks don't inherit the task events settings from the parent task. You need to set the task events settings for each task separately.



More about Ray Tasks
--------------------

.. toctree::
    :maxdepth: 1

    tasks/nested-tasks.rst
    tasks/generators.rst


.. _core-use-guide:

User Guides
===========

This section explains how to use Ray's key concepts to build distributed applications.

If you’re brand new to Ray, we recommend starting with the :ref:`walkthrough <core-walkthrough>`.

.. toctree::
    :maxdepth: 4

    tasks
    actors
    objects
    handling-dependencies
    scheduling/index.rst
    fault-tolerance
    patterns/index.rst
    advanced-topics


.. _ray-dag-guide:

Lazy Computation Graphs with the Ray DAG API
============================================

With ``ray.remote`` you have the flexibility of running an application where
computation is executed remotely at runtime. For a ``ray.remote`` decorated
class or function, you can also use ``.bind`` on the body to build a static
computation graph.

.. note::

     Ray DAG is designed to be a developer facing API where recommended use cases
     are

     1) Locally iterate and test your application authored by higher level libraries.

     2) Build libraries on top of the Ray DAG APIs.


When ``.bind()`` is called on a ``ray.remote`` decorated class or function, it will
generate an intermediate representation (IR) node that act as backbone and
building blocks of the DAG that is statically holding the computation graph
together, where each IR node is resolved to value at execution time with
respect to their topological order.

The IR node can also be assigned to a variable and passed into other nodes as
arguments.

Ray DAG with functions
----------------------

The IR node generated by ``.bind()`` on a ``ray.remote`` decorated function is
executed as a Ray Task upon execution which will be solved to the task output.

This example shows how to build a chain of functions where each node can be
executed as root node while iterating, or used as input args or kwargs of other
functions to form more complex DAGs.

Any IR node can be executed directly ``dag_node.execute()`` that acts as root
of the DAG, where all other non-reachable nodes from the root will be igored.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/ray-dag.py
          :language: python
          :start-after: __dag_tasks_begin__
          :end-before: __dag_tasks_end__


Ray DAG with classes and class methods
--------------------------------------

The IR node generated by ``.bind()`` on a ``ray.remote`` decorated class is
executed as a Ray Actor upon execution. The Actor will be instantiated every
time the node is executed, and the classmethod calls can form a chain of
function calls specific to the parent actor instance.

DAG IR nodes generated from a function, class or classmethod can be combined
together to form a DAG.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/ray-dag.py
          :language: python
          :start-after: __dag_actors_begin__
          :end-before: __dag_actors_end__



Ray DAG with custom InputNode
-----------------------------

``InputNode`` is the singleton node of a DAG that represents user input value at
runtime. It should be used within a context manager with no args, and called
as args of ``dag_node.execute()``

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/ray-dag.py
          :language: python
          :start-after: __dag_input_node_begin__
          :end-before: __dag_input_node_end__

Ray DAG with multiple MultiOutputNode
-------------------------------------

``MultiOutputNode`` is useful when you have more than 1 output from a DAG. ``dag_node.execute()``
returns a list of Ray object references passed to ``MultiOutputNode``. The below example
shows the multi output node of 2 outputs.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/ray-dag.py
          :language: python
          :start-after: __dag_multi_output_node_begin__
          :end-before: __dag_multi_output_node_end__

Reuse Ray Actors in DAGs
------------------------
Actors can be a part of the DAG definition with the ``Actor.bind()`` API.
However, when a DAG finishes execution, Ray kills Actors created with ``bind``.

You can avoid killing your Actors whenever DAG finishes by creating Actors with ``Actor.remote()``.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ./doc_code/ray-dag.py
          :language: python
          :start-after: __dag_actor_reuse_begin__
          :end-before: __dag_actor_reuse_end__


More resources
--------------

You can find more application patterns and examples in the following resources
from other Ray libraries built on top of Ray DAG API with the same mechanism.

| `Visualization of DAGs <https://docs.ray.io/en/master/serve/model_composition.html#visualizing-the-graph>`_


Lifetimes of a User-Spawn Process
=================================

When you spawns child processes from Ray workers, you are responsible for managing the lifetime of child processes. However, it is not always possible, especially when worker crashes and child processes are spawned from libraries (torch dataloader).

To avoid leaking user-spawned processes, Ray provides mechanisms to kill all user-spawned processes when a worker that starts it exits. This feature prevents GPU memory leaks from child processes (e.g., torch).

We have 2 environment variables to handle subprocess killing on worker exit:

- ``RAY_kill_child_processes_on_worker_exit`` (default ``true``): Only works on Linux. If true, the worker kills all *direct* child processes on exit. This won't work if the worker crashed. This is NOT recursive, in that grandchild processes are not killed by this mechanism.

- ``RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper`` (default ``false``): Only works on Linux greater than or equal to 3.4. If true, Raylet *recursively* kills any child processes and grandchild processes that were spawned by the worker after the worker exits. This works even if the worker crashed. The killing happens within 10 seconds after the worker death.

On non-Linux platforms, user-spawned process is not controlled by Ray. The user is responsible for managing the lifetime of the child processes. If the parent Ray worker process dies, the child processes will continue to run.

Note: The feature is meant to be a last resort to kill orphaned processes. It is not a replacement for proper process management. Users should still manage the lifetime of their processes and clean up properly.

.. contents::
  :local:

User-Spawned Process Killed on Worker Exit
------------------------------------------

In the following example, we use Ray Actor to spawn a user process. The user process is a long running process that prints "Hello, world!" every second. The user process is killed when the actor is killed.

.. testcode::

  import ray
  import psutil
  import subprocess
  import time
  import os

  ray.init(_system_config={"kill_child_processes_on_worker_exit_with_raylet_subreaper":True})

  @ray.remote
  class MyActor:
    def __init__(self):
      pass

    def start(self):
      # Start a user process
      process = subprocess.Popen(["/bin/bash", "-c", "sleep 10000"])
      return process.pid

    def signal_my_pid(self):
      import signal
      os.kill(os.getpid(), signal.SIGKILL)


  actor = MyActor.remote()

  pid = ray.get(actor.start.remote())
  assert psutil.pid_exists(pid)  # the subprocess running

  actor.signal_my_pid.remote()  # sigkill'ed, the worker's subprocess killing no longer works
  time.sleep(11)  # raylet kills orphans every 10s
  assert not psutil.pid_exists(pid)


Enabling the feature
-------------------------

To enable the subreaper feature, set the environment variable ``RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper`` to ``true`` **when starting the Ray cluster**, If a Ray cluster is already running, you need to restart the Ray cluster to apply the change. Setting ``env_var`` in a runtime environment will NOT work.

.. code-block:: bash

  RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper=true ray start --head

Another way is to enable it during ``ray.init()`` by adding a ``_system_config`` like this:

.. code-block::

  ray.init(_system_config={"kill_child_processes_on_worker_exit_with_raylet_subreaper":True})


⚠️ Caution: Core worker now reaps zombies, toggle back if you wait to ``waitpid``
----------------------------------------------------------------------------------

When the feature is enabled, the worker process becomes a subreaper (see the next section), meaning there can be some grandchildren processes that are reparented to the worker process. To reap these processes, the worker sets the ``SIGCHLD`` signal to ``SIG_IGN``. This makes the worker not receive the ``SIGCHLD`` signal when its children exit. If you need to wait for a child process to exit, you need to reset the ``SIGCHLD`` signal to ``SIG_DFL``.

.. code-block::

  import signal
  signal.signal(signal.SIGCHLD, signal.SIG_DFL)


Under the hood
-------------------------

This feature is implemented by setting the `prctl(PR_SET_CHILD_SUBREAPER, 1)` flag on the Raylet process which spawns all Ray workers. See `prctl(2) <https://man7.org/linux/man-pages/man2/prctl.2.html>`_. This flag makes the Raylet process a "subreaper" which means that if a descendant child process dies, the dead child's children processes reparent to the Raylet process.

Raylet maintains a list of "known" direct children pid it spawns, and when the Raylet process receives the SIGCHLD signal, it knows that one of its child processes (e.g. the workers) has died, and maybe there are reparented orphan processes. Raylet lists all children pids (with ppid = raylet pid), and if a child pid is not "known" (i.e. not in the list of direct children pids), Raylet thinks it is an orphan process and kills it via `SIGKILL`.

For a deep chain of process creations, Raylet would do the killing step by step. For example, in a chain like this:

.. code-block::

  raylet -> the worker -> user process A -> user process B -> user process C

When the ``the worker`` dies, ``Raylet`` kills the ``user process A``, because it's not on the "known" children list. When ``user process A`` dies, ``Raylet`` kills ``user process B``, and so on.

An edge case is, if the ``the worker`` is still alive but the ``user process A`` is dead, then ``user process B`` gets reparented and risks being killed. To mitigate, ``Ray`` also sets the ``the worker`` as a subreaper, so it can adopt the reparented processes. ``Core worker`` does not kill unknown children processes, so a user "daemon" process e.g. ``user process B`` that outlives ``user process A`` can live along. However if the ``the worker`` dies, the user daemon process gets reparented to ``raylet`` and gets killed.

Related PR: `Use subreaper to kill unowned subprocesses in raylet. (#42992) <https://github.com/ray-project/ray/pull/42992>`_

.. _core-key-concepts:

Key Concepts
============

This section overviews Ray's key concepts. These primitives work together to enable Ray to flexibly support a broad range of distributed applications.

.. _task-key-concept:

Tasks
-----

Ray enables arbitrary functions to be executed asynchronously on separate Python workers. These asynchronous Ray functions are called "tasks". Ray enables tasks to specify their resource requirements in terms of CPUs, GPUs, and custom resources. These resource requests are used by the cluster scheduler to distribute tasks across the cluster for parallelized execution.

See the :ref:`User Guide for Tasks <ray-remote-functions>`.

.. _actor-key-concept:

Actors
------

Actors extend the Ray API from functions (tasks) to classes. An actor is essentially a stateful worker (or a service). When a new actor is instantiated, a new worker is created, and methods of the actor are scheduled on that specific worker and can access and mutate the state of that worker. Like tasks, actors support CPU, GPU, and custom resource requirements.

See the :ref:`User Guide for Actors <actor-guide>`.

Objects
-------

In Ray, tasks and actors create and compute on objects. We refer to these objects as *remote objects* because they can be stored anywhere in a Ray cluster, and we use *object refs* to refer to them. Remote objects are cached in Ray's distributed `shared-memory <https://en.wikipedia.org/wiki/Shared_memory>`__ *object store*, and there is one object store per node in the cluster. In the cluster setting, a remote object can live on one or many nodes, independent of who holds the object ref(s).

See the :ref:`User Guide for Objects <objects-in-ray>`.

Placement Groups
----------------

Placement groups allow users to atomically reserve groups of resources across multiple nodes (i.e., gang scheduling). They can be then used to schedule Ray tasks and actors packed as close as possible for locality (PACK), or spread apart (SPREAD). Placement groups are generally used for gang-scheduling actors, but also support tasks.

See the :ref:`User Guide for Placement Groups <ray-placement-group-doc-ref>`.

Environment Dependencies
------------------------

When Ray executes tasks and actors on remote machines, their environment dependencies (e.g., Python packages, local files, environment variables) must be available for the code to run. To address this problem, you can (1) prepare your dependencies on the cluster in advance using the Ray :ref:`Cluster Launcher <vm-cluster-quick-start>`, or (2) use Ray's :ref:`runtime environments <runtime-environments>` to install them on the fly.

See the :ref:`User Guide for Environment Dependencies <handling_dependencies>`.


.. _objects-in-ray:

Objects
=======

In Ray, tasks and actors create and compute on objects. We refer to these objects as **remote objects** because they can be stored anywhere in a Ray cluster, and we use **object refs** to refer to them. Remote objects are cached in Ray's distributed `shared-memory <https://en.wikipedia.org/wiki/Shared_memory>`__ **object store**, and there is one object store per node in the cluster. In the cluster setting, a remote object can live on one or many nodes, independent of who holds the object ref(s).

An **object ref** is essentially a pointer or a unique ID that can be used to refer to a
remote object without seeing its value. If you're familiar with futures, Ray object refs are conceptually
similar.

Object refs can be created in two ways.

  1. They are returned by remote function calls.
  2. They are returned by :func:`ray.put() <ray.put>`.

.. tab-set::

    .. tab-item:: Python

      .. testcode::

        import ray

        # Put an object in Ray's object store.
        y = 1
        object_ref = ray.put(y)

    .. tab-item:: Java

      .. code-block:: java

        // Put an object in Ray's object store.
        int y = 1;
        ObjectRef<Integer> objectRef = Ray.put(y);

    .. tab-item:: C++

      .. code-block:: c++

        // Put an object in Ray's object store.
        int y = 1;
        ray::ObjectRef<int> object_ref = ray::Put(y);

.. note::

    Remote objects are immutable. That is, their values cannot be changed after
    creation. This allows remote objects to be replicated in multiple object
    stores without needing to synchronize the copies.


Fetching Object Data
--------------------

You can use the :func:`ray.get() <ray.get>` method to fetch the result of a remote object from an object ref.
If the current node's object store does not contain the object, the object is downloaded.

.. tab-set::

    .. tab-item:: Python

        If the object is a `numpy array <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`__
        or a collection of numpy arrays, the ``get`` call is zero-copy and returns arrays backed by shared object store memory.
        Otherwise, we deserialize the object data into a Python object.

        .. testcode::

          import ray
          import time

          # Get the value of one object ref.
          obj_ref = ray.put(1)
          assert ray.get(obj_ref) == 1

          # Get the values of multiple object refs in parallel.
          assert ray.get([ray.put(i) for i in range(3)]) == [0, 1, 2]

          # You can also set a timeout to return early from a ``get``
          # that's blocking for too long.
          from ray.exceptions import GetTimeoutError
          # ``GetTimeoutError`` is a subclass of ``TimeoutError``.

          @ray.remote
          def long_running_function():
              time.sleep(8)

          obj_ref = long_running_function.remote()
          try:
              ray.get(obj_ref, timeout=4)
          except GetTimeoutError:  # You can capture the standard "TimeoutError" instead
              print("`get` timed out.")

        .. testoutput::

          `get` timed out.

    .. tab-item:: Java

        .. code-block:: java

          // Get the value of one object ref.
          ObjectRef<Integer> objRef = Ray.put(1);
          Assert.assertTrue(objRef.get() == 1);
          // You can also set a timeout(ms) to return early from a ``get`` that's blocking for too long.
          Assert.assertTrue(objRef.get(1000) == 1);

          // Get the values of multiple object refs in parallel.
          List<ObjectRef<Integer>> objectRefs = new ArrayList<>();
          for (int i = 0; i < 3; i++) {
            objectRefs.add(Ray.put(i));
          }
          List<Integer> results = Ray.get(objectRefs);
          Assert.assertEquals(results, ImmutableList.of(0, 1, 2));

          // Ray.get timeout example: Ray.get will throw an RayTimeoutException if time out.
          public class MyRayApp {
            public static int slowFunction() throws InterruptedException {
              TimeUnit.SECONDS.sleep(10);
              return 1;
            }
          }
          Assert.assertThrows(RayTimeoutException.class,
            () -> Ray.get(Ray.task(MyRayApp::slowFunction).remote(), 3000));

    .. tab-item:: C++

        .. code-block:: c++

          // Get the value of one object ref.
          ray::ObjectRef<int> obj_ref = ray::Put(1);
          assert(*obj_ref.Get() == 1);

          // Get the values of multiple object refs in parallel.
          std::vector<ray::ObjectRef<int>> obj_refs;
          for (int i = 0; i < 3; i++) {
            obj_refs.emplace_back(ray::Put(i));
          }
          auto results = ray::Get(obj_refs);
          assert(results.size() == 3);
          assert(*results[0] == 0);
          assert(*results[1] == 1);
          assert(*results[2] == 2);

Passing Object Arguments
------------------------

Ray object references can be freely passed around a Ray application. This means that they can be passed as arguments to tasks, actor methods, and even stored in other objects. Objects are tracked via *distributed reference counting*, and their data is automatically freed once all references to the object are deleted.

There are two different ways one can pass an object to a Ray task or method. Depending on the way an object is passed, Ray will decide whether to *de-reference* the object prior to task execution.

**Passing an object as a top-level argument**: When an object is passed directly as a top-level argument to a task, Ray will de-reference the object. This means that Ray will fetch the underlying data for all top-level object reference arguments, not executing the task until the object data becomes fully available.

.. literalinclude:: doc_code/obj_val.py

**Passing an object as a nested argument**: When an object is passed within a nested object, for example, within a Python list, Ray will *not* de-reference it. This means that the task will need to call ``ray.get()`` on the reference to fetch the concrete value. However, if the task never calls ``ray.get()``, then the object value never needs to be transferred to the machine the task is running on. We recommend passing objects as top-level arguments where possible, but nested arguments can be useful for passing objects on to other tasks without needing to see the data.

.. literalinclude:: doc_code/obj_ref.py

The top-level vs not top-level passing convention also applies to actor constructors and actor method calls:

.. testcode::

    @ray.remote
    class Actor:
      def __init__(self, arg):
        pass

      def method(self, arg):
        pass

    obj = ray.put(2)

    # Examples of passing objects to actor constructors.
    actor_handle = Actor.remote(obj)  # by-value
    actor_handle = Actor.remote([obj])  # by-reference

    # Examples of passing objects to actor method calls.
    actor_handle.method.remote(obj)  # by-value
    actor_handle.method.remote([obj])  # by-reference

Closure Capture of Objects
--------------------------

You can also pass objects to tasks via *closure-capture*. This can be convenient when you have a large object that you want to share verbatim between many tasks or actors, and don't want to pass it repeatedly as an argument. Be aware however that defining a task that closes over an object ref will pin the object via reference-counting, so the object will not be evicted until the job completes.

.. literalinclude:: doc_code/obj_capture.py

Nested Objects
--------------

Ray also supports nested object references. This allows you to build composite objects that themselves hold references to further sub-objects.

.. testcode::

    # Objects can be nested within each other. Ray will keep the inner object
    # alive via reference counting until all outer object references are deleted.
    object_ref_2 = ray.put([object_ref])

Fault Tolerance
---------------

Ray can automatically recover from object data loss
via :ref:`lineage reconstruction <fault-tolerance-objects-reconstruction>`
but not :ref:`owner <fault-tolerance-ownership>` failure.
See :ref:`Ray fault tolerance <fault-tolerance>` for more details.

More about Ray Objects
----------------------

.. toctree::
    :maxdepth: 1

    objects/serialization.rst
    objects/object-spilling.rst


Miscellaneous Topics
====================

This page will cover some miscellaneous topics in Ray.

.. contents::
  :local:

Dynamic Remote Parameters
-------------------------

You can dynamically adjust resource requirements or return values of ``ray.remote`` during execution with ``.options``.

For example, here we instantiate many copies of the same actor with varying resource requirements. Note that to create these actors successfully, Ray will need to be started with sufficient CPU resources and the relevant custom resources:

.. testcode::

  import ray

  @ray.remote(num_cpus=4)
  class Counter(object):
      def __init__(self):
          self.value = 0

      def increment(self):
          self.value += 1
          return self.value

  a1 = Counter.options(num_cpus=1, resources={"Custom1": 1}).remote()
  a2 = Counter.options(num_cpus=2, resources={"Custom2": 1}).remote()
  a3 = Counter.options(num_cpus=3, resources={"Custom3": 1}).remote()

You can specify different resource requirements for tasks (but not for actor methods):

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

    ray.init(num_cpus=1, num_gpus=1)

    @ray.remote
    def g():
        return ray.get_gpu_ids()

    object_gpu_ids = g.remote()
    assert ray.get(object_gpu_ids) == []

    dynamic_object_gpu_ids = g.options(num_cpus=1, num_gpus=1).remote()
    assert ray.get(dynamic_object_gpu_ids) == [0]

And vary the number of return values for tasks (and actor methods too):

.. testcode::

    @ray.remote
    def f(n):
        return list(range(n))

    id1, id2 = f.options(num_returns=2).remote(2)
    assert ray.get(id1) == 0
    assert ray.get(id2) == 1

And specify a name for tasks (and actor methods too) at task submission time:

.. testcode::

   import setproctitle

   @ray.remote
   def f(x):
      assert setproctitle.getproctitle() == "ray::special_f"
      return x + 1

   obj = f.options(name="special_f").remote(3)
   assert ray.get(obj) == 4

This name will appear as the task name in the machine view of the dashboard, will appear
as the worker process name when this task is executing (if a Python task), and will
appear as the task name in the logs.

.. image:: images/task_name_dashboard.png


Overloaded Functions
--------------------
Ray Java API supports calling overloaded java functions remotely. However, due to the limitation of Java compiler type inference, one must explicitly cast the method reference to the correct function type. For example, consider the following.

Overloaded normal task call:

.. code:: java

    public static class MyRayApp {

      public static int overloadFunction() {
        return 1;
      }

      public static int overloadFunction(int x) {
        return x;
      }
    }

    // Invoke overloaded functions.
    Assert.assertEquals((int) Ray.task((RayFunc0<Integer>) MyRayApp::overloadFunction).remote().get(), 1);
    Assert.assertEquals((int) Ray.task((RayFunc1<Integer, Integer>) MyRayApp::overloadFunction, 2).remote().get(), 2);

Overloaded actor task call:

.. code:: java

    public static class Counter {
      protected int value = 0;

      public int increment() {
        this.value += 1;
        return this.value;
      }
    }

    public static class CounterOverloaded extends Counter {
      public int increment(int diff) {
        super.value += diff;
        return super.value;
      }

      public int increment(int diff1, int diff2) {
        super.value += diff1 + diff2;
        return super.value;
      }
    }

.. code:: java

    ActorHandle<CounterOverloaded> a = Ray.actor(CounterOverloaded::new).remote();
    // Call an overloaded actor method by super class method reference.
    Assert.assertEquals((int) a.task(Counter::increment).remote().get(), 1);
    // Call an overloaded actor method, cast method reference first.
    a.task((RayFunc1<CounterOverloaded, Integer>) CounterOverloaded::increment).remote();
    a.task((RayFunc2<CounterOverloaded, Integer, Integer>) CounterOverloaded::increment, 10).remote();
    a.task((RayFunc3<CounterOverloaded, Integer, Integer, Integer>) CounterOverloaded::increment, 10, 10).remote();
    Assert.assertEquals((int) a.task(Counter::increment).remote().get(), 33);

Inspecting Cluster State
------------------------

Applications written on top of Ray will often want to have some information
or diagnostics about the cluster. Some common questions include:

    1. How many nodes are in my autoscaling cluster?
    2. What resources are currently available in my cluster, both used and total?
    3. What are the objects currently in my cluster?

For this, you can use the global state API.

Node Information
~~~~~~~~~~~~~~~~

To get information about the current nodes in your cluster, you can use ``ray.nodes()``:

.. autofunction:: ray.nodes
    :noindex:

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

    import ray

    ray.init()
    print(ray.nodes())

.. testoutput::
  :options: +MOCK

    [{'NodeID': '2691a0c1aed6f45e262b2372baf58871734332d7',
      'Alive': True,
      'NodeManagerAddress': '192.168.1.82',
      'NodeManagerHostname': 'host-MBP.attlocal.net',
      'NodeManagerPort': 58472,
      'ObjectManagerPort': 52383,
      'ObjectStoreSocketName': '/tmp/ray/session_2020-08-04_11-00-17_114725_17883/sockets/plasma_store',
      'RayletSocketName': '/tmp/ray/session_2020-08-04_11-00-17_114725_17883/sockets/raylet',
      'MetricsExportPort': 64860,
      'alive': True,
      'Resources': {'CPU': 16.0, 'memory': 100.0, 'object_store_memory': 34.0, 'node:192.168.1.82': 1.0}}]

The above information includes:

  - `NodeID`: A unique identifier for the raylet.
  - `alive`: Whether the node is still alive.
  - `NodeManagerAddress`: PrivateIP of the node that the raylet is on.
  - `Resources`: The total resource capacity on the node.
  - `MetricsExportPort`: The port number at which metrics are exposed to through a `Prometheus endpoint <ray-metrics.html>`_.

Resource Information
~~~~~~~~~~~~~~~~~~~~

To get information about the current total resource capacity of your cluster, you can use ``ray.cluster_resources()``.

.. autofunction:: ray.cluster_resources
    :noindex:


To get information about the current available resource capacity of your cluster, you can use ``ray.available_resources()``.

.. autofunction:: ray.available_resources
    :noindex:

Running Large Ray Clusters
--------------------------

Here are some tips to run Ray with more than 1k nodes. When running Ray with such
a large number of nodes, several system settings may need to be tuned to enable
communication between such a large number of machines.

Tuning Operating System Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because all nodes and workers connect to the GCS, many network connections will
be created and the operating system has to support that number of connections.

Maximum open files
******************

The OS has to be configured to support opening many TCP connections since every
worker and raylet connects to the GCS. In POSIX systems, the current limit can
be checked by ``ulimit -n`` and if it's small, it should be increased according to
the OS manual.

ARP cache
*********

Another thing that needs to be configured is the ARP cache. In a large cluster,
all the worker nodes connect to the head node, which adds a lot of entries to
the ARP table. Ensure that the ARP cache size is large enough to handle this
many nodes.
Failure to do this will result in the head node hanging. When this happens,
``dmesg`` will show errors like ``neighbor table overflow message``.

In Ubuntu, the ARP cache size can be tuned in ``/etc/sysctl.conf`` by increasing
the value of ``net.ipv4.neigh.default.gc_thresh1`` - ``net.ipv4.neigh.default.gc_thresh3``.
For more details, please refer to the OS manual.

Tuning Ray Settings
~~~~~~~~~~~~~~~~~~~

.. note::
  There is an ongoing `project <https://github.com/ray-project/ray/projects/15>`_ focusing on
  improving Ray's scalability and stability. Feel free to share your thoughts and use cases.

To run a large cluster, several parameters need to be tuned in Ray.

Benchmark
~~~~~~~~~

The machine setup:

- 1 head node: m5.4xlarge (16 vCPUs/64GB mem)
- 2000 worker nodes: m5.large (2 vCPUs/8GB mem)

The OS setup:

- Set the maximum number of opening files to 1048576
- Increase the ARP cache size:
    - ``net.ipv4.neigh.default.gc_thresh1=2048``
    - ``net.ipv4.neigh.default.gc_thresh2=4096``
    - ``net.ipv4.neigh.default.gc_thresh3=8192``


The Ray setup:

- ``RAY_event_stats=false``

Test workload:

- Test script: `code <https://github.com/ray-project/ray/blob/master/release/benchmarks/distributed/many_nodes_tests/actor_test.py>`_



.. list-table:: Benchmark result
   :header-rows: 1

   * - Number of actors
     - Actor launch time
     - Actor ready time
     - Total time
   * - 20k (10 actors / node)
     - 14.5s
     - 136.1s
     - 150.7s


.. _configuring-ray:

Configuring Ray
===============

.. note:: For running Java applications, please see `Java Applications`_.

This page discusses the various way to configure Ray, both from the Python API
and from the command line. Take a look at the ``ray.init`` `documentation
<package-ref.html#ray.init>`__ for a complete overview of the configurations.

.. important:: For the multi-node setting, you must first run ``ray start`` on the command line to start the Ray cluster services on the machine before ``ray.init`` in Python to connect to the cluster services. On a single machine, you can run ``ray.init()`` without ``ray start``, which will both start the Ray cluster services and connect to them.


.. _cluster-resources:

Cluster Resources
-----------------

Ray by default detects available resources.

.. testcode::
  :hide:

  import ray
  ray.shutdown()

.. testcode::

  import ray

  # This automatically detects available resources in the single machine.
  ray.init()

If not running cluster mode, you can specify cluster resources overrides through ``ray.init`` as follows.

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

  # If not connecting to an existing cluster, you can specify resources overrides:
  ray.init(num_cpus=8, num_gpus=1)

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

  # Specifying custom resources
  ray.init(num_gpus=1, resources={'Resource1': 4, 'Resource2': 16})

When starting Ray from the command line, pass the ``--num-cpus`` and ``--num-gpus`` flags into ``ray start``. You can also specify custom resources.

.. code-block:: bash

  # To start a head node.
  $ ray start --head --num-cpus=<NUM_CPUS> --num-gpus=<NUM_GPUS>

  # To start a non-head node.
  $ ray start --address=<address> --num-cpus=<NUM_CPUS> --num-gpus=<NUM_GPUS>

  # Specifying custom resources
  ray start [--head] --num-cpus=<NUM_CPUS> --resources='{"Resource1": 4, "Resource2": 16}'

If using the command line, connect to the Ray cluster as follow:

.. testcode::
  :skipif: True

  # Connect to ray. Notice if connected to existing cluster, you don't specify resources.
  ray.init(address=<address>)

.. _omp-num-thread-note:

.. note::
    Ray sets the environment variable ``OMP_NUM_THREADS=<num_cpus>`` if ``num_cpus`` is set on
    the task/actor via :func:`ray.remote() <ray.remote>` and :meth:`task.options() <ray.remote_function.RemoteFunction.options>`/:meth:`actor.options() <ray.actor.ActorClass.options>`.
    Ray sets ``OMP_NUM_THREADS=1`` if ``num_cpus`` is not specified; this
    is done to avoid performance degradation with many workers (issue #6998). You can
    also override this by explicitly setting ``OMP_NUM_THREADS`` to override anything Ray sets by default.
    ``OMP_NUM_THREADS`` is commonly used in numpy, PyTorch, and Tensorflow to perform multi-threaded
    linear algebra. In multi-worker setting, we want one thread per worker instead of many threads
    per worker to avoid contention. Some other libraries may have their own way to configure
    parallelism. For example, if you're using OpenCV, you should manually set the number of
    threads using cv2.setNumThreads(num_threads) (set to 0 to disable multi-threading).


.. _temp-dir-log-files:

Logging and Debugging
---------------------

Each Ray session will have a unique name. By default, the name is
``session_{timestamp}_{pid}``. The format of ``timestamp`` is
``%Y-%m-%d_%H-%M-%S_%f`` (See `Python time format <strftime.org>`__ for details);
the pid belongs to the startup process (the process calling ``ray.init()`` or
the Ray process executed by a shell in ``ray start``).

For each session, Ray will place all its temporary files under the
*session directory*. A *session directory* is a subdirectory of the
*root temporary path* (``/tmp/ray`` by default),
so the default session directory is ``/tmp/ray/{ray_session_name}``.
You can sort by their names to find the latest session.

Change the *root temporary directory* by passing ``--temp-dir={your temp path}`` to ``ray start``.

(There is not currently a stable way to change the root temporary directory when calling ``ray.init()``, but if you need to, you can provide the ``_temp_dir`` argument to ``ray.init()``.)

Look :ref:`Logging Directory Structure <logging-directory-structure>` for more details.

.. _ray-ports:

Ports configurations
--------------------
Ray requires bi-directional communication among its nodes in a cluster. Each node opens specific ports to receive incoming network requests.

All Nodes
~~~~~~~~~
- ``--node-manager-port``: Raylet port for node manager. Default: Random value.
- ``--object-manager-port``: Raylet port for object manager. Default: Random value.
- ``--runtime-env-agent-port``: Raylet port for runtime env agent. Default: Random value.

The node manager and object manager run as separate processes with their own ports for communication.

The following options specify the ports used by dashboard agent process.

- ``--dashboard-agent-grpc-port``: The port to listen for grpc on. Default: Random value.
- ``--dashboard-agent-listen-port``: The port to listen for http on. Default: 52365.
- ``--metrics-export-port``: The port to use to expose Ray metrics. Default: Random value.

The following options specify the range of ports used by worker processes across machines. All ports in the range should be open.

- ``--min-worker-port``: Minimum port number worker can be bound to. Default: 10002.
- ``--max-worker-port``: Maximum port number worker can be bound to. Default: 19999.

Port numbers are how Ray disambiguates input and output to and from multiple workers on a single node. Each worker will take input and give output on a single port number. Thus, for example, by default, there is a maximum of 10,000 workers on each node, irrespective of number of CPUs.

In general, it is recommended to give Ray a wide range of possible worker ports, in case any of those ports happen to be in use by some other program on your machine. However, when debugging it is useful to explicitly specify a short list of worker ports such as ``--worker-port-list=10000,10001,10002,10003,10004`` (note that this will limit the number of workers, just like specifying a narrow range).

Head Node
~~~~~~~~~
In addition to ports specified above, the head node needs to open several more ports.

- ``--port``: Port of Ray (GCS server). The head node will start a GCS server listening on this port. Default: 6379.
- ``--ray-client-server-port``: Listening port for Ray Client Server. Default: 10001.
- ``--redis-shard-ports``: Comma-separated list of ports for non-primary Redis shards. Default: Random values.
- ``--dashboard-grpc-port``: The gRPC port used by the dashboard. Default: Random value.

- If ``--include-dashboard`` is true (the default), then the head node must open ``--dashboard-port``. Default: 8265.

If ``--include-dashboard`` is true but the ``--dashboard-port`` is not open on
the head node, you will repeatedly get

.. code-block:: bash

  WARNING worker.py:1114 -- The agent on node <hostname of node that tried to run a task> failed with the following error:
  Traceback (most recent call last):
    File "/usr/local/lib/python3.8/dist-packages/grpc/aio/_call.py", line 285, in __await__
      raise _create_rpc_error(self._cython_call._initial_metadata,
  grpc.aio._call.AioRpcError: <AioRpcError of RPC that terminated with:
    status = StatusCode.UNAVAILABLE
    details = "failed to connect to all addresses"
    debug_error_string = "{"description":"Failed to pick subchannel","file":"src/core/ext/filters/client_channel/client_channel.cc","file_line":4165,"referenced_errors":[{"description":"failed to connect to all addresses","file":"src/core/ext/filters/client_channel/lb_policy/pick_first/pick_first.cc","file_line":397,"grpc_status":14}]}"

(Also, you will not be able to access the dashboard.)

If you see that error, check whether the ``--dashboard-port`` is accessible
with ``nc`` or ``nmap`` (or your browser).

.. code-block:: bash

  $ nmap -sV --reason -p 8265 $HEAD_ADDRESS
  Nmap scan report for compute04.berkeley.edu (123.456.78.910)
  Host is up, received reset ttl 60 (0.00065s latency).
  rDNS record for 123.456.78.910: compute04.berkeley.edu
  PORT     STATE SERVICE REASON         VERSION
  8265/tcp open  http    syn-ack ttl 60 aiohttp 3.7.2 (Python 3.8)
  Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .

Note that the dashboard runs as a separate subprocess which can crash invisibly
in the background, so even if you checked port 8265 earlier, the port might be
closed *now* (for the prosaic reason that there is no longer a service running
on it). This also means that if that port is unreachable, if you ``ray stop``
and ``ray start``, it may become reachable again due to the dashboard
restarting.

If you don't want the dashboard, set ``--include-dashboard=false``.

TLS Authentication
------------------

Ray can be configured to use TLS on it's gRPC channels.
This means that connecting to the Ray head requires
an appropriate set of credentials and also that data exchanged between
various processes (client, head, workers) is encrypted.

In TLS, the private key and public key are used for encryption and decryption. The
former is kept secret by the owner and the latter is shared with the other party.
This pattern ensures that only the intended recipient can read the message.

A Certificate Authority (CA) is a trusted third party that certifies the identity of the
public key owner. The digital certificate issued by the CA contains the public key itself,
the identity of the public key owner, and the expiration date of the certificate. Note that
if the owner of the public key does not want to obtain a digital certificate from a CA,
they can generate a self-signed certificate with some tools like OpenSSL.

To obtain a digital certificate, the owner of the public key must generate a Certificate Signing
Request (CSR). The CSR contains information about the owner of the public
key and the public key itself. For Ray, some additional steps are required for achieving
a successful TLS encryption.

Here is a step-by-step guide for adding TLS Authentication to a static Kubernetes Ray cluster using
a self-signed certificates:

Step 1: Generate a private key and self-signed certificate for CA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  openssl req -x509 \
              -sha256 -days 3650 \
              -nodes \
              -newkey rsa:2048 \
              -subj "/CN=*.ray.io/C=US/L=San Francisco" \
              -keyout ca.key -out ca.crt

Use the following command to encode the private key file and the self-signed certificate file,
then paste encoded strings to the secret.yaml.

.. code-block:: bash

  cat ca.key | base64
  cat ca.crt | base64

# Alternatively, the command automatically encode and create the secret for the CA keypair.
kubectl create secret generic ca-tls --from-file=ca.crt=<path-to-ca.crt> --from-file=ca.key=<path-to-ca.key>

Step 2: Generate individual private keys and self-signed certificates for the Ray head and workers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `YAML file
<https://raw.githubusercontent.com/ray-project/ray/master/doc/source/cluster/kubernetes/configs/static-ray-cluster.tls.yaml>`__, has a ConfigMap named `tls` that
includes two shell scripts: `gencert_head.sh` and `gencert_worker.sh`. These scripts produce the private key
and self-signed certificate files (`tls.key` and `tls.crt`) for both head and worker Pods in the initContainer
of each deployment. By using the initContainer, we can dynamically retrieve the `POD_IP` to the `[alt_names]` section.

The scripts perform the following steps: first, a 2048-bit RSA private key is generated and saved as
`/etc/ray/tls/tls.key`. Then, a Certificate Signing Request (CSR) is generated using the `tls.key` file
and the `csr.conf` configuration file. Finally, a self-signed certificate (`tls.crt`) is created using
the Certificate Authority's (`ca.key and ca.crt`) keypair and the CSR (`ca.csr`).

Step 3: Set the environment variables for both Ray head and worker to enable TLS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TLS is enabled by setting environment variables.

- ``RAY_USE_TLS``: Either 1 or 0 to use/not-use TLS. If this is set to 1 then all of the environment variables below must be set. Default: 0.
- ``RAY_TLS_SERVER_CERT``: Location of a `certificate file (tls.crt)`, which is presented to other endpoints to achieve mutual authentication.
- ``RAY_TLS_SERVER_KEY``: Location of a `private key file (tls.key)`, which is the cryptographic means to prove to other endpoints that you are the authorized user of a given certificate.
- ``RAY_TLS_CA_CERT``: Location of a `CA certificate file (ca.crt)`, which allows TLS to decide whether an endpoint's certificate has been signed by the correct authority.

Step 4: Verify TLS authentication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  # Log in to the worker Pod
  kubectl exec -it ${WORKER_POD} -- bash

  # Since the head Pod has the certificate of the full qualified DNS resolution for the Ray head service, the connection to the worker Pods
  # is established successfully
  ray health-check --address service-ray-head.default.svc.cluster.local:6379

  # Since service-ray-head hasn't added to the alt_names section in the certificate, the connection fails and an error
  # message similar to the following is displayed: "Peer name service-ray-head is not in peer certificate".
  ray health-check --address service-ray-head:6379

  # After you add `DNS.3 = service-ray-head` to the alt_names sections and deploy the YAML again, the connection is able to work.


Enabling TLS causes a performance hit due to the extra overhead of mutual
authentication and encryption.
Testing has shown that this overhead is large for small workloads and becomes
relatively smaller for large workloads.
The exact overhead depends on the nature of your workload.

Java Applications
-----------------

.. important:: For the multi-node setting, you must first run ``ray start`` on the command line to start the Ray cluster services on the machine before ``Ray.init()`` in Java to connect to the cluster services. On a single machine, you can run ``Ray.init()`` without ``ray start``, which will both start the Ray cluster services and connect to them.

.. _code_search_path:

Code Search Path
~~~~~~~~~~~~~~~~

If you want to run a Java application in a multi-node cluster, you must specify the code search path in your driver. The code search path is to tell Ray where to load jars when starting Java workers. Your jar files must be distributed to the same path(s) on all nodes of the Ray cluster before running your code.

.. code-block:: bash

  $ java -classpath <classpath> \
      -Dray.address=<address> \
      -Dray.job.code-search-path=/path/to/jars/ \
      <classname> <args>

The ``/path/to/jars/`` here points to a directory which contains jars. All jars in the directory will be loaded by workers. You can also provide multiple directories for this parameter.

.. code-block:: bash

  $ java -classpath <classpath> \
      -Dray.address=<address> \
      -Dray.job.code-search-path=/path/to/jars1:/path/to/jars2:/path/to/pys1:/path/to/pys2 \
      <classname> <args>

You don't need to configure code search path if you run a Java application in a single-node cluster.

See ``ray.job.code-search-path`` under :ref:`Driver Options <java-driver-options>` for more information.

.. note:: Currently we don't provide a way to configure Ray when running a Java application in single machine mode. If you need to configure Ray, run ``ray start`` to start the Ray cluster first.

.. _java-driver-options:

Driver Options
~~~~~~~~~~~~~~

There is a limited set of options for Java drivers. They are not for configuring the Ray cluster, but only for configuring the driver.

Ray uses `Typesafe Config <https://lightbend.github.io/config/>`__ to read options. There are several ways to set options:

- System properties. You can configure system properties either by adding options in the format of ``-Dkey=value`` in the driver command line, or by invoking ``System.setProperty("key", "value");`` before ``Ray.init()``.
- A `HOCON format <https://github.com/lightbend/config/blob/master/HOCON.md>`__ configuration file. By default, Ray will try to read the file named ``ray.conf`` in the root of the classpath. You can customize the location of the file by setting system property ``ray.config-file`` to the path of the file.

.. note:: Options configured by system properties have higher priority than options configured in the configuration file.

The list of available driver options:

- ``ray.address``

  - The cluster address if the driver connects to an existing Ray cluster. If it is empty, a new Ray cluster will be created.
  - Type: ``String``
  - Default: empty string.

- ``ray.job.code-search-path``

  - The paths for Java workers to load code from. Currently only directories are supported. You can specify one or more directories split by a ``:``. You don't need to configure code search path if you run a Java application in single machine mode or local mode. Code search path is also used for loading Python code if it's specified. This is required for :ref:`cross_language`. If code search path is specified, you can only run Python remote functions which can be found in the code search path.
  - Type: ``String``
  - Default: empty string.
  - Example: ``/path/to/jars1:/path/to/jars2:/path/to/pys1:/path/to/pys2``

- ``ray.job.namespace``

  - The namespace of this job. It's used for isolation between jobs. Jobs in different namespaces cannot access each other. If it's not specified, a randomized value will be used instead.
  - Type: ``String``
  - Default: A random UUID string value.

.. _`Apache Arrow`: https://arrow.apache.org/


.. _core-walkthrough:

What is Ray Core?
=================

.. toctree::
    :maxdepth: 1
    :hidden:

    Key Concepts <key-concepts>
    User Guides <user-guide>
    Examples <examples/overview>
    api/index


Ray Core provides a small number of core primitives (i.e., tasks, actors, objects) for building and scaling distributed applications. Below we'll walk through simple examples that show you how to turn your functions and classes easily into Ray tasks and actors, and how to work with Ray objects.

Getting Started
---------------

To get started, install Ray via ``pip install -U ray``. See :ref:`Installing Ray <installation>` for more installation options. The following few sections will walk through the basics of using Ray Core.

The first step is to import and initialize Ray:

.. literalinclude:: doc_code/getting_started.py
    :language: python
    :start-after: __starting_ray_start__
    :end-before: __starting_ray_end__

.. note::

  In recent versions of Ray (>=1.5), ``ray.init()`` is automatically called on the first use of a Ray remote API.

Running a Task
--------------

Ray lets you run functions as remote tasks in the cluster. To do this, you decorate your function with ``@ray.remote`` to declare that you want to run this function remotely.
Then, you call that function with ``.remote()`` instead of calling it normally.
This remote call returns a future, a so-called Ray *object reference*, that you can then fetch with ``ray.get``:

.. literalinclude:: doc_code/getting_started.py
    :language: python
    :start-after: __running_task_start__
    :end-before: __running_task_end__

Calling an Actor
----------------

Ray provides actors to allow you to parallelize computation across multiple actor instances. When you instantiate a class that is a Ray actor, Ray will start a remote instance of that class in the cluster. This actor can then execute remote method calls and maintain its own internal state:

.. literalinclude:: doc_code/getting_started.py
    :language: python
    :start-after: __calling_actor_start__
    :end-before: __calling_actor_end__

The above covers very basic actor usage. For a more in-depth example, including using both tasks and actors together, check out :ref:`monte-carlo-pi`.

Passing an Object
-----------------

As seen above, Ray stores task and actor call results in its :ref:`distributed object store <objects-in-ray>`, returning *object references* that can be later retrieved. Object references can also be created explicitly via ``ray.put``, and object references can be passed to tasks as substitutes for argument values:

.. literalinclude:: doc_code/getting_started.py
    :language: python
    :start-after: __passing_object_start__
    :end-before: __passing_object_end__

Next Steps
----------

.. tip:: To check how your application is doing, you can use the :ref:`Ray dashboard <observability-getting-started>`.

Ray's key primitives are simple, but can be composed together to express almost any kind of distributed computation.
Learn more about Ray's :ref:`key concepts <core-key-concepts>` with the following user guides:

.. grid:: 1 2 3 3
    :gutter: 1
    :class-container: container pb-3


    .. grid-item-card::
        :img-top: /images/tasks.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: ray-remote-functions

            Using remote functions (Tasks)

    .. grid-item-card::
        :img-top: /images/actors.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: ray-remote-classes

            Using remote classes (Actors)

    .. grid-item-card::
        :img-top: /images/objects.png
        :class-img-top: pt-2 w-75 d-block mx-auto fixed-height-img

        .. button-ref:: objects-in-ray

            Working with Ray Objects


.. _fault-tolerance:

Fault Tolerance
===============

Ray is a distributed system, and that means failures can happen. Generally, failures can
be classified into two classes: 1) application-level failures, and 2)
system-level failures.  The former can happen because of bugs in user-level
code, or if external systems fail. The latter can be triggered by node
failures, network failures, or just bugs in Ray. Here, we describe the
mechanisms that Ray provides to allow applications to recover from failures.

To handle application-level failures, Ray provides mechanisms to catch errors,
retry failed code, and handle misbehaving code. See the pages for :ref:`task
<fault-tolerance-tasks>` and :ref:`actor <fault-tolerance-actors>` fault
tolerance for more information on these mechanisms.

Ray also provides several mechanisms to automatically recover from internal system-level failures like :ref:`node failures <fault-tolerance-nodes>`.
In particular, Ray can automatically recover from some failures in the :ref:`distributed object store <fault-tolerance-objects>`.

How to Write Fault Tolerant Ray Applications
--------------------------------------------

There are several recommendations to make Ray applications fault tolerant:

First, if the fault tolerance mechanisms provided by Ray don't work for you,
you can always catch :ref:`exceptions <ray-core-exceptions>` caused by failures and recover manually.

.. literalinclude:: doc_code/fault_tolerance_tips.py
    :language: python
    :start-after: __manual_retry_start__
    :end-before: __manual_retry_end__

Second, avoid letting an ``ObjectRef`` outlive its :ref:`owner <fault-tolerance-objects>` task or actor
(the task or actor that creates the initial ``ObjectRef`` by calling :meth:`ray.put() <ray.put>` or ``foo.remote()``).
As long as there are still references to an object,
the owner worker of the object keeps running even after the corresponding task or actor finishes.
If the owner worker fails, Ray :ref:`cannot recover <fault-tolerance-ownership>` the object automatically for those who try to access the object.
One example of creating such outlived objects is returning ``ObjectRef`` created by ``ray.put()`` from a task:

.. literalinclude:: doc_code/fault_tolerance_tips.py
    :language: python
    :start-after: __return_ray_put_start__
    :end-before: __return_ray_put_end__

In the above example, object ``x`` outlives its owner task ``a``.
If the worker process running task ``a`` fails, calling ``ray.get`` on ``x_ref`` afterwards will result in an ``OwnerDiedError`` exception.

A fault tolerant version is returning ``x`` directly so that it is owned by the driver and it's only accessed within the lifetime of the driver.
If ``x`` is lost, Ray can automatically recover it via :ref:`lineage reconstruction <fault-tolerance-objects-reconstruction>`.
See :doc:`/ray-core/patterns/return-ray-put` for more details.

.. literalinclude:: doc_code/fault_tolerance_tips.py
    :language: python
    :start-after: __return_directly_start__
    :end-before: __return_directly_end__

Third, avoid using :ref:`custom resource requirements <custom-resources>` that can only be satisfied by a particular node.
If that particular node fails, the running tasks or actors cannot be retried.

.. literalinclude:: doc_code/fault_tolerance_tips.py
    :language: python
    :start-after: __node_ip_resource_start__
    :end-before: __node_ip_resource_end__

If you prefer running a task on a particular node, you can use the :class:`NodeAffinitySchedulingStrategy <ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy>`.
It allows you to specify the affinity as a soft constraint so even if the target node fails, the task can still be retried on other nodes.

.. literalinclude:: doc_code/fault_tolerance_tips.py
    :language: python
    :start-after: _node_affinity_scheduling_strategy_start__
    :end-before: __node_affinity_scheduling_strategy_end__


More about Ray Fault Tolerance
------------------------------

.. toctree::
    :maxdepth: 1

    fault_tolerance/tasks.rst
    fault_tolerance/actors.rst
    fault_tolerance/objects.rst
    fault_tolerance/nodes.rst
    fault_tolerance/gcs.rst


.. _generators:

Ray Generators
==============

`Python generators <https://docs.python.org/3/howto/functional.html#generators>`_ are functions
that behave like iterators, yielding one value per iteration. Ray also supports the generators API.

Any generator function decorated with ``ray.remote`` becomes a Ray generator task.
Generator tasks stream outputs back to the caller before the task finishes.

.. code-block:: diff

    +import ray
     import time

     # Takes 25 seconds to finish.
    +@ray.remote
     def f():
         for i in range(5):
             time.sleep(5)
             yield i

    -for obj in f():
    +for obj_ref in f.remote():
         # Prints every 5 seconds and stops after 25 seconds.
    -    print(obj)
    +    print(ray.get(obj_ref))


The above Ray generator yields the output every 5 seconds 5 times.
With a normal Ray task, you have to wait 25 seconds to access the output.
With a Ray generator, the caller can access the object reference
before the task ``f`` finishes.

**The Ray generator is useful when**

- You want to reduce heap memory or object store memory usage by yielding and garbage collecting (GC) the output before the task finishes.
- You are familiar with the Python generator and want the equivalent programming models.

**Ray libraries use the Ray generator to support streaming use cases**

- :ref:`Ray Serve <rayserve>` uses Ray generators to support :ref:`streaming responses <serve-http-streaming-response>`.
- :ref:`Ray Data <data>` is a streaming data processing library, which uses Ray generators to control and reduce concurrent memory usages.

**Ray generator works with existing Ray APIs seamlessly**

- You can use Ray generators in both actor and non-actor tasks.
- Ray generators work with all actor execution models, including :ref:`threaded actors <threaded-actors>` and :ref:`async actors <async-actors>`.
- Ray generators work with built-in :ref:`fault tolerance features <fault-tolerance>` such as retry or lineage reconstruction.
- Ray generators work with Ray APIs such as :ref:`ray.wait <generators-wait>`, :ref:`ray.cancel <generators-cancel>`, etc.

Getting started
---------------
Define a Python generator function and decorate it with ``ray.remote``
to create a Ray generator.

.. literalinclude:: doc_code/streaming_generator.py
    :language: python
    :start-after: __streaming_generator_define_start__
    :end-before: __streaming_generator_define_end__

The Ray generator task returns an ``ObjectRefGenerator`` object, which is
compatible with generator and async generator APIs. You can access the
``next``, ``__iter__``, ``__anext__``, ``__aiter__`` APIs from the class.

Whenever a task invokes ``yield``, a corresponding output is ready and available from a generator as a Ray object reference.
You can call ``next(gen)`` to obtain an object reference.
If ``next`` has no more items to generate, it raises ``StopIteration``. If ``__anext__`` has no more items to generate, it raises
``StopAsyncIteration``

The ``next`` API blocks the thread until the task generates a next object reference with ``yield``.
Since the ``ObjectRefGenerator`` is just a Python generator, you can also use a for loop to
iterate object references.

If you want to avoid blocking a thread, you can either use asyncio or :ref:`ray.wait API <generators-wait>`.

.. literalinclude:: doc_code/streaming_generator.py
    :language: python
    :start-after: __streaming_generator_execute_start__
    :end-before: __streaming_generator_execute_end__

.. note::

    For a normal Python generator, a generator function is paused and resumed when ``next`` function is
    called on a generator. Ray eagerly executes a generator task to completion regardless of whether the caller is polling the partial results or not.

Error handling
--------------

If a generator task has a failure (by an application exception or system error such as an unexpected node failure),
the ``next(gen)`` returns an object reference that contains an exception. When you call ``ray.get``,
Ray raises the exception.

.. literalinclude:: doc_code/streaming_generator.py
    :language: python
    :start-after: __streaming_generator_exception_start__
    :end-before: __streaming_generator_exception_end__

In the above example, if the an application fails the task, Ray returns the object reference with an exception
in a correct order. For example, if Ray raises the exception after the second yield, the third
``next(gen)`` returns an object reference with an exception all the time. If a system error fails the task,
(e.g., a node failure or worker process failure), ``next(gen)`` returns the object reference that contains the system level exception
at any time without an ordering guarantee.
It means when you have N yields, the generator can create from 1 to N + 1 object references
(N output + ref with a system-level exception) when there failures occur.

Generator from Actor Tasks
--------------------------
The Ray generator is compatible with **all actor execution models**. It seamlessly works with
regular actors, :ref:`async actors <async-actors>`, and :ref:`threaded actors <threaded-actors>`.

.. literalinclude:: doc_code/streaming_generator.py
    :language: python
    :start-after: __streaming_generator_actor_model_start__
    :end-before: __streaming_generator_actor_model_end__

Using the Ray generator with asyncio
------------------------------------
The returned ``ObjectRefGenerator`` is also compatible with asyncio. You can
use ``__anext__`` or ``async for`` loops.

.. literalinclude:: doc_code/streaming_generator.py
    :language: python
    :start-after: __streaming_generator_asyncio_start__
    :end-before: __streaming_generator_asyncio_end__

Garbage collection of object references
---------------------------------------
The returned ref from ``next(generator)`` is just a regular Ray object reference and is distributed ref counted in the same way.
If references are not consumed from a generator by the ``next`` API, references are garbage collected (GC’ed) when the generator is GC’ed.

.. literalinclude:: doc_code/streaming_generator.py
    :language: python
    :start-after: __streaming_generator_gc_start__
    :end-before: __streaming_generator_gc_end__

In the following example, Ray counts ``ref1`` as a normal Ray object reference after Ray returns it. Other references
that aren't consumed with ``next(gen)`` are removed when the generator is GC'ed. In this example, garbage collection happens when you call ``del gen``.

Fault tolerance
---------------
:ref:`Fault tolerance features <fault-tolerance>` work with
Ray generator tasks and actor tasks. For example;

- :ref:`Task fault tolerance features <task-fault-tolerance>`: ``max_retries``, ``retry_exceptions``
- :ref:`Actor fault tolerance features <actor-fault-tolerance>`: ``max_restarts``, ``max_task_retries``
- :ref:`Object fault tolerance features <object-fault-tolerance>`: object reconstruction

.. _generators-cancel:

Cancellation
------------
The :func:`ray.cancel() <ray.cancel>` function works with both Ray generator tasks and actor tasks.
Semantic-wise, cancelling a generator task isn't different from cancelling a regular task.
When you cancel a task, ``next(gen)`` can return the reference that contains :class:`TaskCancelledError <ray.exceptions.TaskCancelledError>` without any special ordering guarantee.

.. _generators-wait:

How to wait for generator without blocking a thread (compatibility to ray.wait and ray.get)
-------------------------------------------------------------------------------------------
When using a generator, ``next`` API blocks its thread until a next object reference is available.
However, you may not want this behavior all the time. You may want to wait for a generator without blocking a thread.
Unblocking wait is possible with the Ray generator in the following ways:

**Wait until a generator task completes**

``ObjectRefGenerator`` has an API ``completed``. It returns an object reference that is available when a generator task finishes or errors.
For example, you can do ``ray.get(<generator_instance>.completed())`` to wait until a task completes. Note that using ``ray.get`` to ``ObjectRefGenerator`` isn't allowed.

**Use asyncio and await**

``ObjectRefGenerator`` is compatible with asyncio. You can create multiple asyncio tasks that create a generator task
and wait for it to avoid blocking a thread.

.. literalinclude:: doc_code/streaming_generator.py
    :language: python
    :start-after: __streaming_generator_concurrency_asyncio_start__
    :end-before: __streaming_generator_concurrency_asyncio_end__

**Use ray.wait**

You can pass ``ObjectRefGenerator`` as an input to ``ray.wait``. The generator is "ready" if a `next item`
is available. Once Ray finds from a ready list, ``next(gen)`` returns the next object reference immediately without blocking. See the example below for more details.

.. literalinclude:: doc_code/streaming_generator.py
    :language: python
    :start-after: __streaming_generator_wait_simple_start__
    :end-before: __streaming_generator_wait_simple_end__

All the input arguments (such as ``timeout``, ``num_returns``, and ``fetch_local``) from ``ray.wait`` works with a generator.

``ray.wait`` can mix regular Ray object references with generators for inputs. In this case, the application should handle
all input arguments (such as ``timeout``, ``num_returns``, and ``fetch_local``) from ``ray.wait`` work with generators.

.. literalinclude:: doc_code/streaming_generator.py
    :language: python
    :start-after: __streaming_generator_wait_complex_start__
    :end-before: __streaming_generator_wait_complex_end__

Thread safety
-------------
``ObjectRefGenerator`` object is not thread-safe.

Limitation
----------
Ray generators don't support these features:

- ``throw``, ``send``, and ``close`` APIs.
- ``return`` statements from generators.
- Passing ``ObjectRefGenerator`` to another task or actor.
- :ref:`Ray Client <ray-client-ref>`


Working with Jupyter Notebooks & JupyterLab
===========================================

This document describes best practices for using Ray with Jupyter Notebook / 
JupyterLab.
We use AWS for the purpose of illustration, but the arguments should also apply to
other Cloud providers.
Feel free to contribute if you think this document is missing anything.

Setting Up Notebook
-------------------

1. Ensure your EC2 instance has enough EBS volume if you plan to run the 
Notebook on it.
The Deep Learning AMI, pre-installed libraries and environmental set-up 
will by default consume ~76% of the disk prior to any Ray work.
With additional applications running, the Notebook could fail frequently
due to full disk. 
Kernel restart loses progressing cell outputs, especially if we rely on 
them to track experiment progress. 
Related issue: `Autoscaler should allow configuration of disk space and 
should use a larger default. <https://github.com/ray-project/ray/issues/1376>`_.

2. Avoid unnecessary memory usage.
IPython stores the output of every cell in a local Python variable
indefinitely. This causes Ray to pin the objects even though you application
may not actually be using them.
Therefore, explicitly calling ``print`` or ``repr`` is better than letting 
the Notebook automatically generate the output.
Another option is to just altogether disable IPython caching with the 
following (run from bash/zsh):

.. code-block:: console

    echo 'c = get_config()
    c.InteractiveShell.cache_size = 0 # disable cache
    ' >>  ~/.ipython/profile_default/ipython_config.py

This will still allow printing, but stop IPython from caching altogether.

.. tip::
  While the above settings help reduce memory footprint, it's always a good 
  practice to remove references that are no longer needed in your application
  to free space in the object store.

3. Understand the node’s responsibility. 
Assuming the Notebook runs on a EC2 instance,
do you plan to start a ray runtime locally on this instance,
or do you plan to use this instance as a cluster launcher? 
Jupyter Notebook is more suitable for the first scenario. 
CLI’s such as ``ray exec`` and ``ray submit`` fit the second use case better.

4. Forward the ports.
Assuming the Notebook runs on an EC2 instance,
you should forward both the Notebook port and the Ray Dashboard port.
The default ports are 8888 and 8265 respectively. 
They will increase if the default ones are not available.
You can forward them with the following (run from bash/zsh):

.. code-block:: console

    ssh -i /path/my-key-pair.pem -N -f -L localhost:8888:localhost:8888 my-instance-user-name@my-instance-IPv6-address
    ssh -i /path/my-key-pair.pem -N -f -L localhost:8265:localhost:8265 my-instance-user-name@my-instance-IPv6-address


.. _serialization-guide:

Serialization
=============

Since Ray processes do not share memory space, data transferred between workers and nodes will need to **serialized** and **deserialized**. Ray uses the `Plasma object store <https://arrow.apache.org/blog/2017/08/08/plasma-in-memory-object-store/>`_ to efficiently transfer objects across different processes and different nodes. Numpy arrays in the object store are shared between workers on the same node (zero-copy deserialization).

Overview
--------

Ray has decided to use a customized `Pickle protocol version 5 <https://www.python.org/dev/peps/pep-0574/>`_ backport to replace the original PyArrow serializer. This gets rid of several previous limitations (e.g. cannot serialize recursive objects).

Ray is currently compatible with Pickle protocol version 5, while Ray supports serialization of a wider range of objects (e.g. lambda & nested functions, dynamic classes) with the help of cloudpickle.

.. _plasma-store:

Plasma Object Store
~~~~~~~~~~~~~~~~~~~

Plasma is an in-memory object store. It has been originally developed as part of Apache Arrow. Prior to Ray's version 1.0.0 release, Ray forked Arrow's Plasma code into Ray's code base in order to disentangle and continue development with respect to Ray's architecture and performance needs.

Plasma is used to efficiently transfer objects across different processes and different nodes. All objects in Plasma object store are **immutable** and held in shared memory. This is so that they can be accessed efficiently by many workers on the same node.

Each node has its own object store. When data is put into the object store, it does not get automatically broadcasted to other nodes. Data remains local to the writer until requested by another task or actor on another node.

Serializing ObjectRefs
~~~~~~~~~~~~~~~~~~~~~~

Explicitly serializing `ObjectRefs` using `ray.cloudpickle` should be used as a last resort. Passing `ObjectRefs` through Ray task arguments and return values is the recommended approach.

Ray `ObjectRefs` can be serialized using `ray.cloudpickle`. The `ObjectRef` can then be deserialized and accessed with `ray.get()`. Note that `ray.cloudpickle` must be used; other pickle tools are not guaranteed to work. Additionally, the process that deserializes the `ObjectRef` must be part of the same Ray cluster that serialized it.

When serialized, the `ObjectRef`'s value will remain pinned in Ray's shared memory object store. The object must be explicitly freed by calling `ray._private.internal_api.free(obj_ref)`.

.. warning::
  
  `ray._private.internal_api.free(obj_ref)` is a private API and may be changed in future Ray versions.

This code example demonstrates how to serialize an `ObjectRef`, store it in external storage, deserialize and use it, and lastly free its object.

.. literalinclude:: /ray-core/doc_code/object_ref_serialization.py

Numpy Arrays
~~~~~~~~~~~~

Ray optimizes for numpy arrays by using Pickle protocol 5 with out-of-band data.
The numpy array is stored as a read-only object, and all Ray workers on the same node can read the numpy array in the object store without copying (zero-copy reads). Each numpy array object in the worker process holds a pointer to the relevant array held in shared memory. Any writes to the read-only object will require the user to first copy it into the local process memory.

.. tip:: You can often avoid serialization issues by using only native types (e.g., numpy arrays or lists/dicts of numpy arrays and other primitive types), or by using Actors hold objects that cannot be serialized.

Fixing "assignment destination is read-only"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because Ray puts numpy arrays in the object store, when deserialized as arguments in remote functions they will become read-only. For example, the following code snippet will crash:

.. literalinclude:: /ray-core/doc_code/deser.py

To avoid this issue, you can manually copy the array at the destination if you need to mutate it (``arr = arr.copy()``). Note that this is effectively like disabling the zero-copy deserialization feature provided by Ray.

Serialization notes
-------------------

- Ray is currently using Pickle protocol version 5. The default pickle protocol used by most python distributions is protocol 3. Protocol 4 & 5 are more efficient than protocol 3 for larger objects.

- For non-native objects, Ray will always keep a single copy even it is referred multiple times in an object:

  .. testcode::

    import ray
    import numpy as np

    obj = [np.zeros(42)] * 99
    l = ray.get(ray.put(obj))
    assert l[0] is l[1]  # no problem!

- Whenever possible, use numpy arrays or Python collections of numpy arrays for maximum performance.

- Lock objects are mostly unserializable, because copying a lock is meaningless and could cause serious concurrency problems. You may have to come up with a workaround if your object contains a lock.

Customized Serialization
------------------------

Sometimes you may want to customize your serialization process because
the default serializer used by Ray (pickle5 + cloudpickle) does
not work for you (fail to serialize some objects, too slow for certain objects, etc.).

There are at least 3 ways to define your custom serialization process:

1. If you want to customize the serialization of a type of objects,
   and you have access to the code, you can define ``__reduce__``
   function inside the corresponding class. This is commonly done
   by most Python libraries. Example code:

   .. testcode::

     import ray
     import sqlite3

     class DBConnection:
         def __init__(self, path):
             self.path = path
             self.conn = sqlite3.connect(path)

         # without '__reduce__', the instance is unserializable.
         def __reduce__(self):
             deserializer = DBConnection
             serialized_data = (self.path,)
             return deserializer, serialized_data

     original = DBConnection("/tmp/db")
     print(original.conn)

     copied = ray.get(ray.put(original))
     print(copied.conn)

  .. testoutput::

    <sqlite3.Connection object at ...>
    <sqlite3.Connection object at ...>


2. If you want to customize the serialization of a type of objects,
   but you cannot access or modify the corresponding class, you can
   register the class with the serializer you use:

   .. testcode::

      import ray
      import threading

      class A:
          def __init__(self, x):
              self.x = x
              self.lock = threading.Lock()  # could not be serialized!

      try:
        ray.get(ray.put(A(1)))  # fail!
      except TypeError:
        pass

      def custom_serializer(a):
          return a.x

      def custom_deserializer(b):
          return A(b)

      # Register serializer and deserializer for class A:
      ray.util.register_serializer(
        A, serializer=custom_serializer, deserializer=custom_deserializer)
      ray.get(ray.put(A(1)))  # success!

      # You can deregister the serializer at any time.
      ray.util.deregister_serializer(A)
      try:
        ray.get(ray.put(A(1)))  # fail!
      except TypeError:
        pass

      # Nothing happens when deregister an unavailable serializer.
      ray.util.deregister_serializer(A)

   NOTE: Serializers are managed locally for each Ray worker. So for every Ray worker,
   if you want to use the serializer, you need to register the serializer. Deregister
   a serializer also only applies locally.

   If you register a new serializer for a class, the new serializer would replace
   the old serializer immediately in the worker. This API is also idempotent, there are
   no side effects caused by re-registering the same serializer.

3. We also provide you an example, if you want to customize the serialization
   of a specific object:

   .. testcode::

     import threading

     class A:
         def __init__(self, x):
             self.x = x
             self.lock = threading.Lock()  # could not serialize!

     try:
        ray.get(ray.put(A(1)))  # fail!
     except TypeError:
        pass

     class SerializationHelperForA:
         """A helper class for serialization."""
         def __init__(self, a):
             self.a = a

         def __reduce__(self):
             return A, (self.a.x,)

     ray.get(ray.put(SerializationHelperForA(A(1))))  # success!
     # the serializer only works for a specific object, not all A
     # instances, so we still expect failure here.
     try:
        ray.get(ray.put(A(1)))  # still fail!
     except TypeError:
        pass


Troubleshooting
---------------

Use ``ray.util.inspect_serializability`` to identify tricky pickling issues. This function can be used to trace a potential non-serializable object within any Python object -- whether it be a function, class, or object instance.

Below, we demonstrate this behavior on a function with a non-serializable object (threading lock):

.. testcode::

    from ray.util import inspect_serializability
    import threading

    lock = threading.Lock()

    def test():
        print(lock)

    inspect_serializability(test, name="test")

The resulting output is:

.. testoutput::
  :options: +MOCK

    =============================================================
    Checking Serializability of <function test at 0x7ff130697e50>
    =============================================================
    !!! FAIL serialization: cannot pickle '_thread.lock' object
    Detected 1 global variables. Checking serializability...
        Serializing 'lock' <unlocked _thread.lock object at 0x7ff1306a9f30>...
        !!! FAIL serialization: cannot pickle '_thread.lock' object
        WARNING: Did not find non-serializable object in <unlocked _thread.lock object at 0x7ff1306a9f30>. This may be an oversight.
    =============================================================
    Variable:

    	FailTuple(lock [obj=<unlocked _thread.lock object at 0x7ff1306a9f30>, parent=<function test at 0x7ff130697e50>])

    was found to be non-serializable. There may be multiple other undetected variables that were non-serializable.
    Consider either removing the instantiation/imports of these variables or moving the instantiation into the scope of the function/class.
    =============================================================
    Check https://docs.ray.io/en/master/ray-core/objects/serialization.html#troubleshooting for more information.
    If you have any suggestions on how to improve this error message, please reach out to the Ray developers on github.com/ray-project/ray/issues/
    =============================================================

For even more detailed information, set environmental variable ``RAY_PICKLE_VERBOSE_DEBUG='2'`` before importing Ray. This enables
serialization with python-based backend instead of C-Pickle, so you can debug into python code at the middle of serialization.
However, this would make serialization much slower.

Known Issues
------------

Users could experience memory leak when using certain python3.8 & 3.9 versions. This is due to `a bug in python's pickle module <https://bugs.python.org/issue39492>`_.

This issue has been solved for Python 3.8.2rc1, Python 3.9.0 alpha 4 or late versions.


Object Spilling
===============
.. _object-spilling:

Ray 1.3+ spills objects to external storage once the object store is full. By default, objects are spilled to Ray's temporary directory in the local filesystem.

Single node
-----------

Ray uses object spilling by default. Without any setting, objects are spilled to `[temp_folder]/spill`. On Linux and MacOS, the `temp_folder` is `/tmp` by default.

To configure the directory where objects are spilled to, use:

.. testcode::
  :hide:

  import ray
  ray.shutdown()

.. testcode::

    import json
    import ray

    ray.init(
        _system_config={
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
            )
        },
    )

You can also specify multiple directories for spilling to spread the IO load and disk space
usage across multiple physical devices if needed (e.g., SSD devices):

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

    import json
    import ray

    ray.init(
        _system_config={
            "max_io_workers": 4,  # More IO workers for parallelism.
            "object_spilling_config": json.dumps(
                {
                  "type": "filesystem",
                  "params": {
                    # Multiple directories can be specified to distribute
                    # IO across multiple mounted physical devices.
                    "directory_path": [
                      "/tmp/spill",
                      "/tmp/spill_1",
                      "/tmp/spill_2",
                    ]
                  },
                }
            )
        },
    )


.. note::

    To optimize the performance, it is recommended to use an SSD instead of an HDD when using object spilling for memory-intensive workloads.

If you are using an HDD, it is recommended that you specify a large buffer size (> 1MB) to reduce IO requests during spilling.

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

    import json
    import ray

    ray.init(
        _system_config={
            "object_spilling_config": json.dumps(
                {
                  "type": "filesystem",
                  "params": {
                    "directory_path": "/tmp/spill",
                    "buffer_size": 1_000_000,
                  }
                },
            )
        },
    )

To prevent running out of disk space, local object spilling will throw ``OutOfDiskError`` if the disk utilization exceeds the predefined threshold.
If multiple physical devices are used, any physical device's over-usage will trigger the ``OutOfDiskError``.
The default threshold is 0.95 (95%). You can adjust the threshold by setting ``local_fs_capacity_threshold``, or set it to 1 to disable the protection.

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::

    import json
    import ray

    ray.init(
        _system_config={
            # Allow spilling until the local disk is 99% utilized.
            # This only affects spilling to the local file system.
            "local_fs_capacity_threshold": 0.99,
            "object_spilling_config": json.dumps(
                {
                  "type": "filesystem",
                  "params": {
                    "directory_path": "/tmp/spill",
                  }
                },
            )
        },
    )


To enable object spilling to remote storage (any URI supported by `smart_open <https://pypi.org/project/smart-open/>`__):

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::
  :skipif: True

    import json
    import ray

    ray.init(
        _system_config={
            "max_io_workers": 4,  # More IO workers for remote storage.
            "min_spilling_size": 100 * 1024 * 1024,  # Spill at least 100MB at a time.
            "object_spilling_config": json.dumps(
                {
                  "type": "smart_open",
                  "params": {
                    "uri": "s3://bucket/path"
                  },
                  "buffer_size": 100 * 1024 * 1024,  # Use a 100MB buffer for writes
                },
            )
        },
    )

It is recommended that you specify a large buffer size (> 1MB) to reduce IO requests during spilling.

Spilling to multiple remote storages is also supported.

.. testcode::
  :hide:

  ray.shutdown()

.. testcode::
  :skipif: True

    import json
    import ray

    ray.init(
        _system_config={
            "max_io_workers": 4,  # More IO workers for remote storage.
            "min_spilling_size": 100 * 1024 * 1024,  # Spill at least 100MB at a time.
            "object_spilling_config": json.dumps(
                {
                  "type": "smart_open",
                  "params": {
                    "uri": ["s3://bucket/path1", "s3://bucket/path2", "s3://bucket/path3"],
                  },
                  "buffer_size": 100 * 1024 * 1024, # Use a 100MB buffer for writes
                },
            )
        },
    )

Remote storage support is still experimental.

Cluster mode
------------
To enable object spilling in multi node clusters:

.. code-block:: bash

  # Note that `object_spilling_config`'s value should be json format.
  # You only need to specify the config when starting the head node, all the worker nodes will get the same config from the head node.
  ray start --head --system-config='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"/tmp/spill\"}}"}'

Stats
-----

When spilling is happening, the following INFO level messages will be printed to the raylet logs (e.g., ``/tmp/ray/session_latest/logs/raylet.out``)::

  local_object_manager.cc:166: Spilled 50 MiB, 1 objects, write throughput 230 MiB/s
  local_object_manager.cc:334: Restored 50 MiB, 1 objects, read throughput 505 MiB/s

You can also view cluster-wide spill stats by using the ``ray memory`` command::

  --- Aggregate object store stats across all nodes ---
  Plasma memory usage 50 MiB, 1 objects, 50.0% full
  Spilled 200 MiB, 4 objects, avg write throughput 570 MiB/s
  Restored 150 MiB, 3 objects, avg read throughput 1361 MiB/s

If you only want to display cluster-wide spill stats, use ``ray memory --stats-only``.


.. _core-resources:

Resources
=========

Ray allows you to seamlessly scale your applications from a laptop to a cluster without code change.
**Ray resources** are key to this capability.
They abstract away physical machines and let you express your computation in terms of resources,
while the system manages scheduling and autoscaling based on resource requests.

A resource in Ray is a key-value pair where the key denotes a resource name, and the value is a float quantity.
For convenience, Ray has native support for CPU, GPU, and memory resource types; CPU, GPU and memory are called **pre-defined resources**.
Besides those, Ray also supports :ref:`custom resources <custom-resources>`.

.. _logical-resources:

Physical Resources and Logical Resources
----------------------------------------

Physical resources are resources that a machine physically has such as physical CPUs and GPUs
and logical resources are virtual resources defined by a system.

Ray resources are **logical** and don’t need to have 1-to-1 mapping with physical resources.
For example, you can start a Ray head node with 0 logical CPUs via ``ray start --head --num-cpus=0``
even if it physically has eight
(This signals the Ray scheduler to not schedule any tasks or actors that require logical CPU resources
on the head node, mainly to reserve the head node for running Ray system processes.).
They are mainly used for admission control during scheduling.

The fact that resources are logical has several implications:

- Resource requirements of tasks or actors do NOT impose limits on actual physical resource usage.
  For example, Ray doesn't prevent a ``num_cpus=1`` task from launching multiple threads and using multiple physical CPUs.
  It's your responsibility to make sure tasks or actors use no more resources than specified via resource requirements.
- Ray doesn't provide CPU isolation for tasks or actors.
  For example, Ray won't reserve a physical CPU exclusively and pin a ``num_cpus=1`` task to it.
  Ray will let the operating system schedule and run the task instead.
  If needed, you can use operating system APIs like ``sched_setaffinity`` to pin a task to a physical CPU.
- Ray does provide :ref:`GPU <gpu-support>` isolation in the form of *visible devices* by automatically setting the ``CUDA_VISIBLE_DEVICES`` environment variable,
  which most ML frameworks will respect for purposes of GPU assignment.

.. figure:: ../images/physical_resources_vs_logical_resources.svg

  Physical resources vs logical resources

.. _custom-resources:

Custom Resources
----------------

Besides pre-defined resources, you can also specify a Ray node's custom resources and request them in your tasks or actors.
Some use cases for custom resources:

- Your node has special hardware and you can represent it as a custom resource.
  Then your tasks or actors can request the custom resource via ``@ray.remote(resources={"special_hardware": 1})``
  and Ray will schedule the tasks or actors to the node that has the custom resource.
- You can use custom resources as labels to tag nodes and you can achieve label based affinity scheduling.
  For example, you can do ``ray.remote(resources={"custom_label": 0.001})`` to schedule tasks or actors to nodes with ``custom_label`` custom resource.
  For this use case, the actual quantity doesn't matter, and the convention is to specify a tiny number so that the label resource is
  not the limiting factor for parallelism.

.. _specify-node-resources:

Specifying Node Resources
-------------------------

By default, Ray nodes start with pre-defined CPU, GPU, and memory resources. The quantities of these logical resources on each node are set to the physical quantities auto detected by Ray.
By default, logical resources are configured by the following rule.

.. warning::

    Ray **does not permit dynamic updates of resource capacities after Ray has been started on a node**.

- **Number of logical CPUs (``num_cpus``)**: Set to the number of CPUs of the machine/container.
- **Number of logical GPUs (``num_gpus``)**: Set to the number of GPUs of the machine/container.
- **Memory (``memory``)**: Set to 70% of "available memory" when ray runtime starts.
- **Object Store Memory (``object_store_memory``)**: Set to 30% of "available memory" when ray runtime starts. Note that the object store memory is not logical resource, and users cannot use it for scheduling.

However, you can always override that by manually specifying the quantities of pre-defined resources and adding custom resources.
There are several ways to do that depending on how you start the Ray cluster:

.. tab-set::

    .. tab-item:: ray.init()

        If you are using :func:`ray.init() <ray.init>` to start a single node Ray cluster, you can do the following to manually specify node resources:

        .. literalinclude:: ../doc_code/resources.py
            :language: python
            :start-after: __specifying_node_resources_start__
            :end-before: __specifying_node_resources_end__

    .. tab-item:: ray start

        If you are using :ref:`ray start <ray-start-doc>` to start a Ray node, you can run:

        .. code-block:: shell

            ray start --head --num-cpus=3 --num-gpus=4 --resources='{"special_hardware": 1, "custom_label": 1}'

    .. tab-item:: ray up

        If you are using :ref:`ray up <ray-up-doc>` to start a Ray cluster, you can set the :ref:`resources field <cluster-configuration-resources-type>` in the yaml file:

        .. code-block:: yaml

            available_node_types:
              head:
                ...
                resources:
                  CPU: 3
                  GPU: 4
                  special_hardware: 1
                  custom_label: 1

    .. tab-item:: KubeRay

        If you are using :ref:`KubeRay <kuberay-index>` to start a Ray cluster, you can set the :ref:`rayStartParams field <rayStartParams>` in the yaml file:

        .. code-block:: yaml

            headGroupSpec:
              rayStartParams:
                num-cpus: "3"
                num-gpus: "4"
                resources: '"{\"special_hardware\": 1, \"custom_label\": 1}"'


.. _resource-requirements:

Specifying Task or Actor Resource Requirements
----------------------------------------------

Ray allows specifying a task or actor's logical resource requirements (e.g., CPU, GPU, and custom resources).
The task or actor will only run on a node if there are enough required logical resources
available to execute the task or actor.

By default, Ray tasks use 1 logical CPU resource and Ray actors use 1 logical CPU for scheduling, and 0 logical CPU for running.
(This means, by default, actors cannot get scheduled on a zero-cpu node, but an infinite number of them can run on any non-zero cpu node.
The default resource requirements for actors was chosen for historical reasons.
It's recommended to always explicitly set ``num_cpus`` for actors to avoid any surprises.
If resources are specified explicitly, they are required for both scheduling and running.)

You can also explicitly specify a task's or actor's logical resource requirements (for example, one task may require a GPU) instead of using default ones via :func:`ray.remote() <ray.remote>`
and :meth:`task.options() <ray.remote_function.RemoteFunction.options>`/:meth:`actor.options() <ray.actor.ActorClass.options>`.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/resources.py
            :language: python
            :start-after: __specifying_resource_requirements_start__
            :end-before: __specifying_resource_requirements_end__

    .. tab-item:: Java

        .. code-block:: java

            // Specify required resources.
            Ray.task(MyRayApp::myFunction).setResource("CPU", 1.0).setResource("GPU", 1.0).setResource("special_hardware", 1.0).remote();

            Ray.actor(Counter::new).setResource("CPU", 2.0).setResource("GPU", 1.0).remote();

    .. tab-item:: C++

        .. code-block:: c++

            // Specify required resources.
            ray::Task(MyFunction).SetResource("CPU", 1.0).SetResource("GPU", 1.0).SetResource("special_hardware", 1.0).Remote();

            ray::Actor(CreateCounter).SetResource("CPU", 2.0).SetResource("GPU", 1.0).Remote();

Task and actor resource requirements have implications for the Ray's scheduling concurrency.
In particular, the sum of the logical resource requirements of all of the
concurrently executing tasks and actors on a given node cannot exceed the node's total logical resources.
This property can be used to :ref:`limit the number of concurrently running tasks or actors to avoid issues like OOM <core-patterns-limit-running-tasks>`.

.. _fractional-resource-requirements:

Fractional Resource Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray supports fractional resource requirements.
For example, if your task or actor is IO bound and has low CPU usage, you can specify fractional CPU ``num_cpus=0.5`` or even zero CPU ``num_cpus=0``.
The precision of the fractional resource requirement is 0.0001 so you should avoid specifying a double that's beyond that precision.

.. literalinclude:: ../doc_code/resources.py
    :language: python
    :start-after: __specifying_fractional_resource_requirements_start__
    :end-before: __specifying_fractional_resource_requirements_end__

.. note::

  GPU, TPU, and neuron_cores resource requirements that are greater than 1, need to be whole numbers. For example, ``num_gpus=1.5`` is invalid.

.. tip::

  Besides resource requirements, you can also specify an environment for a task or actor to run in,
  which can include Python packages, local files, environment variables, and more. See :ref:`Runtime Environments <runtime-environments>` for details.

.. _gpu-support:
.. _accelerator-support:

Accelerator Support
===================

Accelerators (e.g. GPUs) are critical for many machine learning applications.
Ray Core natively supports many accelerators as pre-defined :ref:`resource <core-resources>` types and allows tasks and actors to specify their accelerator :ref:`resource requirements <resource-requirements>`.

The accelerators natively supported by Ray Core are:

.. list-table::
   :header-rows: 1

   * - Accelerator
     - Ray Resource Name
     - Support Level
   * - Nvidia GPU
     - GPU
     - Fully tested, supported by the Ray team
   * - AMD GPU
     - GPU
     - Experimental, supported by the community
   * - Intel GPU
     - GPU
     - Experimental, supported by the community
   * - `AWS Neuron Core <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/model-architecture-fit.html>`_
     - neuron_cores
     - Experimental, supported by the community
   * - Google TPU
     - TPU
     - Experimental, supported by the community
   * - Intel Gaudi
     - HPU
     - Experimental, supported by the community
   * - Huawei Ascend
     - NPU
     - Experimental, supported by the community

Starting Ray Nodes with Accelerators
------------------------------------

By default, Ray sets the quantity of accelerator resources of a node to the physical quantities of accelerators auto detected by Ray.
If you need to, you can :ref:`override <specify-node-resources>` this.

.. tab-set::

    .. tab-item:: Nvidia GPU
        :sync: Nvidia GPU

        .. tip::

            You can set the ``CUDA_VISIBLE_DEVICES`` environment variable before starting a Ray node
            to limit the Nvidia GPUs that are visible to Ray.
            For example, ``CUDA_VISIBLE_DEVICES=1,3 ray start --head --num-gpus=2``
            lets Ray only see devices 1 and 3.

    .. tab-item:: AMD GPU
        :sync: AMD GPU

        .. tip::

            You can set the ``ROCR_VISIBLE_DEVICES`` environment variable before starting a Ray node
            to limit the AMD GPUs that are visible to Ray.
            For example, ``ROCR_VISIBLE_DEVICES=1,3 ray start --head --num-gpus=2``
            lets Ray only see devices 1 and 3.

    .. tab-item:: Intel GPU
        :sync: Intel GPU

        .. tip::

            You can set the ``ONEAPI_DEVICE_SELECTOR`` environment variable before starting a Ray node
            to limit the Intel GPUs that are visible to Ray.
            For example, ``ONEAPI_DEVICE_SELECTOR=1,3 ray start --head --num-gpus=2``
            lets Ray only see devices 1 and 3.

    .. tab-item:: AWS Neuron Core
        :sync: AWS Neuron Core

        .. tip::

            You can set the ``NEURON_RT_VISIBLE_CORES`` environment variable before starting a Ray node
            to limit the AWS Neuro Cores that are visible to Ray.
            For example, ``NEURON_RT_VISIBLE_CORES=1,3 ray start --head --resources='{"neuron_cores": 2}'``
            lets Ray only see devices 1 and 3.

    .. tab-item:: Google TPU
        :sync: Google TPU

        .. tip::

            You can set the ``TPU_VISIBLE_CHIPS`` environment variable before starting a Ray node
            to limit the Google TPUs that are visible to Ray.
            For example, ``TPU_VISIBLE_CHIPS=1,3 ray start --head --resources='{"TPU": 2}'``
            lets Ray only see devices 1 and 3.

    .. tab-item:: Intel Gaudi
        :sync: Intel Gaudi

        .. tip::

            You can set the ``HABANA_VISIBLE_MODULES`` environment variable before starting a Ray node
            to limit the Intel Gaudi HPUs that are visible to Ray.
            For example, ``HABANA_VISIBLE_MODULES=1,3 ray start --head --resources='{"HPU": 2}'``
            lets Ray only see devices 1 and 3.

    .. tab-item:: Huawei Ascend
        :sync: Huawei Ascend

        .. tip::

            You can set the ``ASCEND_RT_VISIBLE_DEVICES`` environment variable before starting a Ray node
            to limit the Huawei Ascend NPUs that are visible to Ray.
            For example, ``ASCEND_RT_VISIBLE_DEVICES=1,3 ray start --head --resources='{"NPU": 2}'``
            lets Ray only see devices 1 and 3.

.. note::

  There is nothing preventing you from specifying a larger number of
  accelerator resources (e.g. ``num_gpus``) than the true number of accelerators on the machine given Ray resources are :ref:`logical <logical-resources>`.
  In this case, Ray acts as if the machine has the number of accelerators you specified
  for the purposes of scheduling tasks and actors that require accelerators.
  Trouble only occurs if those tasks and actors
  attempt to actually use accelerators that don't exist.

Using accelerators in Tasks and Actors
--------------------------------------

If a task or actor requires accelerators, you can specify the corresponding :ref:`resource requirements <resource-requirements>` (e.g. ``@ray.remote(num_gpus=1)``).
Ray then schedules the task or actor to a node that has enough free accelerator resources
and assign accelerators to the task or actor by setting the corresponding environment variable (e.g. ``CUDA_VISIBLE_DEVICES``) before running the task or actor code.

.. tab-set::

    .. tab-item:: Nvidia GPU
        :sync: Nvidia GPU

        .. testcode::

            import os
            import ray

            ray.init(num_gpus=2)

            @ray.remote(num_gpus=1)
            class GPUActor:
                def ping(self):
                    print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
                    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

            @ray.remote(num_gpus=1)
            def gpu_task():
                print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
                print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

            gpu_actor = GPUActor.remote()
            ray.get(gpu_actor.ping.remote())
            # The actor uses the first GPU so the task uses the second one.
            ray.get(gpu_task.remote())

        .. testoutput::
            :options: +MOCK

            (GPUActor pid=52420) GPU IDs: [0]
            (GPUActor pid=52420) CUDA_VISIBLE_DEVICES: 0
            (gpu_task pid=51830) GPU IDs: [1]
            (gpu_task pid=51830) CUDA_VISIBLE_DEVICES: 1

    .. tab-item:: AMD GPU
        :sync: AMD GPU

        .. testcode::
            :hide:

            ray.shutdown()

        .. testcode::
            :skipif: True

            import os
            import ray

            ray.init(num_gpus=2)

            @ray.remote(num_gpus=1)
            class GPUActor:
                def ping(self):
                    print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
                    print("ROCR_VISIBLE_DEVICES: {}".format(os.environ["ROCR_VISIBLE_DEVICES"]))

            @ray.remote(num_gpus=1)
            def gpu_task():
                print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
                print("ROCR_VISIBLE_DEVICES: {}".format(os.environ["ROCR_VISIBLE_DEVICES"]))

            gpu_actor = GPUActor.remote()
            ray.get(gpu_actor.ping.remote())
            # The actor uses the first GPU so the task uses the second one.
            ray.get(gpu_task.remote())

        .. testoutput::
            :options: +MOCK

            (GPUActor pid=52420) GPU IDs: [0]
            (GPUActor pid=52420) ROCR_VISIBLE_DEVICES: 0
            (gpu_task pid=51830) GPU IDs: [1]
            (gpu_task pid=51830) ROCR_VISIBLE_DEVICES: 1

    .. tab-item:: Intel GPU
        :sync: Intel GPU

        .. testcode::
            :hide:

            ray.shutdown()

        .. testcode::
            :skipif: True

            import os
            import ray

            ray.init(num_gpus=2)

            @ray.remote(num_gpus=1)
            class GPUActor:
                def ping(self):
                    print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
                    print("ONEAPI_DEVICE_SELECTOR: {}".format(os.environ["ONEAPI_DEVICE_SELECTOR"]))

            @ray.remote(num_gpus=1)
            def gpu_task():
                print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
                print("ONEAPI_DEVICE_SELECTOR: {}".format(os.environ["ONEAPI_DEVICE_SELECTOR"]))

            gpu_actor = GPUActor.remote()
            ray.get(gpu_actor.ping.remote())
            # The actor uses the first GPU so the task uses the second one.
            ray.get(gpu_task.remote())

        .. testoutput::
            :options: +MOCK

            (GPUActor pid=52420) GPU IDs: [0]
            (GPUActor pid=52420) ONEAPI_DEVICE_SELECTOR: 0
            (gpu_task pid=51830) GPU IDs: [1]
            (gpu_task pid=51830) ONEAPI_DEVICE_SELECTOR: 1

    .. tab-item:: AWS Neuron Core
        :sync: AWS Neuron Core

        .. testcode::
            :hide:

            ray.shutdown()

        .. testcode::

            import os
            import ray

            ray.init(resources={"neuron_cores": 2})

            @ray.remote(resources={"neuron_cores": 1})
            class NeuronCoreActor:
                def ping(self):
                    print("Neuron Core IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["neuron_cores"]))
                    print("NEURON_RT_VISIBLE_CORES: {}".format(os.environ["NEURON_RT_VISIBLE_CORES"]))

            @ray.remote(resources={"neuron_cores": 1})
            def neuron_core_task():
                print("Neuron Core IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["neuron_cores"]))
                print("NEURON_RT_VISIBLE_CORES: {}".format(os.environ["NEURON_RT_VISIBLE_CORES"]))

            neuron_core_actor = NeuronCoreActor.remote()
            ray.get(neuron_core_actor.ping.remote())
            # The actor uses the first Neuron Core so the task uses the second one.
            ray.get(neuron_core_task.remote())

        .. testoutput::
            :options: +MOCK

            (NeuronCoreActor pid=52420) Neuron Core IDs: [0]
            (NeuronCoreActor pid=52420) NEURON_RT_VISIBLE_CORES: 0
            (neuron_core_task pid=51830) Neuron Core IDs: [1]
            (neuron_core_task pid=51830) NEURON_RT_VISIBLE_CORES: 1

    .. tab-item:: Google TPU
        :sync: Google TPU

        .. testcode::
            :hide:

            ray.shutdown()

        .. testcode::

            import os
            import ray

            ray.init(resources={"TPU": 2})

            @ray.remote(resources={"TPU": 1})
            class TPUActor:
                def ping(self):
                    print("TPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["TPU"]))
                    print("TPU_VISIBLE_CHIPS: {}".format(os.environ["TPU_VISIBLE_CHIPS"]))

            @ray.remote(resources={"TPU": 1})
            def tpu_task():
                print("TPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["TPU"]))
                print("TPU_VISIBLE_CHIPS: {}".format(os.environ["TPU_VISIBLE_CHIPS"]))

            tpu_actor = TPUActor.remote()
            ray.get(tpu_actor.ping.remote())
            # The actor uses the first TPU so the task uses the second one.
            ray.get(tpu_task.remote())

        .. testoutput::
            :options: +MOCK

            (TPUActor pid=52420) TPU IDs: [0]
            (TPUActor pid=52420) TPU_VISIBLE_CHIPS: 0
            (tpu_task pid=51830) TPU IDs: [1]
            (tpu_task pid=51830) TPU_VISIBLE_CHIPS: 1

    .. tab-item:: Intel Gaudi
        :sync: Intel Gaudi

        .. testcode::
            :hide:

            ray.shutdown()

        .. testcode::

            import os
            import ray

            ray.init(resources={"HPU": 2})

            @ray.remote(resources={"HPU": 1})
            class HPUActor:
                def ping(self):
                    print("HPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["HPU"]))
                    print("HABANA_VISIBLE_MODULES: {}".format(os.environ["HABANA_VISIBLE_MODULES"]))

            @ray.remote(resources={"HPU": 1})
            def hpu_task():
                print("HPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["HPU"]))
                print("HABANA_VISIBLE_MODULES: {}".format(os.environ["HABANA_VISIBLE_MODULES"]))

            hpu_actor = HPUActor.remote()
            ray.get(hpu_actor.ping.remote())
            # The actor uses the first HPU so the task uses the second one.
            ray.get(hpu_task.remote())

        .. testoutput::
            :options: +MOCK

            (HPUActor pid=52420) HPU IDs: [0]
            (HPUActor pid=52420) HABANA_VISIBLE_MODULES: 0
            (hpu_task pid=51830) HPU IDs: [1]
            (hpu_task pid=51830) HABANA_VISIBLE_MODULES: 1

    .. tab-item:: Huawei Ascend
        :sync: Huawei Ascend

        .. testcode::
            :hide:

            ray.shutdown()

        .. testcode::

            import os
            import ray

            ray.init(resources={"NPU": 2})

            @ray.remote(resources={"NPU": 1})
            class NPUActor:
                def ping(self):
                    print("NPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["NPU"]))
                    print("ASCEND_RT_VISIBLE_DEVICES: {}".format(os.environ["ASCEND_RT_VISIBLE_DEVICES"]))

            @ray.remote(resources={"NPU": 1})
            def npu_task():
                print("NPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["NPU"]))
                print("ASCEND_RT_VISIBLE_DEVICES: {}".format(os.environ["ASCEND_RT_VISIBLE_DEVICES"]))

            npu_actor = NPUActor.remote()
            ray.get(npu_actor.ping.remote())
            # The actor uses the first NPU so the task uses the second one.
            ray.get(npu_task.remote())

        .. testoutput::
            :options: +MOCK

            (NPUActor pid=52420) NPU IDs: [0]
            (NPUActor pid=52420) ASCEND_RT_VISIBLE_DEVICES: 0
            (npu_task pid=51830) NPU IDs: [1]
            (npu_task pid=51830) ASCEND_RT_VISIBLE_DEVICES: 1


Inside a task or actor, :func:`ray.get_runtime_context().get_accelerator_ids() <ray.runtime_context.RuntimeContext.get_accelerator_ids>` returns a
list of accelerator IDs that are available to the task or actor.
Typically, it is not necessary to call ``get_accelerator_ids()`` because Ray
automatically sets the corresponding environment variable (e.g. ``CUDA_VISIBLE_DEVICES``),
which most ML frameworks respect for purposes of accelerator assignment.

**Note:** The remote function or actor defined above doesn't actually use any
accelerators. Ray schedules it on a node which has at least one accelerator, and
reserves one accelerator for it while it is being executed, however it is up to the
function to actually make use of the accelerator. This is typically done through an
external library like TensorFlow. Here is an example that actually uses accelerators.
In order for this example to work, you need to install the GPU version of
TensorFlow.

.. testcode::

    @ray.remote(num_gpus=1)
    def gpu_task():
        import tensorflow as tf

        # Create a TensorFlow session. TensorFlow restricts itself to use the
        # GPUs specified by the CUDA_VISIBLE_DEVICES environment variable.
        tf.Session()


**Note:** It is certainly possible for the person to
ignore assigned accelerators and to use all of the accelerators on the machine. Ray does
not prevent this from happening, and this can lead to too many tasks or actors using the
same accelerator at the same time. However, Ray does automatically set the
environment variable (e.g. ``CUDA_VISIBLE_DEVICES``), which restricts the accelerators used
by most deep learning frameworks assuming it's not overridden by the user.

Fractional Accelerators
-----------------------

Ray supports :ref:`fractional resource requirements <fractional-resource-requirements>`
so multiple tasks and actors can share the same accelerator.

.. tab-set::

    .. tab-item:: Nvidia GPU
        :sync: Nvidia GPU

        .. testcode::
            :hide:

            ray.shutdown()

        .. testcode::

            ray.init(num_cpus=4, num_gpus=1)

            @ray.remote(num_gpus=0.25)
            def f():
                import time

                time.sleep(1)

            # The four tasks created here can execute concurrently
            # and share the same GPU.
            ray.get([f.remote() for _ in range(4)])

    .. tab-item:: AMD GPU
        :sync: AMD GPU

        .. testcode::
            :hide:

            ray.shutdown()

        .. testcode::

            ray.init(num_cpus=4, num_gpus=1)

            @ray.remote(num_gpus=0.25)
            def f():
                import time

                time.sleep(1)

            # The four tasks created here can execute concurrently
            # and share the same GPU.
            ray.get([f.remote() for _ in range(4)])

    .. tab-item:: Intel GPU
        :sync: Intel GPU

        .. testcode::
            :hide:

            ray.shutdown()

        .. testcode::

            ray.init(num_cpus=4, num_gpus=1)

            @ray.remote(num_gpus=0.25)
            def f():
                import time

                time.sleep(1)

            # The four tasks created here can execute concurrently
            # and share the same GPU.
            ray.get([f.remote() for _ in range(4)])

    .. tab-item:: AWS Neuron Core
        :sync: AWS Neuron Core

        AWS Neuron Core doesn't support fractional resource.

    .. tab-item:: Google TPU
        :sync: Google TPU

        Google TPU doesn't support fractional resource.

    .. tab-item:: Intel Gaudi
        :sync: Intel Gaudi

        Intel Gaudi doesn't support fractional resource.

    .. tab-item:: Huawei Ascend
        :sync: Huawei Ascend

        .. testcode::
            :hide:

            ray.shutdown()

        .. testcode::

            ray.init(num_cpus=4, resources={"NPU": 1})

            @ray.remote(resources={"NPU": 0.25})
            def f():
                import time

                time.sleep(1)

            # The four tasks created here can execute concurrently
            # and share the same NPU.
            ray.get([f.remote() for _ in range(4)])


**Note:** It is the user's responsibility to make sure that the individual tasks
don't use more than their share of the accelerator memory.
Pytorch and TensorFlow can be configured to limit its memory usage.

When Ray assigns accelerators of a node to tasks or actors with fractional resource requirements,
it packs one accelerator before moving on to the next one to avoid fragmentation.

.. testcode::
    :hide:

    ray.shutdown()

.. testcode::

    ray.init(num_gpus=3)

    @ray.remote(num_gpus=0.5)
    class FractionalGPUActor:
        def ping(self):
            print("GPU id: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))

    fractional_gpu_actors = [FractionalGPUActor.remote() for _ in range(3)]
    # Ray tries to pack GPUs if possible.
    [ray.get(fractional_gpu_actors[i].ping.remote()) for i in range(3)]

.. testoutput::
    :options: +MOCK

    (FractionalGPUActor pid=57417) GPU id: [0]
    (FractionalGPUActor pid=57416) GPU id: [0]
    (FractionalGPUActor pid=57418) GPU id: [1]

.. _gpu-leak:

Workers not Releasing GPU Resources
-----------------------------------

Currently, when a worker executes a task that uses a GPU (e.g.,
through TensorFlow), the task may allocate memory on the GPU and may not release
it when the task finishes executing. This can lead to problems the next time a
task tries to use the same GPU. To address the problem, Ray disables the worker
process reuse between GPU tasks by default, where the GPU resources is released after
the task process exits. Since this adds overhead to GPU task scheduling,
you can re-enable worker reuse by setting ``max_calls=0``
in the :func:`ray.remote <ray.remote>` decorator.

.. testcode::

    # By default, ray does not reuse workers for GPU tasks to prevent
    # GPU resource leakage.
    @ray.remote(num_gpus=1)
    def leak_gpus():
        import tensorflow as tf

        # This task allocates memory on the GPU and then never release it.
        tf.Session()

.. _accelerator-types:

Accelerator Types
-----------------

Ray supports resource specific accelerator types. The `accelerator_type` option can be used to force to a task or actor to run on a node with a specific type of accelerator.
Under the hood, the accelerator type option is implemented as a :ref:`custom resource requirement <custom-resources>` of ``"accelerator_type:<type>": 0.001``.
This forces the task or actor to be placed on a node with that particular accelerator type available.
This also lets the multi-node-type autoscaler know that there is demand for that type of resource, potentially triggering the launch of new nodes providing that accelerator.

.. testcode::
    :hide:

    ray.shutdown()
    import ray.util.accelerators
    import ray._private.ray_constants as ray_constants

    v100_resource_name = f"{ray_constants.RESOURCE_CONSTRAINT_PREFIX}{ray.util.accelerators.NVIDIA_TESLA_V100}"
    ray.init(num_gpus=4, resources={v100_resource_name: 1})

.. testcode::

    from ray.util.accelerators import NVIDIA_TESLA_V100

    @ray.remote(num_gpus=1, accelerator_type=NVIDIA_TESLA_V100)
    def train(data):
        return "This function was run on a node with a Tesla V100 GPU"

    ray.get(train.remote(1))

See ``ray.util.accelerators`` for available accelerator types.


.. _ray-scheduling:

Scheduling
==========

For each task or actor, Ray will choose a node to run it and the scheduling decision is based on the following factors.

.. _ray-scheduling-resources:

Resources
---------

Each task or actor has the :ref:`specified resource requirements <resource-requirements>`.
Given that, a node can be in one of the following states:

- Feasible: the node has the required resources to run the task or actor.
  Depending on the current availability of these resources, there are two sub-states:

  - Available: the node has the required resources and they are free now.
  - Unavailable: the node has the required resources but they are currently being used by other tasks or actors.

- Infeasible: the node doesn't have the required resources. For example a CPU-only node is infeasible for a GPU task.

Resource requirements are **hard** requirements meaning that only feasible nodes are eligible to run the task or actor.
If there are feasible nodes, Ray will either choose an available node or wait until a unavailable node to become available
depending on other factors discussed below.
If all nodes are infeasible, the task or actor cannot be scheduled until feasible nodes are added to the cluster.

.. _ray-scheduling-strategies:

Scheduling Strategies
---------------------

Tasks or actors support a :func:`scheduling_strategy <ray.remote>` option to specify the strategy used to decide the best node among feasible nodes.
Currently the supported strategies are the followings.

"DEFAULT"
~~~~~~~~~

``"DEFAULT"`` is the default strategy used by Ray.
Ray schedules tasks or actors onto a group of the top k nodes.
Specifically, the nodes are sorted to first favor those that already have tasks or actors scheduled (for locality),
then to favor those that have low resource utilization (for load balancing).
Within the top k group, nodes are chosen randomly to further improve load-balancing and mitigate delays from cold-start in large clusters.

Implementation-wise, Ray calculates a score for each node in a cluster based on the utilization of its logical resources.
If the utilization is below a threshold (controlled by the OS environment variable ``RAY_scheduler_spread_threshold``, default is 0.5), the score is 0,
otherwise it is the resource utilization itself (score 1 means the node is fully utilized).
Ray selects the best node for scheduling by randomly picking from the top k nodes with the lowest scores.
The value of ``k`` is the max of (number of nodes in the cluster * ``RAY_scheduler_top_k_fraction`` environment variable) and ``RAY_scheduler_top_k_absolute`` environment variable.
By default, it's 20% of the total number of nodes.

Currently Ray handles actors that don't require any resources (i.e., ``num_cpus=0`` with no other resources) specially by randomly choosing a node in the cluster without considering resource utilization.
Since nodes are randomly chosen, actors that don't require any resources are effectively SPREAD across the cluster.

.. literalinclude:: ../doc_code/scheduling.py
    :language: python
    :start-after: __default_scheduling_strategy_start__
    :end-before: __default_scheduling_strategy_end__

"SPREAD"
~~~~~~~~

``"SPREAD"`` strategy will try to spread the tasks or actors among available nodes.

.. literalinclude:: ../doc_code/scheduling.py
    :language: python
    :start-after: __spread_scheduling_strategy_start__
    :end-before: __spread_scheduling_strategy_end__

PlacementGroupSchedulingStrategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy` will schedule the task or actor to where the placement group is located.
This is useful for actor gang scheduling. See :ref:`Placement Group <ray-placement-group-doc-ref>` for more details.

NodeAffinitySchedulingStrategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy` is a low-level strategy that allows a task or actor to be scheduled onto a particular node specified by its node id.
The ``soft`` flag specifies whether the task or actor is allowed to run somewhere else if the specified node doesn't exist (e.g. if the node dies)
or is infeasible because it does not have the resources required to run the task or actor.
In these cases, if ``soft`` is True, the task or actor will be scheduled onto a different feasible node.
Otherwise, the task or actor will fail with :py:class:`~ray.exceptions.TaskUnschedulableError` or :py:class:`~ray.exceptions.ActorUnschedulableError`.
As long as the specified node is alive and feasible, the task or actor will only run there
regardless of the ``soft`` flag. This means if the node currently has no available resources, the task or actor will wait until resources
become available.
This strategy should *only* be used if other high level scheduling strategies (e.g. :ref:`placement group <ray-placement-group-doc-ref>`) cannot give the
desired task or actor placements. It has the following known limitations:

- It's a low-level strategy which prevents optimizations by a smart scheduler.
- It cannot fully utilize an autoscaling cluster since node ids must be known when the tasks or actors are created.
- It can be difficult to make the best static placement decision
  especially in a multi-tenant cluster: for example, an application won't know what else is being scheduled onto the same nodes.

.. literalinclude:: ../doc_code/scheduling.py
    :language: python
    :start-after: __node_affinity_scheduling_strategy_start__
    :end-before: __node_affinity_scheduling_strategy_end__

.. _ray-scheduling-locality:

Locality-Aware Scheduling
-------------------------

By default, Ray prefers available nodes that have large task arguments local
to avoid transferring data over the network. If there are multiple large task arguments,
the node with most object bytes local is preferred.
This takes precedence over the ``"DEFAULT"`` scheduling strategy,
which means Ray will try to run the task on the locality preferred node regardless of the node resource utilization.
However, if the locality preferred node is not available, Ray may run the task somewhere else.
When other scheduling strategies are specified,
they have higher precedence and data locality is no longer considered.

.. note::

  Locality-aware scheduling is only for tasks not actors.

.. literalinclude:: ../doc_code/scheduling.py
    :language: python
    :start-after: __locality_aware_scheduling_start__
    :end-before: __locality_aware_scheduling_end__

More about Ray Scheduling
-------------------------

.. toctree::
    :maxdepth: 1

    resources
    accelerators
    placement-group
    memory-management
    ray-oom-prevention


Placement Groups
================

.. _ray-placement-group-doc-ref:

Placement groups allow users to atomically reserve groups of resources across multiple nodes (i.e., gang scheduling).
They can be then used to schedule Ray tasks and actors packed as close as possible for locality (PACK), or spread apart 
(SPREAD). Placement groups are generally used for gang-scheduling actors, but also support tasks.

Here are some real-world use cases:

- **Distributed Machine Learning Training**: Distributed Training (e.g., :ref:`Ray Train <train-docs>` and :ref:`Ray Tune <tune-main>`) uses the placement group APIs to enable gang scheduling. In these settings, all resources for a trial must be available at the same time. Gang scheduling is a critical technique to enable all-or-nothing scheduling for deep learning training. 
- **Fault tolerance in distributed training**: Placement groups can be used to configure fault tolerance. In Ray Tune, it can be beneficial to pack related resources from a single trial together, so that a node failure impacts a low number of trials. In libraries that support elastic training (e.g., XGBoost-Ray), spreading the resources across multiple nodes can help to ensure that training continues even when a node dies.

Key Concepts
------------

Bundles
~~~~~~~

A **bundle** is a collection of "resources". It could be a single resource, ``{"CPU": 1}``, or a group of resources, ``{"CPU": 1, "GPU": 4}``. 
A bundle is a unit of reservation for placement groups. "Scheduling a bundle" means we find a node that fits the bundle and reserve the resources specified by the bundle. 
A bundle must be able to fit on a single node on the Ray cluster. For example, if you only have an 8 CPU node, and if you have a bundle that requires ``{"CPU": 9}``, this bundle cannot be scheduled.

Placement Group
~~~~~~~~~~~~~~~

A **placement group** reserves the resources from the cluster. The reserved resources can only be used by tasks or actors that use the :ref:`PlacementGroupSchedulingStrategy <ray-placement-group-schedule-tasks-actors-ref>`.

- Placement groups are represented by a list of bundles. For example, ``{"CPU": 1} * 4`` means you'd like to reserve 4 bundles of 1 CPU (i.e., it reserves 4 CPUs).
- Bundles are then placed according to the :ref:`placement strategies <pgroup-strategy>` across nodes on the cluster.
- After the placement group is created, tasks or actors can be then scheduled according to the placement group and even on individual bundles.

Create a Placement Group (Reserve Resources)
--------------------------------------------

You can create a placement group using :func:`ray.util.placement_group() <ray.util.placement_group.placement_group>`. 
Placement groups take in a list of bundles and a :ref:`placement strategy <pgroup-strategy>`. 
Note that each bundle must be able to fit on a single node on the Ray cluster.
For example, if you only have a 8 CPU node, and if you have a bundle that requires ``{"CPU": 9}``,
this bundle cannot be scheduled.

Bundles are specified by a list of dictionaries, e.g., ``[{"CPU": 1}, {"CPU": 1, "GPU": 1}]``).

- ``CPU`` corresponds to ``num_cpus`` as used in :func:`ray.remote <ray.remote>`.
- ``GPU`` corresponds to ``num_gpus`` as used in :func:`ray.remote <ray.remote>`.
- ``memory`` corresponds to ``memory`` as used in :func:`ray.remote <ray.remote>`
- Other resources corresponds to ``resources`` as used in :func:`ray.remote <ray.remote>` (E.g., ``ray.init(resources={"disk": 1})`` can have a bundle of ``{"disk": 1}``).

Placement group scheduling is asynchronous. The `ray.util.placement_group` returns immediately.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/placement_group_example.py
            :language: python
            :start-after: __create_pg_start__
            :end-before: __create_pg_end__


    .. tab-item:: Java

        .. code-block:: java

          // Initialize Ray.
          Ray.init();

          // Construct a list of bundles.
          Map<String, Double> bundle = ImmutableMap.of("CPU", 1.0);
          List<Map<String, Double>> bundles = ImmutableList.of(bundle);

          // Make a creation option with bundles and strategy.
          PlacementGroupCreationOptions options =
            new PlacementGroupCreationOptions.Builder()
              .setBundles(bundles)
              .setStrategy(PlacementStrategy.STRICT_SPREAD)
              .build();

          PlacementGroup pg = PlacementGroups.createPlacementGroup(options);

    .. tab-item:: C++

        .. code-block:: c++

          // Initialize Ray.
          ray::Init();

          // Construct a list of bundles.
          std::vector<std::unordered_map<std::string, double>> bundles{{{"CPU", 1.0}}};

          // Make a creation option with bundles and strategy.
          ray::internal::PlacementGroupCreationOptions options{
              false, "my_pg", bundles, ray::internal::PlacementStrategy::PACK};

          ray::PlacementGroup pg = ray::CreatePlacementGroup(options);

You can block your program until the placement group is ready using one of two APIs:

* :func:`ready <ray.util.placement_group.PlacementGroup.ready>`, which is compatible with ``ray.get``
* :func:`wait <ray.util.placement_group.PlacementGroup.wait>`, which blocks the program until the placement group is ready)

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/placement_group_example.py
            :language: python
            :start-after: __ready_pg_start__
            :end-before: __ready_pg_end__

    .. tab-item:: Java

        .. code-block:: java

          // Wait for the placement group to be ready within the specified time(unit is seconds).
          boolean ready = pg.wait(60);
          Assert.assertTrue(ready);

          // You can look at placement group states using this API.
          List<PlacementGroup> allPlacementGroup = PlacementGroups.getAllPlacementGroups();
          for (PlacementGroup group: allPlacementGroup) {
            System.out.println(group);
          }

    .. tab-item:: C++

        .. code-block:: c++

          // Wait for the placement group to be ready within the specified time(unit is seconds).
          bool ready = pg.Wait(60);
          assert(ready);

          // You can look at placement group states using this API.
          std::vector<ray::PlacementGroup> all_placement_group = ray::GetAllPlacementGroups();
          for (const ray::PlacementGroup &group : all_placement_group) {
            std::cout << group.GetName() << std::endl;
          }

Let's verify the placement group is successfully created.

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list placement-groups

.. code-block:: bash

  ======== List: 2023-04-07 01:15:05.682519 ========
  Stats:
  ------------------------------
  Total: 1

  Table:
  ------------------------------
      PLACEMENT_GROUP_ID                    NAME      CREATOR_JOB_ID  STATE
  0  3cd6174711f47c14132155039c0501000000                  01000000  CREATED

The placement group is successfully created. Out of the ``{"CPU": 2, "GPU": 2}`` resources, the placement group reserves ``{"CPU": 1, "GPU": 1}``. 
The reserved resources can only be used when you schedule tasks or actors with a placement group.
The diagram below demonstrates the "1 CPU and 1 GPU" bundle that the placement group reserved.

.. image:: ../images/pg_image_1.png
    :align: center

Placement groups are atomically created; if a bundle cannot fit in any of the current nodes, 
the entire placement group is not ready and no resources are reserved.
To illustrate, let's create another placement group that requires ``{"CPU":1}, {"GPU": 2}`` (2 bundles).

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/placement_group_example.py
            :language: python
            :start-after: __create_pg_failed_start__
            :end-before: __create_pg_failed_end__

You can verify the new placement group is pending creation.

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list placement-groups

.. code-block:: bash

  ======== List: 2023-04-07 01:16:23.733410 ========
  Stats:
  ------------------------------
  Total: 2

  Table:
  ------------------------------
      PLACEMENT_GROUP_ID                    NAME      CREATOR_JOB_ID  STATE
  0  3cd6174711f47c14132155039c0501000000                  01000000  CREATED
  1  e1b043bebc751c3081bddc24834d01000000                  01000000  PENDING <---- the new placement group.

You can also verify that the ``{"CPU": 1, "GPU": 2}`` bundles cannot be allocated, using the ``ray status`` CLI command.

.. code-block:: bash

  ray status

.. code-block:: bash

  Resources
  ---------------------------------------------------------------
  Usage:
  0.0/2.0 CPU (0.0 used of 1.0 reserved in placement groups)
  0.0/2.0 GPU (0.0 used of 1.0 reserved in placement groups)
  0B/3.46GiB memory
  0B/1.73GiB object_store_memory

  Demands:
  {'CPU': 1.0} * 1, {'GPU': 2.0} * 1 (PACK): 1+ pending placement groups <--- 1 placement group is pending creation.

The current cluster has ``{"CPU": 2, "GPU": 2}``. We already created a ``{"CPU": 1, "GPU": 1}`` bundle, so only ``{"CPU": 1, "GPU": 1}`` is left in the cluster.
If we create 2 bundles ``{"CPU": 1}, {"GPU": 2}``, we can create a first bundle successfully, but can't schedule the second bundle.
Since we cannot create every bundle on the cluster, the placement group is not created, including the ``{"CPU": 1}`` bundle.

.. image:: ../images/pg_image_2.png
    :align: center

When the placement group cannot be scheduled in any way, it is called "infeasible". 
Imagine you schedule ``{"CPU": 4}`` bundle, but you only have a single node with 2 CPUs. There's no way to create this bundle in your cluster.
The Ray Autoscaler is aware of placement groups, and auto-scales the cluster to ensure pending groups can be placed as needed. 

If Ray Autoscaler cannot provide resources to schedule a placement group, Ray does *not* print a warning about infeasible groups and tasks and actors that use the groups. 
You can observe the scheduling state of the placement group from the :ref:`dashboard or state APIs <ray-placement-group-observability-ref>`.

.. _ray-placement-group-schedule-tasks-actors-ref:

Schedule Tasks and Actors to Placement Groups (Use Reserved Resources)
----------------------------------------------------------------------

In the previous section, we created a placement group that reserved ``{"CPU": 1, "GPU: 1"}`` from a 2 CPU and 2 GPU node.

Now let's schedule an actor to the placement group. 
You can schedule actors or tasks to a placement group using
:class:`options(scheduling_strategy=PlacementGroupSchedulingStrategy(...)) <ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy>`.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/placement_group_example.py
            :language: python
            :start-after: __schedule_pg_start__
            :end-before: __schedule_pg_end__

    .. tab-item:: Java

        .. code-block:: java

          public static class Counter {
            private int value;

            public Counter(int initValue) {
              this.value = initValue;
            }

            public int getValue() {
              return value;
            }

            public static String ping() {
              return "pong";
            }
          }

          // Create GPU actors on a gpu bundle.
          for (int index = 0; index < 1; index++) {
            Ray.actor(Counter::new, 1)
              .setPlacementGroup(pg, 0)
              .remote();
          }

    .. tab-item:: C++

        .. code-block:: c++

          class Counter {
          public:
            Counter(int init_value) : value(init_value){}
            int GetValue() {return value;}
            std::string Ping() {
              return "pong";
            }
          private:
            int value;
          };

          // Factory function of Counter class.
          static Counter *CreateCounter() {
            return new Counter();
          };

          RAY_REMOTE(&Counter::Ping, &Counter::GetValue, CreateCounter);

          // Create GPU actors on a gpu bundle.
          for (int index = 0; index < 1; index++) {
            ray::Actor(CreateCounter)
              .SetPlacementGroup(pg, 0)
              .Remote(1);
          }

.. note::

  When you use an actor with a placement group, always specify ``num_cpus``.

  When you don't specify (e.g., ``num_cpus=0``), a placement group option is ignored,
  and the task and actor don't use the reserved resources.
  
  Note that by default (with no arguments to ``ray.remote``),

  - Ray task requires 1 CPU
  - Ray actor requires 1 CPU when it is scheduled. But after it is created, it occupies 0 CPU.

  When scheduling an actor without resource requirements and a placement group, the placement group has to be created (since it requires 1 CPU to be scheduled).
  However, when the actor is created, it ignores the placement group.

The actor is scheduled now! One bundle can be used by multiple tasks and actors (i.e., the bundle to task (or actor) is a one-to-many relationship). 
In this case, since the actor uses 1 CPU, 1 GPU remains from the bundle. 
You can verify this from the CLI command ``ray status``. You can see the 1 CPU is reserved by the placement group, and 1.0 is used (by the actor we created).

.. code-block:: bash

  ray status

.. code-block:: bash

  Resources
  ---------------------------------------------------------------
  Usage:
  1.0/2.0 CPU (1.0 used of 1.0 reserved in placement groups) <---
  0.0/2.0 GPU (0.0 used of 1.0 reserved in placement groups)
  0B/4.29GiB memory
  0B/2.00GiB object_store_memory

  Demands:
  (no resource demands)

You can also verify the actor is created using ``ray list actors``.

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list actors --detail

.. code-block:: bash

  -   actor_id: b5c990f135a7b32bfbb05e1701000000
      class_name: Actor
      death_cause: null
      is_detached: false
      job_id: '01000000'
      name: ''
      node_id: b552ca3009081c9de857a31e529d248ba051a4d3aeece7135dde8427
      pid: 8795
      placement_group_id: d2e660ac256db230dbe516127c4a01000000 <------
      ray_namespace: e5b19111-306c-4cd8-9e4f-4b13d42dff86
      repr_name: ''
      required_resources:
          CPU_group_d2e660ac256db230dbe516127c4a01000000: 1.0
      serialized_runtime_env: '{}'
      state: ALIVE

Since 1 GPU remains, let's create a new actor that requires 1 GPU.
This time, we also specify the ``placement_group_bundle_index``. Each bundle is given an "index" within the placement group.
For example, a placement group of 2 bundles ``[{"CPU": 1}, {"GPU": 1}]`` has index 0 bundle ``{"CPU": 1}`` 
and index 1 bundle ``{"GPU": 1}``. Since we only have 1 bundle, we only have index 0. If you don't specify a bundle, the actor (or task)
is scheduled on a random bundle that has unallocated reserved resources.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/placement_group_example.py
            :language: python
            :start-after: __schedule_pg_3_start__
            :end-before: __schedule_pg_3_end__

We succeed to schedule the GPU actor! The below image describes 2 actors scheduled into the placement group. 

.. image:: ../images/pg_image_3.png
    :align: center

You can also verify that the reserved resources are all used, with the ``ray status`` command.

.. code-block:: bash

  ray status

.. code-block:: bash

  Resources
  ---------------------------------------------------------------
  Usage:
  1.0/2.0 CPU (1.0 used of 1.0 reserved in placement groups)
  1.0/2.0 GPU (1.0 used of 1.0 reserved in placement groups) <----
  0B/4.29GiB memory
  0B/2.00GiB object_store_memory

.. _pgroup-strategy:

Placement Strategy
------------------

One of the features the placement group provides is to add placement constraints among bundles.

For example, you'd like to pack your bundles to the same
node or spread out to multiple nodes as much as possible. You can specify the strategy via ``strategy`` argument.
This way, you can make sure your actors and tasks can be scheduled with certain placement constraints.

The example below creates a placement group with 2 bundles with a PACK strategy;
both bundles have to be created in the same node. Note that it is a soft policy. If the bundles cannot be packed
into a single node, they are spread to other nodes. If you'd like to avoid the problem, you can instead use `STRICT_PACK` 
policies, which fail to create placement groups if placement requirements cannot be satisfied.

.. literalinclude:: ../doc_code/placement_group_example.py
    :language: python
    :start-after: __strategy_pg_start__
    :end-before: __strategy_pg_end__

The image below demonstrates the PACK policy. Three of the ``{"CPU": 2}`` bundles are located in the same node.

.. image:: ../images/pg_image_4.png
    :align: center

The image below demonstrates the SPREAD policy. Each of three of the ``{"CPU": 2}`` bundles are located in three different nodes.

.. image:: ../images/pg_image_5.png
    :align: center

Ray supports four placement group strategies. The default scheduling policy is ``PACK``.

**STRICT_PACK**

All bundles must be placed into a single node on the cluster. Use this strategy when you want to maximize the locality.

**PACK**

All provided bundles are packed onto a single node on a best-effort basis.
If strict packing is not feasible (i.e., some bundles do not fit on the node), bundles can be placed onto other nodes.

**STRICT_SPREAD**

Each bundle must be scheduled in a separate node.

**SPREAD**

Each bundle is spread onto separate nodes on a best-effort basis.
If strict spreading is not feasible, bundles can be placed on overlapping nodes.

Remove Placement Groups (Free Reserved Resources)
-------------------------------------------------

By default, a placement group's lifetime is scoped to the driver that creates placement groups 
(unless you make it a :ref:`detached placement group <placement-group-detached>`). When the placement group is created from
a :ref:`detached actor <actor-lifetimes>`, the lifetime is scoped to the detached actor.
In Ray, the driver is the Python script that calls ``ray.init``.

Reserved resources (bundles) from the placement group are automatically freed when the driver or detached actor
that creates placement group exits. To free the reserved resources manually, remove the placement
group using the :func:`remove_placement_group <ray.util.remove_placement_group>` API (which is also an asynchronous API).

.. note::

  When you remove the placement group, actors or tasks that still use the reserved resources are
  forcefully killed.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/placement_group_example.py
            :language: python
            :start-after: __remove_pg_start__
            :end-before: __remove_pg_end__

    .. tab-item:: Java

        .. code-block:: java

          PlacementGroups.removePlacementGroup(placementGroup.getId());

          PlacementGroup removedPlacementGroup = PlacementGroups.getPlacementGroup(placementGroup.getId());
          Assert.assertEquals(removedPlacementGroup.getState(), PlacementGroupState.REMOVED);

    .. tab-item:: C++

        .. code-block:: c++

          ray::RemovePlacementGroup(placement_group.GetID());

          ray::PlacementGroup removed_placement_group = ray::GetPlacementGroup(placement_group.GetID());
          assert(removed_placement_group.GetState(), ray::PlacementGroupState::REMOVED);

.. _ray-placement-group-observability-ref:

Observe and Debug Placement Groups
----------------------------------

Ray provides several useful tools to inspect the placement group states and resource usage.

- **Ray Status** is a CLI tool for viewing the resource usage and scheduling resource requirements of placement groups.
- **Ray Dashboard** is a UI tool for inspecting placement group states.
- **Ray State API** is a CLI for inspecting placement group states.

.. tab-set::

    .. tab-item:: ray status (CLI)

      The CLI command ``ray status`` provides the autoscaling status of the cluster.
      It provides the "resource demands" from unscheduled placement groups as well as the resource reservation status.

      .. code-block:: bash

        Resources
        ---------------------------------------------------------------
        Usage:
        1.0/2.0 CPU (1.0 used of 1.0 reserved in placement groups)
        0.0/2.0 GPU (0.0 used of 1.0 reserved in placement groups)
        0B/4.29GiB memory
        0B/2.00GiB object_store_memory

    .. tab-item:: Dashboard

      The :ref:`dashboard job view <dash-jobs-view>` provides the placement group table that displays the scheduling state and metadata of the placement group.

      .. note::

        Ray dashboard is only available when you install Ray is with ``pip install "ray[default]"``.

    .. tab-item:: Ray State API

      :ref:`Ray state API <state-api-overview-ref>` is a CLI tool for inspecting the state of Ray resources (tasks, actors, placement groups, etc.).

      ``ray list placement-groups`` provides the metadata and the scheduling state of the placement group.
      ``ray list placement-groups --detail`` provides statistics and scheduling state in a greater detail.

      .. note::

        State API is only available when you install Ray is with ``pip install "ray[default]"``

Inspect Placement Group Scheduling State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the above tools, you can see the state of the placement group. The definition of states are specified in the following files:

- `High level state <https://github.com/ray-project/ray/blob/03a9d2166988b16b7cbf51dac0e6e586455b28d8/src/ray/protobuf/gcs.proto#L579>`_
- `Details <https://github.com/ray-project/ray/blob/03a9d2166988b16b7cbf51dac0e6e586455b28d8/src/ray/protobuf/gcs.proto#L524>`_

.. image:: ../images/pg_image_6.png
    :align: center

[Advanced] Child Tasks and Actors
---------------------------------

By default, child actors and tasks don't share the same placement group that the parent uses.
To automatically schedule child actors or tasks to the same placement group,
set ``placement_group_capture_child_tasks`` to True.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/placement_group_capture_child_tasks_example.py
          :language: python
          :start-after: __child_capture_pg_start__
          :end-before: __child_capture_pg_end__

    .. tab-item:: Java

        It's not implemented for Java APIs yet.

When ``placement_group_capture_child_tasks`` is True, but you don't want to schedule
child tasks and actors to the same placement group, specify ``PlacementGroupSchedulingStrategy(placement_group=None)``.

.. literalinclude:: ../doc_code/placement_group_capture_child_tasks_example.py
  :language: python
  :start-after: __child_capture_disable_pg_start__
  :end-before: __child_capture_disable_pg_end__

[Advanced] Named Placement Group
--------------------------------

A placement group can be given a globally unique name.
This allows you to retrieve the placement group from any job in the Ray cluster.
This can be useful if you cannot directly pass the placement group handle to
the actor or task that needs it, or if you are trying to
access a placement group launched by another driver.
Note that the placement group is still destroyed if its lifetime isn't `detached`.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/placement_group_example.py
            :language: python
            :start-after: __get_pg_start__
            :end-before: __get_pg_end__

    .. tab-item:: Java

        .. code-block:: java

          // Create a placement group with a unique name.
          Map<String, Double> bundle = ImmutableMap.of("CPU", 1.0);
          List<Map<String, Double>> bundles = ImmutableList.of(bundle);

          PlacementGroupCreationOptions options =
            new PlacementGroupCreationOptions.Builder()
              .setBundles(bundles)
              .setStrategy(PlacementStrategy.STRICT_SPREAD)
              .setName("global_name")
              .build();

          PlacementGroup pg = PlacementGroups.createPlacementGroup(options);
          pg.wait(60);

          ...

          // Retrieve the placement group later somewhere.
          PlacementGroup group = PlacementGroups.getPlacementGroup("global_name");
          Assert.assertNotNull(group);

    .. tab-item:: C++

        .. code-block:: c++

          // Create a placement group with a globally unique name.
          std::vector<std::unordered_map<std::string, double>> bundles{{{"CPU", 1.0}}};

          ray::PlacementGroupCreationOptions options{
              true/*global*/, "global_name", bundles, ray::PlacementStrategy::STRICT_SPREAD};

          ray::PlacementGroup pg = ray::CreatePlacementGroup(options);
          pg.Wait(60);

          ...

          // Retrieve the placement group later somewhere.
          ray::PlacementGroup group = ray::GetGlobalPlacementGroup("global_name");
          assert(!group.Empty());

        We also support non-global named placement group in C++, which means that the placement group name is only valid within the job and cannot be accessed from another job.

        .. code-block:: c++

          // Create a placement group with a job-scope-unique name.
          std::vector<std::unordered_map<std::string, double>> bundles{{{"CPU", 1.0}}};

          ray::PlacementGroupCreationOptions options{
              false/*non-global*/, "non_global_name", bundles, ray::PlacementStrategy::STRICT_SPREAD};

          ray::PlacementGroup pg = ray::CreatePlacementGroup(options);
          pg.Wait(60);

          ...

          // Retrieve the placement group later somewhere in the same job.
          ray::PlacementGroup group = ray::GetPlacementGroup("non_global_name");
          assert(!group.Empty());

.. _placement-group-detached:

[Advanced] Detached Placement Group
-----------------------------------

By default, the lifetimes of placement groups belong to the driver and actor.

- If the placement group is created from a driver, it is destroyed when the driver is terminated.
- If it is created from a detached actor, it is killed when the detached actor is killed.

To keep the placement group alive regardless of its job or detached actor, specify
`lifetime="detached"`. For example:

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/placement_group_example.py
            :language: python
            :start-after: __detached_pg_start__
            :end-before: __detached_pg_end__

    .. tab-item:: Java

        The lifetime argument is not implemented for Java APIs yet.

Let's terminate the current script and start a new Python script. Call ``ray list placement-groups``, and you can see the placement group is not removed.

Note that the lifetime option is decoupled from the name. If we only specified
the name without specifying ``lifetime="detached"``, then the placement group can
only be retrieved as long as the original driver is still running.
It is recommended to always specify the name when creating the detached placement group.

[Advanced] Fault Tolerance
--------------------------

.. _ray-placement-group-ft-ref:

Rescheduling Bundles on a Dead Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If nodes that contain some bundles of a placement group die, all the bundles are rescheduled on different nodes by 
GCS (i.e., we try reserving resources again). This means that the initial creation of placement group is "atomic", 
but once it is created, there could be partial placement groups. 
Rescheduling bundles have higher scheduling priority than other placement group scheduling.

Provide Resources for Partially Lost Bundles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If there are not enough resources to schedule the partially lost bundles, 
the placement group waits, assuming Ray Autoscaler will start a new node to satisfy the resource requirements. 
If the additional resources cannot be provided (e.g., you don't use the Autoscaler or the Autoscaler hits the resource limit), 
the placement group remains in the partially created state indefinitely.

Fault Tolerance of Actors and Tasks that Use the Bundle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Actors and tasks that use the bundle (reserved resources) are rescheduled based on their :ref:`fault tolerant policy <fault-tolerance>` once the
bundle is recovered.

API Reference
-------------
:ref:`Placement Group API reference <ray-placement-group-ref>`


Out-Of-Memory Prevention
========================

If application tasks or actors consume a large amount of heap space, it can cause the node to run out of memory (OOM). When that happens, the operating system will start killing worker or raylet processes, disrupting the application. OOM may also stall metrics and if this happens on the head node, it may stall the :ref:`dashboard <observability-getting-started>` or other control processes and cause the cluster to become unusable.

In this section we will go over:

- What is the memory monitor and how it works

- How to enable and configure it

- How to use the memory monitor to detect and resolve memory issues

Also view :ref:`Debugging Out of Memory <troubleshooting-out-of-memory>` to learn how to troubleshoot out-of-memory issues.

.. _ray-oom-monitor:

What is the memory monitor?
---------------------------

The memory monitor is a component that runs within the :ref:`raylet <whitepaper>` process on each node. It periodically checks the memory usage, which includes the worker heap, the object store, and the raylet as described in :ref:`memory management <memory>`. If the combined usage exceeds a configurable threshold the raylet will kill a task or actor process to free up memory and prevent Ray from failing.

It's available on Linux and is tested with Ray running inside a container that is using cgroup v1/v2. If you encounter issues when running the memory monitor outside of a container, :ref:`file an issue or post a question <oom-questions>`.

How do I disable the memory monitor?
--------------------------------------

The memory monitor is enabled by default and can be disabled by setting the environment variable ``RAY_memory_monitor_refresh_ms`` to zero when Ray starts (e.g., RAY_memory_monitor_refresh_ms=0 ray start ...). 

How do I configure the memory monitor?
--------------------------------------

The memory monitor is controlled by the following environment variables:

- ``RAY_memory_monitor_refresh_ms (int, defaults to 250)`` is the interval to check memory usage and kill tasks or actors if needed. Task killing is disabled when this value is 0. The memory monitor selects and kills one task at a time and waits for it to be killed before choosing another one, regardless of how frequent the memory monitor runs.

- ``RAY_memory_usage_threshold (float, defaults to 0.95)`` is the threshold when the node is beyond the memory
  capacity. If the memory usage is above this fraction it will start killing processes to free up memory. Ranges from [0, 1].

Using the Memory Monitor
------------------------

.. _ray-oom-retry-policy:

Retry policy
~~~~~~~~~~~~

When a task or actor is killed by the memory monitor it will be retried with exponential backoff. There is a cap on the retry delay, which is 60 seconds. If tasks are killed by the memory monitor, it retries infinitely (not respecting :ref:`max_retries <task-fault-tolerance>`). If actors are killed by the memory monitor, it doesn't recreate the actor infinitely (It respects :ref:`max_restarts <actor-fault-tolerance>`, which is 0 by default).

Worker killing policy
~~~~~~~~~~~~~~~~~~~~~

The memory monitor avoids infinite loops of task retries by ensuring at least one task is able to run for each caller on each node. If it is unable to ensure this, the workload will fail with an OOM error. Note that this is only an issue for tasks, since the memory monitor will not indefinitely retry actors. If the workload fails, refer to :ref:`how to address memory issues <addressing-memory-issues>` on how to adjust the workload to make it pass. For code example, see the :ref:`last task <last-task-example>` example below.

When a worker needs to be killed, the policy first prioritizes tasks that are retriable, i.e. when :ref:`max_retries <task-fault-tolerance>` or :ref:`max_restarts <actor-fault-tolerance>` is > 0. This is done to minimize workload failure. Actors by default are not retriable since :ref:`max_restarts <actor-fault-tolerance>` defaults to 0. Therefore, by default, tasks are preferred to actors when it comes to what gets killed first.

When there are multiple callers that has created tasks, the policy will pick a task from the caller with the most number of running tasks. If two callers have the same number of tasks it picks the caller whose earliest task has a later start time. This is done to ensure fairness and allow each caller to make progress.

Amongst the tasks that share the same caller, the latest started task will be killed first.

Below is an example to demonstrate the policy. In the example we have a script that creates two tasks, which in turn creates four more tasks each. The tasks are colored such that each color forms a "group" of tasks where they belong to the same caller.

.. image:: ../images/oom_killer_example.svg
  :width: 1024
  :alt: Initial state of the task graph

If, at this point, the node runs out of memory, it will pick a task from the caller with the most number of tasks, and kill its task whose started the last:

.. image:: ../images/oom_killer_example_killed_one.svg
  :width: 1024
  :alt: Initial state of the task graph

If, at this point, the node still runs out of memory, the process will repeat:

.. image:: ../images/oom_killer_example_killed_two.svg
  :width: 1024
  :alt: Initial state of the task graph

.. _last-task-example:

.. dropdown:: Example: Workloads fails if the last task of the caller is killed

    Let's create an application oom.py that runs a single task that requires more memory than what is available. It is set to infinite retry by setting ``max_retries`` to -1.

    The worker killer policy sees that it is the last task of the caller, and will fail the workload when it kills the task as it is the last one for the caller, even when the task is set to retry forver.

    .. literalinclude:: ../doc_code/ray_oom_prevention.py
          :language: python
          :start-after: __last_task_start__
          :end-before: __last_task_end__


    Set ``RAY_event_stats_print_interval_ms=1000`` so it prints the worker kill summary every second, since by default it prints every minute.

    .. code-block:: bash

        RAY_event_stats_print_interval_ms=1000 python oom.py

        (raylet) node_manager.cc:3040: 1 Workers (tasks / actors) killed due to memory pressure (OOM), 0 Workers crashed due to other reasons at node (ID: 2c82620270df6b9dd7ae2791ef51ee4b5a9d5df9f795986c10dd219c, IP: 172.31.183.172) over the last time period. To see more information about the Workers killed on this node, use `ray logs raylet.out -ip 172.31.183.172`
        (raylet) 
        (raylet) Refer to the documentation on how to address the out of memory issue: https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html. Consider provisioning more memory on this node or reducing task parallelism by requesting more CPUs per task. To adjust the kill threshold, set the environment variable `RAY_memory_usage_threshold` when starting Ray. To disable worker killing, set the environment variable `RAY_memory_monitor_refresh_ms` to zero.
                task failed with OutOfMemoryError, which is expected
                Verify the task was indeed executed twice via ``task_oom_retry``:


.. dropdown:: Example: memory monitor prefers to kill a retriable task

    Let's first start ray and specify the memory threshold.

    .. code-block:: bash

        RAY_memory_usage_threshold=0.4 ray start --head


    Let's create an application two_actors.py that submits two actors, where the first one is retriable and the second one is non-retriable.

    .. literalinclude:: ../doc_code/ray_oom_prevention.py
          :language: python
          :start-after: __two_actors_start__
          :end-before: __two_actors_end__


    Run the application to see that only the first actor was killed.

    .. code-block:: bash

        $ python two_actors.py
        
        First started actor, which is retriable, was killed by the memory monitor.
        Second started actor, which is not-retriable, finished.

.. _addressing-memory-issues:

Addressing memory issues
------------------------

When the application fails due to OOM, consider reducing the memory usage of the tasks and actors, increasing the memory capacity of the node, or :ref:`limit the number of concurrently running tasks <core-patterns-limit-running-tasks>`.


.. _oom-questions:

Questions or Issues?
--------------------

.. include:: /_includes/_help.rst


.. _memory:

Memory Management
=================

This page describes how memory management works in Ray.

Also view :ref:`Debugging Out of Memory <troubleshooting-out-of-memory>` to learn how to troubleshoot out-of-memory issues.

Concepts
~~~~~~~~

There are several ways that Ray applications use memory:

..
  https://docs.google.com/drawings/d/1wHHnAJZ-NsyIv3TUXQJTYpPz6pjB6PUm2M40Zbfb1Ak/edit

.. image:: ../images/memory.svg

Ray system memory: this is memory used internally by Ray
  - **GCS**: memory used for storing the list of nodes and actors present in the cluster. The amount of memory used for these purposes is typically quite small.
  - **Raylet**: memory used by the C++ raylet process running on each node. This cannot be controlled, but is typically quite small.

Application memory: this is memory used by your application
  - **Worker heap**: memory used by your application (e.g., in Python code or TensorFlow), best measured as the *resident set size (RSS)* of your application minus its *shared memory usage (SHR)* in commands such as ``top``. The reason you need to subtract *SHR* is that object store shared memory is reported by the OS as shared with each worker. Not subtracting *SHR* will result in double counting memory usage.
  - **Object store memory**: memory used when your application creates objects in the object store via ``ray.put`` and when it returns values from remote functions. Objects are reference counted and evicted when they fall out of scope. An object store server runs on each node. By default, when starting an instance, Ray reserves 30% of available memory. The size of the object store can be controlled by `--object-store-memory <https://docs.ray.io/en/master/cluster/cli.html#cmdoption-ray-start-object-store-memory>`_. The memory is by default allocated to ``/dev/shm`` (shared memory) for Linux. For MacOS, Ray uses ``/tmp`` (disk), which can impact the performance compared to Linux. In Ray 1.3+, objects are :ref:`spilled to disk <object-spilling>` if the object store fills up.
  - **Object store shared memory**: memory used when your application reads objects via ``ray.get``. Note that if an object is already present on the node, this does not cause additional allocations. This allows large objects to be efficiently shared among many actors and tasks.

ObjectRef Reference Counting
----------------------------

Ray implements distributed reference counting so that any ``ObjectRef`` in scope in the cluster is pinned in the object store. This includes local python references, arguments to pending tasks, and IDs serialized inside of other objects.

.. _debug-with-ray-memory:

Debugging using 'ray memory'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ray memory`` command can be used to help track down what ``ObjectRef`` references are in scope and may be causing an ``ObjectStoreFullError``.

Running ``ray memory`` from the command line while a Ray application is running will give you a dump of all of the ``ObjectRef`` references that are currently held by the driver, actors, and tasks in the cluster.

.. code-block:: text

  ======== Object references status: 2021-02-23 22:02:22.072221 ========
  Grouping by node address...        Sorting by object size...


  --- Summary for node address: 192.168.0.15 ---
  Mem Used by Objects  Local References  Pinned Count  Pending Tasks  Captured in Objects  Actor Handles
  287 MiB              4                 0             0              1                    0

  --- Object references for node address: 192.168.0.15 ---
  IP Address    PID    Type    Object Ref                                                Size    Reference Type      Call Site
  192.168.0.15  6465   Driver  ffffffffffffffffffffffffffffffffffffffff0100000001000000  15 MiB  LOCAL_REFERENCE     (put object)
                                                                                                                    | test.py:
                                                                                                                    <module>:17

  192.168.0.15  6465   Driver  a67dc375e60ddd1affffffffffffffffffffffff0100000001000000  15 MiB  LOCAL_REFERENCE     (task call)
                                                                                                                    | test.py:
                                                                                                                    :<module>:18

  192.168.0.15  6465   Driver  ffffffffffffffffffffffffffffffffffffffff0100000002000000  18 MiB  CAPTURED_IN_OBJECT  (put object)  |
                                                                                                                     test.py:
                                                                                                                    <module>:19

  192.168.0.15  6465   Driver  ffffffffffffffffffffffffffffffffffffffff0100000004000000  21 MiB  LOCAL_REFERENCE     (put object)  |
                                                                                                                     test.py:
                                                                                                                    <module>:20

  192.168.0.15  6465   Driver  ffffffffffffffffffffffffffffffffffffffff0100000003000000  218 MiB  LOCAL_REFERENCE     (put object)  |
                                                                                                                    test.py:
                                                                                                                    <module>:20

  --- Aggregate object store stats across all nodes ---
  Plasma memory usage 0 MiB, 4 objects, 0.0% full


Each entry in this output corresponds to an ``ObjectRef`` that's currently pinning an object in the object store along with where the reference is (in the driver, in a worker, etc.), what type of reference it is (see below for details on the types of references), the size of the object in bytes, the process ID and IP address where the object was instantiated, and where in the application the reference was created.

``ray memory`` comes with features to make the memory debugging experience more effective. For example, you can add arguments ``sort-by=OBJECT_SIZE`` and ``group-by=STACK_TRACE``, which may be particularly helpful for tracking down the line of code where a memory leak occurs. You can see the full suite of options by running ``ray memory --help``.

There are five types of references that can keep an object pinned:

**1. Local ObjectRef references**

.. testcode::

  import ray

  @ray.remote
  def f(arg):
      return arg

  a = ray.put(None)
  b = f.remote(None)

In this example, we create references to two objects: one that is ``ray.put()`` in the object store and another that's the return value from ``f.remote()``.

.. code-block:: text

  --- Summary for node address: 192.168.0.15 ---
  Mem Used by Objects  Local References  Pinned Count  Pending Tasks  Captured in Objects  Actor Handles
  30 MiB               2                 0             0              0                    0

  --- Object references for node address: 192.168.0.15 ---
  IP Address    PID    Type    Object Ref                                                Size    Reference Type      Call Site
  192.168.0.15  6867   Driver  ffffffffffffffffffffffffffffffffffffffff0100000001000000  15 MiB  LOCAL_REFERENCE     (put object)  |
                                                                                                                    test.py:
                                                                                                                    <module>:12

  192.168.0.15  6867   Driver  a67dc375e60ddd1affffffffffffffffffffffff0100000001000000  15 MiB  LOCAL_REFERENCE     (task call)
                                                                                                                    | test.py:
                                                                                                                    :<module>:13

In the output from ``ray memory``, we can see that each of these is marked as a ``LOCAL_REFERENCE`` in the driver process, but the annotation in the "Reference Creation Site" indicates that the first was created as a "put object" and the second from a "task call."

**2. Objects pinned in memory**

.. testcode::

  import numpy as np

  a = ray.put(np.zeros(1))
  b = ray.get(a)
  del a

In this example, we create a ``numpy`` array and then store it in the object store. Then, we fetch the same numpy array from the object store and delete its ``ObjectRef``. In this case, the object is still pinned in the object store because the deserialized copy (stored in ``b``) points directly to the memory in the object store.

.. code-block:: text

  --- Summary for node address: 192.168.0.15 ---
  Mem Used by Objects  Local References  Pinned Count  Pending Tasks  Captured in Objects  Actor Handles
  243 MiB              0                 1             0              0                    0

  --- Object references for node address: 192.168.0.15 ---
  IP Address    PID    Type    Object Ref                                                Size    Reference Type      Call Site
  192.168.0.15  7066   Driver  ffffffffffffffffffffffffffffffffffffffff0100000001000000  243 MiB  PINNED_IN_MEMORY   test.
                                                                                                                    py:<module>:19

The output from ``ray memory`` displays this as the object being ``PINNED_IN_MEMORY``. If we ``del b``, the reference can be freed.

**3. Pending task references**

.. testcode::

  @ray.remote
  def f(arg):
      while True:
          pass

  a = ray.put(None)
  b = f.remote(a)

In this example, we first create an object via ``ray.put()`` and then submit a task that depends on the object.

.. code-block:: text

  --- Summary for node address: 192.168.0.15 ---
  Mem Used by Objects  Local References  Pinned Count  Pending Tasks  Captured in Objects  Actor Handles
  25 MiB               1                 1             1              0                    0

  --- Object references for node address: 192.168.0.15 ---
  IP Address    PID    Type    Object Ref                                                Size    Reference Type      Call Site
  192.168.0.15  7207   Driver  a67dc375e60ddd1affffffffffffffffffffffff0100000001000000  ?       LOCAL_REFERENCE     (task call)
                                                                                                                      | test.py:
                                                                                                                    :<module>:29

  192.168.0.15  7241   Worker  ffffffffffffffffffffffffffffffffffffffff0100000001000000  10 MiB  PINNED_IN_MEMORY    (deserialize task arg)
                                                                                                                      __main__.f

  192.168.0.15  7207   Driver  ffffffffffffffffffffffffffffffffffffffff0100000001000000  15 MiB  USED_BY_PENDING_TASK  (put object)  |
                                                                                                                    test.py:
                                                                                                                    <module>:28

While the task is running, we see that ``ray memory`` shows both a ``LOCAL_REFERENCE`` and a ``USED_BY_PENDING_TASK`` reference for the object in the driver process. The worker process also holds a reference to the object because the Python ``arg`` is directly referencing the memory in the plasma, so it can't be evicted; therefore it is ``PINNED_IN_MEMORY``.

**4. Serialized ObjectRef references**

.. testcode::

  @ray.remote
  def f(arg):
      while True:
          pass

  a = ray.put(None)
  b = f.remote([a])

In this example, we again create an object via ``ray.put()``, but then pass it to a task wrapped in another object (in this case, a list).

.. code-block:: text

  --- Summary for node address: 192.168.0.15 ---
  Mem Used by Objects  Local References  Pinned Count  Pending Tasks  Captured in Objects  Actor Handles
  15 MiB               2                 0             1              0                    0

  --- Object references for node address: 192.168.0.15 ---
  IP Address    PID    Type    Object Ref                                                Size    Reference Type      Call Site
  192.168.0.15  7411   Worker  ffffffffffffffffffffffffffffffffffffffff0100000001000000  ?       LOCAL_REFERENCE     (deserialize task arg)
                                                                                                                      __main__.f

  192.168.0.15  7373   Driver  a67dc375e60ddd1affffffffffffffffffffffff0100000001000000  ?       LOCAL_REFERENCE     (task call)
                                                                                                                    | test.py:
                                                                                                                    :<module>:38

  192.168.0.15  7373   Driver  ffffffffffffffffffffffffffffffffffffffff0100000001000000  15 MiB  USED_BY_PENDING_TASK  (put object)
                                                                                                                    | test.py:
                                                                                                                    <module>:37

Now, both the driver and the worker process running the task hold a ``LOCAL_REFERENCE`` to the object in addition to it being ``USED_BY_PENDING_TASK`` on the driver. If this was an actor task, the actor could even hold a ``LOCAL_REFERENCE`` after the task completes by storing the ``ObjectRef`` in a member variable.

**5. Captured ObjectRef references**

.. testcode::

  a = ray.put(None)
  b = ray.put([a])
  del a

In this example, we first create an object via ``ray.put()``, then capture its ``ObjectRef`` inside of another ``ray.put()`` object, and delete the first ``ObjectRef``. In this case, both objects are still pinned.

.. code-block:: text

  --- Summary for node address: 192.168.0.15 ---
  Mem Used by Objects  Local References  Pinned Count  Pending Tasks  Captured in Objects  Actor Handles
  233 MiB              1                 0             0              1                    0

  --- Object references for node address: 192.168.0.15 ---
  IP Address    PID    Type    Object Ref                                                Size    Reference Type      Call Site
  192.168.0.15  7473   Driver  ffffffffffffffffffffffffffffffffffffffff0100000001000000  15 MiB  CAPTURED_IN_OBJECT  (put object)  |
                                                                                                                    test.py:
                                                                                                                    <module>:41

  192.168.0.15  7473   Driver  ffffffffffffffffffffffffffffffffffffffff0100000002000000  218 MiB  LOCAL_REFERENCE     (put object)  |
                                                                                                                    test.py:
                                                                                                                    <module>:42

In the output of ``ray memory``, we see that the second object displays as a normal ``LOCAL_REFERENCE``, but the first object is listed as ``CAPTURED_IN_OBJECT``.

.. _memory-aware-scheduling:

Memory Aware Scheduling
~~~~~~~~~~~~~~~~~~~~~~~

By default, Ray does not take into account the potential memory usage of a task or actor when scheduling. This is simply because it cannot estimate ahead of time how much memory is required. However, if you know how much memory a task or actor requires, you can specify it in the resource requirements of its ``ray.remote`` decorator to enable memory-aware scheduling:

.. important::

  Specifying a memory requirement does NOT impose any limits on memory usage. The requirements are used for admission control during scheduling only (similar to how CPU scheduling works in Ray). It is up to the task itself to not use more memory than it requested.

To tell the Ray scheduler a task or actor requires a certain amount of available memory to run, set the ``memory`` argument. The Ray scheduler will then reserve the specified amount of available memory during scheduling, similar to how it handles CPU and GPU resources:

.. testcode::

  # reserve 500MiB of available memory to place this task
  @ray.remote(memory=500 * 1024 * 1024)
  def some_function(x):
      pass

  # reserve 2.5GiB of available memory to place this actor
  @ray.remote(memory=2500 * 1024 * 1024)
  class SomeActor:
      def __init__(self, a, b):
          pass

In the above example, the memory quota is specified statically by the decorator, but you can also set them dynamically at runtime using ``.options()`` as follows:

.. testcode::

  # override the memory quota to 100MiB when submitting the task
  some_function.options(memory=100 * 1024 * 1024).remote(x=1)

  # override the memory quota to 1GiB when creating the actor
  SomeActor.options(memory=1000 * 1024 * 1024).remote(a=1, b=2)

Questions or Issues?
--------------------

.. include:: /_includes/_help.rst


Out-of-band Communication
=========================

Typically, Ray actor communication is done through actor method calls and data is shared through the distributed object store.
However, in some use cases out-of-band communication can be useful.

Wrapping Library Processes
--------------------------
Many libraries already have mature, high-performance internal communication stacks and
they leverage Ray as a language-integrated actor scheduler.
The actual communication between actors is mostly done out-of-band using existing communication stacks.
For example, Horovod-on-Ray uses NCCL or MPI-based collective communications, and RayDP uses Spark's internal RPC and object manager.
See `Ray Distributed Library Patterns <https://www.anyscale.com/blog/ray-distributed-library-patterns>`_ for more details.

Ray Collective
--------------
Ray's collective communication library (\ ``ray.util.collective``\ ) allows efficient out-of-band collective and point-to-point communication between distributed CPUs or GPUs.
See :ref:`Ray Collective <ray-collective>` for more details.

HTTP Server
-----------
You can start a http server inside the actor and expose http endpoints to clients
so users outside of the ray cluster can communicate with the actor.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/actor-http-server.py

Similarly, you can expose other types of servers as well (e.g., gRPC servers).

Limitations
-----------

When using out-of-band communication with Ray actors, keep in mind that Ray does not manage the calls between actors. This means that functionality like distributed reference counting will not work with out-of-band communication, so you should take care not to pass object references in this way.


Terminating Actors
==================

Actor processes will be terminated automatically when all copies of the
actor handle have gone out of scope in Python, or if the original creator
process dies.

Note that automatic termination of actors is not yet supported in Java or C++.

.. _ray-kill-actors:

Manual termination via an actor handle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In most cases, Ray will automatically terminate actors that have gone out of
scope, but you may sometimes need to terminate an actor forcefully. This should
be reserved for cases where an actor is unexpectedly hanging or leaking
resources, and for :ref:`detached actors <actor-lifetimes>`, which must be
manually destroyed.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import ray

            @ray.remote
            class Actor:
                pass

            actor_handle = Actor.remote()

            ray.kill(actor_handle)
            # This will not go through the normal Python sys.exit
            # teardown logic, so any exit handlers installed in
            # the actor using ``atexit`` will not be called.


    .. tab-item:: Java

        .. code-block:: java

            actorHandle.kill();
            // This will not go through the normal Java System.exit teardown logic, so any
            // shutdown hooks installed in the actor using ``Runtime.addShutdownHook(...)`` will
            // not be called.

    .. tab-item:: C++

        .. code-block:: c++

            actor_handle.Kill();
            // This will not go through the normal C++ std::exit
            // teardown logic, so any exit handlers installed in
            // the actor using ``std::atexit`` will not be called.


This will cause the actor to immediately exit its process, causing any current,
pending, and future tasks to fail with a ``RayActorError``. If you would like
Ray to :ref:`automatically restart <fault-tolerance-actors>` the actor, make sure to set a nonzero
``max_restarts`` in the ``@ray.remote`` options for the actor, then pass the
flag ``no_restart=False`` to ``ray.kill``.

For :ref:`named and detached actors <actor-lifetimes>`, calling ``ray.kill`` on
an actor handle destroys the actor and allow the name to be reused.

Use `ray list actors --detail` from :ref:`State API <state-api-overview-ref>` to see the death cause of dead actors:

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list actors --detail

.. code-block:: bash

  ---
  -   actor_id: e8702085880657b355bf7ef001000000
      class_name: Actor
      state: DEAD
      job_id: '01000000'
      name: ''
      node_id: null
      pid: 0
      ray_namespace: dbab546b-7ce5-4cbb-96f1-d0f64588ae60
      serialized_runtime_env: '{}'
      required_resources: {}
      death_cause:
          actor_died_error_context: # <---- You could see the error message w.r.t why the actor exits. 
              error_message: The actor is dead because `ray.kill` killed it.
              owner_id: 01000000ffffffffffffffffffffffffffffffffffffffffffffffff
              owner_ip_address: 127.0.0.1
              ray_namespace: dbab546b-7ce5-4cbb-96f1-d0f64588ae60
              class_name: Actor
              actor_id: e8702085880657b355bf7ef001000000
              never_started: true
              node_ip_address: ''
              pid: 0
              name: ''
      is_detached: false
      placement_group_id: null
      repr_name: ''


Manual termination within the actor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If necessary, you can manually terminate an actor from within one of the actor methods.
This will kill the actor process and release resources associated/assigned to the actor.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            @ray.remote
            class Actor:
                def exit(self):
                    ray.actor.exit_actor()

            actor = Actor.remote()
            actor.exit.remote()

        This approach should generally not be necessary as actors are automatically garbage
        collected. The ``ObjectRef`` resulting from the task can be waited on to wait
        for the actor to exit (calling ``ray.get()`` on it will raise a ``RayActorError``).

    .. tab-item:: Java

        .. code-block:: java

            Ray.exitActor();

        Garbage collection for actors haven't been implemented yet, so this is currently the
        only way to terminate an actor gracefully. The ``ObjectRef`` resulting from the task
        can be waited on to wait for the actor to exit (calling ``ObjectRef::get`` on it will
        throw a ``RayActorException``).

    .. tab-item:: C++

        .. code-block:: c++

            ray::ExitActor();

        Garbage collection for actors haven't been implemented yet, so this is currently the
        only way to terminate an actor gracefully. The ``ObjectRef`` resulting from the task
        can be waited on to wait for the actor to exit (calling ``ObjectRef::Get`` on it will
        throw a ``RayActorException``).

Note that this method of termination waits until any previously submitted
tasks finish executing and then exits the process gracefully with sys.exit.


    
You could see the actor is dead as a result of the user's `exit_actor()` call:

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list actors --detail

.. code-block:: bash

  ---
  -   actor_id: 070eb5f0c9194b851bb1cf1602000000
      class_name: Actor
      state: DEAD
      job_id: '02000000'
      name: ''
      node_id: 47ccba54e3ea71bac244c015d680e202f187fbbd2f60066174a11ced
      pid: 47978
      ray_namespace: 18898403-dda0-485a-9c11-e9f94dffcbed
      serialized_runtime_env: '{}'
      required_resources: {}
      death_cause:
          actor_died_error_context:
              error_message: 'The actor is dead because its worker process has died.
                  Worker exit type: INTENDED_USER_EXIT Worker exit detail: Worker exits
                  by an user request. exit_actor() is called.'
              owner_id: 02000000ffffffffffffffffffffffffffffffffffffffffffffffff
              owner_ip_address: 127.0.0.1
              node_ip_address: 127.0.0.1
              pid: 47978
              ray_namespace: 18898403-dda0-485a-9c11-e9f94dffcbed
              class_name: Actor
              actor_id: 070eb5f0c9194b851bb1cf1602000000
              name: ''
              never_started: false
      is_detached: false
      placement_group_id: null
      repr_name: ''

AsyncIO / Concurrency for Actors
================================

Within a single actor process, it is possible to execute concurrent threads.

Ray offers two types of concurrency within an actor:

 * :ref:`async execution <async-actors>`
 * :ref:`threading <threaded-actors>`


Keep in mind that the Python's `Global Interpreter Lock (GIL) <https://wiki.python.org/moin/GlobalInterpreterLock>`_ will only allow one thread of Python code running at once.

This means if you are just parallelizing Python code, you won't get true parallelism. If you call Numpy, Cython, Tensorflow, or PyTorch code, these libraries will release the GIL when calling into C/C++ functions.

**Neither the** :ref:`threaded-actors` nor :ref:`async-actors` **model will allow you to bypass the GIL.**

.. _async-actors:

AsyncIO for Actors
------------------

Since Python 3.5, it is possible to write concurrent code using the
``async/await`` `syntax <https://docs.python.org/3/library/asyncio.html>`__.
Ray natively integrates with asyncio. You can use ray alongside with popular
async frameworks like aiohttp, aioredis, etc.

.. testcode::

    import ray
    import asyncio

    @ray.remote
    class AsyncActor:
        # multiple invocation of this method can be running in
        # the event loop at the same time
        async def run_concurrent(self):
            print("started")
            await asyncio.sleep(2) # concurrent workload here
            print("finished")

    actor = AsyncActor.remote()

    # regular ray.get
    ray.get([actor.run_concurrent.remote() for _ in range(4)])

    # async ray.get
    async def async_get():
        await actor.run_concurrent.remote()
    asyncio.run(async_get())

.. testoutput::
    :options: +MOCK

    (AsyncActor pid=40293) started
    (AsyncActor pid=40293) started
    (AsyncActor pid=40293) started
    (AsyncActor pid=40293) started
    (AsyncActor pid=40293) finished
    (AsyncActor pid=40293) finished
    (AsyncActor pid=40293) finished
    (AsyncActor pid=40293) finished

.. testcode::
    :hide:

    # NOTE: The outputs from the previous code block can show up in subsequent tests.
    # To prevent flakiness, we wait for the async calls finish.
    import time
    print("Sleeping...")
    time.sleep(3)

.. testoutput::

    ...

ObjectRefs as asyncio.Futures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ObjectRefs can be translated to asyncio.Futures. This feature
make it possible to ``await`` on ray futures in existing concurrent
applications.

Instead of:

.. testcode::

    import ray

    @ray.remote
    def some_task():
        return 1

    ray.get(some_task.remote())
    ray.wait([some_task.remote()])

you can do:

.. testcode::

    import ray
    import asyncio

    @ray.remote
    def some_task():
        return 1

    async def await_obj_ref():
        await some_task.remote()
        await asyncio.wait([some_task.remote()])

    asyncio.run(await_obj_ref())

Please refer to `asyncio doc <https://docs.python.org/3/library/asyncio-task.html>`__
for more `asyncio` patterns including timeouts and ``asyncio.gather``.

If you need to directly access the future object, you can call:

.. testcode::

    import asyncio

    async def convert_to_asyncio_future():
        ref = some_task.remote()
        fut: asyncio.Future = asyncio.wrap_future(ref.future())
        print(await fut)
    asyncio.run(convert_to_asyncio_future())

.. testoutput::

    1

.. _async-ref-to-futures:

ObjectRefs as concurrent.futures.Futures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ObjectRefs can also be wrapped into ``concurrent.futures.Future`` objects. This
is useful for interfacing with existing ``concurrent.futures`` APIs:

.. testcode::

    import concurrent

    refs = [some_task.remote() for _ in range(4)]
    futs = [ref.future() for ref in refs]
    for fut in concurrent.futures.as_completed(futs):
        assert fut.done()
        print(fut.result())

.. testoutput::

    1
    1
    1
    1

Defining an Async Actor
~~~~~~~~~~~~~~~~~~~~~~~

By using `async` method definitions, Ray will automatically detect whether an actor support `async` calls or not.

.. testcode::

    import asyncio

    @ray.remote
    class AsyncActor:
        async def run_task(self):
            print("started")
            await asyncio.sleep(2) # Network, I/O task here
            print("ended")

    actor = AsyncActor.remote()
    # All 5 tasks should start at once. After 2 second they should all finish.
    # they should finish at the same time
    ray.get([actor.run_task.remote() for _ in range(5)])

.. testoutput::
    :options: +MOCK

    (AsyncActor pid=3456) started
    (AsyncActor pid=3456) started
    (AsyncActor pid=3456) started
    (AsyncActor pid=3456) started
    (AsyncActor pid=3456) started
    (AsyncActor pid=3456) ended
    (AsyncActor pid=3456) ended
    (AsyncActor pid=3456) ended
    (AsyncActor pid=3456) ended
    (AsyncActor pid=3456) ended

Under the hood, Ray runs all of the methods inside a single python event loop.
Please note that running blocking ``ray.get`` or ``ray.wait`` inside async
actor method is not allowed, because ``ray.get`` will block the execution
of the event loop.

In async actors, only one task can be running at any point in time (though tasks can be multi-plexed). There will be only one thread in AsyncActor! See :ref:`threaded-actors` if you want a threadpool.

Setting concurrency in Async Actors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can set the number of "concurrent" task running at once using the
``max_concurrency`` flag. By default, 1000 tasks can be running concurrently.

.. testcode::

    import asyncio

    @ray.remote
    class AsyncActor:
        async def run_task(self):
            print("started")
            await asyncio.sleep(1) # Network, I/O task here
            print("ended")

    actor = AsyncActor.options(max_concurrency=2).remote()

    # Only 2 tasks will be running concurrently. Once 2 finish, the next 2 should run.
    ray.get([actor.run_task.remote() for _ in range(8)])

.. testoutput::
    :options: +MOCK

    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) started
    (AsyncActor pid=5859) ended
    (AsyncActor pid=5859) ended

.. _threaded-actors:

Threaded Actors
---------------

Sometimes, asyncio is not an ideal solution for your actor. For example, you may
have one method that performs some computation heavy task while blocking the event loop, not giving up control via ``await``. This would hurt the performance of an Async Actor because Async Actors can only execute 1 task at a time and rely on ``await`` to context switch.


Instead, you can use the ``max_concurrency`` Actor options without any async methods, allowng you to achieve threaded concurrency (like a thread pool).


.. warning::
    When there is at least one ``async def`` method in actor definition, Ray
    will recognize the actor as AsyncActor instead of ThreadedActor.


.. testcode::

    @ray.remote
    class ThreadedActor:
        def task_1(self): print("I'm running in a thread!")
        def task_2(self): print("I'm running in another thread!")

    a = ThreadedActor.options(max_concurrency=2).remote()
    ray.get([a.task_1.remote(), a.task_2.remote()])

.. testoutput::
    :options: +MOCK

    (ThreadedActor pid=4822) I'm running in a thread!
    (ThreadedActor pid=4822) I'm running in another thread!

Each invocation of the threaded actor will be running in a thread pool. The size of the threadpool is limited by the ``max_concurrency`` value.

AsyncIO for Remote Tasks
------------------------

We don't support asyncio for remote tasks. The following snippet will fail:

.. testcode::
    :skipif: True

    @ray.remote
    async def f():
        pass

Instead, you can wrap the ``async`` function with a wrapper to run the task synchronously:

.. testcode::

    async def f():
        pass

    @ray.remote
    def wrapper():
        import asyncio
        asyncio.run(f())


Named Actors
============

An actor can be given a unique name within their :ref:`namespace <namespaces-guide>`.
This allows you to retrieve the actor from any job in the Ray cluster.
This can be useful if you cannot directly
pass the actor handle to the task that needs it, or if you are trying to
access an actor launched by another driver.
Note that the actor will still be garbage-collected if no handles to it
exist. See :ref:`actor-lifetimes` for more details.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import ray

            @ray.remote
            class Counter:
                pass

            # Create an actor with a name
            counter = Counter.options(name="some_name").remote()

            # Retrieve the actor later somewhere
            counter = ray.get_actor("some_name")

    .. tab-item:: Java

        .. code-block:: java

            // Create an actor with a name.
            ActorHandle<Counter> counter = Ray.actor(Counter::new).setName("some_name").remote();

            ...

            // Retrieve the actor later somewhere
            Optional<ActorHandle<Counter>> counter = Ray.getActor("some_name");
            Assert.assertTrue(counter.isPresent());

    .. tab-item:: C++

        .. code-block:: c++

            // Create an actor with a globally unique name
            ActorHandle<Counter> counter = ray::Actor(CreateCounter).SetGlobalName("some_name").Remote();

            ...

            // Retrieve the actor later somewhere
            boost::optional<ray::ActorHandle<Counter>> counter = ray::GetGlobalActor("some_name");

        We also support non-global named actors in C++, which means that the actor name is only valid within the job and the actor cannot be accessed from another job

        .. code-block:: c++

            // Create an actor with a job-scope-unique name
            ActorHandle<Counter> counter = ray::Actor(CreateCounter).SetName("some_name").Remote();

            ...

            // Retrieve the actor later somewhere in the same job
            boost::optional<ray::ActorHandle<Counter>> counter = ray::GetActor("some_name");

.. note::

     Named actors are scoped by namespace. If no namespace is assigned, they will
     be placed in an anonymous namespace by default.

.. tab-set::

    .. tab-item:: Python

        .. testcode::
            :skipif: True

            import ray

            @ray.remote
            class Actor:
              pass

            # driver_1.py
            # Job 1 creates an actor, "orange" in the "colors" namespace.
            ray.init(address="auto", namespace="colors")
            Actor.options(name="orange", lifetime="detached").remote()

            # driver_2.py
            # Job 2 is now connecting to a different namespace.
            ray.init(address="auto", namespace="fruit")
            # This fails because "orange" was defined in the "colors" namespace.
            ray.get_actor("orange")
            # You can also specify the namespace explicitly.
            ray.get_actor("orange", namespace="colors")

            # driver_3.py
            # Job 3 connects to the original "colors" namespace
            ray.init(address="auto", namespace="colors")
            # This returns the "orange" actor we created in the first job.
            ray.get_actor("orange")

    .. tab-item:: Java

        .. code-block:: java

            import ray

            class Actor {
            }

            // Driver1.java
            // Job 1 creates an actor, "orange" in the "colors" namespace.
            System.setProperty("ray.job.namespace", "colors");
            Ray.init();
            Ray.actor(Actor::new).setName("orange").remote();

            // Driver2.java
            // Job 2 is now connecting to a different namespace.
            System.setProperty("ray.job.namespace", "fruits");
            Ray.init();
            // This fails because "orange" was defined in the "colors" namespace.
            Optional<ActorHandle<Actor>> actor = Ray.getActor("orange");
            Assert.assertFalse(actor.isPresent());  // actor.isPresent() is false.

            // Driver3.java
            System.setProperty("ray.job.namespace", "colors");
            Ray.init();
            // This returns the "orange" actor we created in the first job.
            Optional<ActorHandle<Actor>> actor = Ray.getActor("orange");
            Assert.assertTrue(actor.isPresent());  // actor.isPresent() is true.

Get-Or-Create a Named Actor
---------------------------

A common use case is to create an actor only if it doesn't exist.
Ray provides a ``get_if_exists`` option for actor creation that does this out of the box.
This method is available after you set a name for the actor via ``.options()``.

If the actor already exists, a handle to the actor will be returned
and the arguments will be ignored. Otherwise, a new actor will be
created with the specified arguments.

.. tab-set::

    .. tab-item:: Python

        .. literalinclude:: ../doc_code/get_or_create.py

    .. tab-item:: Java

        .. code-block:: java

            // This feature is not yet available in Java.

    .. tab-item:: C++

        .. code-block:: c++

            // This feature is not yet available in C++.


.. _actor-lifetimes:

Actor Lifetimes
---------------

Separately, actor lifetimes can be decoupled from the job, allowing an actor to persist even after the driver process of the job exits. We call these actors *detached*.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            counter = Counter.options(name="CounterActor", lifetime="detached").remote()

        The ``CounterActor`` will be kept alive even after the driver running above script
        exits. Therefore it is possible to run the following script in a different
        driver:

        .. testcode::

            counter = ray.get_actor("CounterActor")

        Note that an actor can be named but not detached. If we only specified the
        name without specifying ``lifetime="detached"``, then the CounterActor can
        only be retrieved as long as the original driver is still running.

    .. tab-item:: Java

        .. code-block:: java

            System.setProperty("ray.job.namespace", "lifetime");
            Ray.init();
            ActorHandle<Counter> counter = Ray.actor(Counter::new).setName("some_name").setLifetime(ActorLifetime.DETACHED).remote();

        The CounterActor will be kept alive even after the driver running above process
        exits. Therefore it is possible to run the following code in a different
        driver:

        .. code-block:: java

            System.setProperty("ray.job.namespace", "lifetime");
            Ray.init();
            Optional<ActorHandle<Counter>> counter = Ray.getActor("some_name");
            Assert.assertTrue(counter.isPresent());

    .. tab-item:: C++

        Customizing lifetime of an actor hasn't been implemented in C++ yet.


Unlike normal actors, detached actors are not automatically garbage-collected by Ray.
Detached actors must be manually destroyed once you are sure that they are no
longer needed. To do this, use ``ray.kill`` to :ref:`manually terminate <ray-kill-actors>` the actor.
After this call, the actor's name may be reused.


Utility Classes
===============

Actor Pool
~~~~~~~~~~

.. tab-set::

    .. tab-item:: Python

        The ``ray.util`` module contains a utility class, ``ActorPool``.
        This class is similar to multiprocessing.Pool and lets you schedule Ray tasks over a fixed pool of actors.

        .. literalinclude:: ../doc_code/actor-pool.py

        See the :class:`package reference <ray.util.ActorPool>` for more information.

    .. tab-item:: Java

        Actor pool hasn't been implemented in Java yet.

    .. tab-item:: C++

        Actor pool hasn't been implemented in C++ yet.

Message passing using Ray Queue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes just using one signal to synchronize is not enough. If you need to send data among many tasks or
actors, you can use :class:`ray.util.queue.Queue <ray.util.queue.Queue>`.

.. literalinclude:: ../doc_code/actor-queue.py

Ray's Queue API has a similar API to Python's ``asyncio.Queue`` and ``queue.Queue``.


Limiting Concurrency Per-Method with Concurrency Groups
=======================================================

Besides setting the max concurrency overall for an actor, Ray allows methods to be separated into *concurrency groups*, each with its own threads(s). This allows you to limit the concurrency per-method, e.g., allow a health-check method to be given its own concurrency quota separate from request serving methods.

.. tip:: Concurrency groups work with both asyncio and threaded actors. The syntax is the same.

.. _defining-concurrency-groups:

Defining Concurrency Groups
---------------------------

This defines two concurrency groups, "io" with max concurrency = 2 and
"compute" with max concurrency = 4.  The methods ``f1`` and ``f2`` are
placed in the "io" group, and the methods ``f3`` and ``f4`` are placed
into the "compute" group. Note that there is always a default
concurrency group for actors, which has a default concurrency of 1000
AsyncIO actors and 1 otherwise.

.. tab-set::

    .. tab-item:: Python

        You can define concurrency groups for actors using the ``concurrency_group`` decorator argument:

        .. testcode::

            import ray

            @ray.remote(concurrency_groups={"io": 2, "compute": 4})
            class AsyncIOActor:
                def __init__(self):
                    pass

                @ray.method(concurrency_group="io")
                async def f1(self):
                    pass

                @ray.method(concurrency_group="io")
                async def f2(self):
                    pass

                @ray.method(concurrency_group="compute")
                async def f3(self):
                    pass

                @ray.method(concurrency_group="compute")
                async def f4(self):
                    pass

                async def f5(self):
                    pass

            a = AsyncIOActor.remote()
            a.f1.remote()  # executed in the "io" group.
            a.f2.remote()  # executed in the "io" group.
            a.f3.remote()  # executed in the "compute" group.
            a.f4.remote()  # executed in the "compute" group.
            a.f5.remote()  # executed in the default group.

    .. tab-item:: Java

        You can define concurrency groups for concurrent actors using the API ``setConcurrencyGroups()`` argument:

        .. code-block:: java

            class ConcurrentActor {
                public long f1() {
                    return Thread.currentThread().getId();
                }

                public long f2() {
                    return Thread.currentThread().getId();
                }

                public long f3(int a, int b) {
                    return Thread.currentThread().getId();
                }

                public long f4() {
                    return Thread.currentThread().getId();
                }

                public long f5() {
                    return Thread.currentThread().getId();
                }
            }

            ConcurrencyGroup group1 =
                new ConcurrencyGroupBuilder<ConcurrentActor>()
                    .setName("io")
                    .setMaxConcurrency(1)
                    .addMethod(ConcurrentActor::f1)
                    .addMethod(ConcurrentActor::f2)
                    .build();
            ConcurrencyGroup group2 =
                new ConcurrencyGroupBuilder<ConcurrentActor>()
                    .setName("compute")
                    .setMaxConcurrency(1)
                    .addMethod(ConcurrentActor::f3)
                    .addMethod(ConcurrentActor::f4)
                    .build();

            ActorHandle<ConcurrentActor> myActor = Ray.actor(ConcurrentActor::new)
                .setConcurrencyGroups(group1, group2)
                .remote();

            myActor.task(ConcurrentActor::f1).remote();  // executed in the "io" group.
            myActor.task(ConcurrentActor::f2).remote();  // executed in the "io" group.
            myActor.task(ConcurrentActor::f3, 3, 5).remote();  // executed in the "compute" group.
            myActor.task(ConcurrentActor::f4).remote();  // executed in the "compute" group.
            myActor.task(ConcurrentActor::f5).remote();  // executed in the "default" group.


.. _default-concurrency-group:

Default Concurrency Group
-------------------------

By default, methods are placed in a default concurrency group which has a concurrency limit of 1000 for AsyncIO actors and 1 otherwise.
The concurrency of the default group can be changed by setting the ``max_concurrency`` actor option.

.. tab-set::

    .. tab-item:: Python

        The following actor has 2 concurrency groups: "io" and "default".
        The max concurrency of "io" is 2, and the max concurrency of "default" is 10.

        .. testcode::

            @ray.remote(concurrency_groups={"io": 2})
            class AsyncIOActor:
                async def f1(self):
                    pass

            actor = AsyncIOActor.options(max_concurrency=10).remote()

    .. tab-item:: Java

        The following concurrent actor has 2 concurrency groups: "io" and "default".
        The max concurrency of "io" is 2, and the max concurrency of "default" is 10.

        .. code-block:: java

            class ConcurrentActor:
                public long f1() {
                    return Thread.currentThread().getId();
                }

            ConcurrencyGroup group =
                new ConcurrencyGroupBuilder<ConcurrentActor>()
                    .setName("io")
                    .setMaxConcurrency(2)
                    .addMethod(ConcurrentActor::f1)
                    .build();

            ActorHandle<ConcurrentActor> myActor = Ray.actor(ConcurrentActor::new)
                  .setConcurrencyGroups(group1)
                  .setMaxConcurrency(10)
                  .remote();


.. _setting-the-concurrency-group-at-runtime:

Setting the Concurrency Group at Runtime
----------------------------------------

You can also dispatch actor methods into a specific concurrency group at runtime.

The following snippet demonstrates setting the concurrency group of the
``f2`` method dynamically at runtime.

.. tab-set::

    .. tab-item:: Python

        You can use the ``.options`` method.

        .. testcode::

            # Executed in the "io" group (as defined in the actor class).
            a.f2.options().remote()

            # Executed in the "compute" group.
            a.f2.options(concurrency_group="compute").remote()

    .. tab-item:: Java

        You can use ``setConcurrencyGroup`` method.

        .. code-block:: java

            // Executed in the "io" group (as defined in the actor creation).
            myActor.task(ConcurrentActor::f2).remote();

            // Executed in the "compute" group.
            myActor.task(ConcurrentActor::f2).setConcurrencyGroup("compute").remote();


.. _actor-task-order:

Actor Task Execution Order
==========================

Synchronous, Single-Threaded Actor
----------------------------------
In Ray, an actor receives tasks from multiple submitters (including driver and workers).
For tasks received from the same submitter, a synchronous, single-threaded actor executes
them following the submission order.
In other words, a given task will not be executed until previously submitted tasks from
the same submitter have finished execution.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import ray

            @ray.remote
            class Counter:
                def __init__(self):
                    self.value = 0

                def add(self, addition):
                    self.value += addition
                    return self.value

            counter = Counter.remote()

            # For tasks from the same submitter,
            # they are executed according to submission order.
            value0 = counter.add.remote(1)
            value1 = counter.add.remote(2)

            # Output: 1. The first submitted task is executed first.
            print(ray.get(value0))
            # Output: 3. The later submitted task is executed later.
            print(ray.get(value1))

        .. testoutput::

            1
            3


However, the actor does not guarantee the execution order of the tasks from different
submitters. For example, suppose an unfulfilled argument blocks a previously submitted
task. In this case, the actor can still execute tasks submitted by a different worker.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import time
            import ray

            @ray.remote
            class Counter:
                def __init__(self):
                    self.value = 0

                def add(self, addition):
                    self.value += addition
                    return self.value

            counter = Counter.remote()

            # Submit task from a worker
            @ray.remote
            def submitter(value):
                return ray.get(counter.add.remote(value))

            # Simulate delayed result resolution.
            @ray.remote
            def delayed_resolution(value):
                time.sleep(5)
                return value

            # Submit tasks from different workers, with
            # the first submitted task waiting for
            # dependency resolution.
            value0 = submitter.remote(delayed_resolution.remote(1))
            value1 = submitter.remote(2)

            # Output: 3. The first submitted task is executed later.
            print(ray.get(value0))
            # Output: 2. The later submitted task is executed first.
            print(ray.get(value1))

        .. testoutput::

            3
            2


Asynchronous or Threaded Actor
------------------------------
:ref:`Asynchronous or threaded actors <async-actors>` do not guarantee the
task execution order. This means the system might execute a task
even though previously submitted tasks are pending execution.

.. tab-set::

    .. tab-item:: Python

        .. testcode::

            import time
            import ray

            @ray.remote
            class AsyncCounter:
                def __init__(self):
                    self.value = 0

                async def add(self, addition):
                    self.value += addition
                    return self.value

            counter = AsyncCounter.remote()

            # Simulate delayed result resolution.
            @ray.remote
            def delayed_resolution(value):
                time.sleep(5)
                return value

            # Submit tasks from the driver, with
            # the first submitted task waiting for
            # dependency resolution.
            value0 = counter.add.remote(delayed_resolution.remote(1))
            value1 = counter.add.remote(2)

            # Output: 3. The first submitted task is executed later.
            print(ray.get(value0))
            # Output: 2. The later submitted task is executed first.
            print(ray.get(value1))

        .. testoutput::

            3
            2


Nested Remote Functions
=======================

Remote functions can call other remote functions, resulting in nested tasks.
For example, consider the following.

.. literalinclude:: ../doc_code/nested-tasks.py
    :language: python
    :start-after: __nested_start__
    :end-before: __nested_end__

Then calling ``g`` and ``h`` produces the following behavior.

.. code:: python

    >>> ray.get(g.remote())
    [ObjectRef(b1457ba0911ae84989aae86f89409e953dd9a80e),
     ObjectRef(7c14a1d13a56d8dc01e800761a66f09201104275),
     ObjectRef(99763728ffc1a2c0766a2000ebabded52514e9a6),
     ObjectRef(9c2f372e1933b04b2936bb6f58161285829b9914)]

    >>> ray.get(h.remote())
    [1, 1, 1, 1]

**One limitation** is that the definition of ``f`` must come before the
definitions of ``g`` and ``h`` because as soon as ``g`` is defined, it
will be pickled and shipped to the workers, and so if ``f`` hasn't been
defined yet, the definition will be incomplete.

Yielding Resources While Blocked
--------------------------------

Ray will release CPU resources when being blocked. This prevents
deadlock cases where the nested tasks are waiting for the CPU
resources held by the parent task.
Consider the following remote function.

.. literalinclude:: ../doc_code/nested-tasks.py
    :language: python
    :start-after: __yield_start__
    :end-before: __yield_end__

When a ``g`` task is executing, it will release its CPU resources when it gets
blocked in the call to ``ray.get``. It will reacquire the CPU resources when
``ray.get`` returns. It will retain its GPU resources throughout the lifetime of
the task because the task will most likely continue to use GPU memory.


.. _dynamic_generators:

.. warning::

    ``num_returns="dynamic"`` :ref:`generator API <dynamic_generators>` is soft deprecated as of Ray 2.8 due to its :ref:`limitation <dynamic-generators-limitation>`. It is hard deprecated as of Ray 2.9.
    Use the :ref:`streaming generator API<generators>` instead.

Dynamic generators
==================

Python generators are functions that behave like iterators, yielding one
value per iteration. Ray supports remote generators for two use cases:

1. To reduce max heap memory usage when returning multiple values from a remote
   function. See the :ref:`design pattern guide <generator-pattern>` for an
   example.
2. When the number of return values is set dynamically by the remote function
   instead of by the caller.

Remote generators can be used in both actor and non-actor tasks.

.. _static-generators:

`num_returns` set by the task caller
------------------------------------

Where possible, the caller should set the remote function's number of return values using ``@ray.remote(num_returns=x)`` or ``foo.options(num_returns=x).remote()``.
Ray will return this many ``ObjectRefs`` to the caller.
The remote task should then return the same number of values, usually as a tuple or list.
Compared to setting the number of return values dynamically, this adds less complexity to user code and less performance overhead, as Ray will know exactly how many ``ObjectRefs`` to return to the caller ahead of time.

Without changing the caller's syntax, we can also use a remote generator function to yield the values iteratively.
The generator should yield the same number of return values specified by the caller, and these will be stored one at a time in Ray's object store.
An error will be raised for generators that yield a different number of values from the one specified by the caller.

For example, we can swap the following code that returns a list of return values:

.. literalinclude:: ../doc_code/pattern_generators.py
    :language: python
    :start-after: __large_values_start__
    :end-before: __large_values_end__

for this code, which uses a generator function:

.. literalinclude:: ../doc_code/pattern_generators.py
    :language: python
    :start-after: __large_values_generator_start__
    :end-before: __large_values_generator_end__

The advantage of doing so is that the generator function does not need to hold all of its return values in memory at once.
It can yield the arrays one at a time to reduce memory pressure.

.. _dynamic-generators:

`num_returns` set by the task executor
--------------------------------------

In some cases, the caller may not know the number of return values to expect from a remote function.
For example, suppose we want to write a task that breaks up its argument into equal-size chunks and returns these.
We may not know the size of the argument until we execute the task, so we don't know the number of return values to expect.

In these cases, we can use a remote generator function that returns a *dynamic* number of values.
To use this feature, set ``num_returns="dynamic"`` in the ``@ray.remote`` decorator or the remote function's ``.options()``.
Then, when invoking the remote function, Ray will return a *single* ``ObjectRef`` that will get populated with an ``DynamicObjectRefGenerator`` when the task completes.
The ``DynamicObjectRefGenerator`` can be used to iterate over a list of ``ObjectRefs`` containing the actual values returned by the task.

.. literalinclude:: ../doc_code/generator.py
    :language: python
    :start-after: __dynamic_generator_start__
    :end-before: __dynamic_generator_end__

We can also pass the ``ObjectRef`` returned by a task with ``num_returns="dynamic"`` to another task. The task will receive the ``DynamicObjectRefGenerator``, which it can use to iterate over the task's return values. Similarly, you can also pass an ``ObjectRefGenerator`` as a task argument.

.. literalinclude:: ../doc_code/generator.py
    :language: python
    :start-after: __dynamic_generator_pass_start__
    :end-before: __dynamic_generator_pass_end__

Exception handling
------------------

If a generator function raises an exception before yielding all its values, the values that it already stored will still be accessible through their ``ObjectRefs``.
The remaining ``ObjectRefs`` will contain the raised exception.
This is true for both static and dynamic ``num_returns``.
If the task was called with ``num_returns="dynamic"``, the exception will be stored as an additional final ``ObjectRef`` in the ``DynamicObjectRefGenerator``.

.. literalinclude:: ../doc_code/generator.py
    :language: python
    :start-after: __generator_errors_start__
    :end-before: __generator_errors_end__

Note that there is currently a known bug where exceptions will not be propagated for generators that yield more values than expected. This can occur in two cases:

1. When ``num_returns`` is set by the caller, but the generator task returns more than this value.
2. When a generator task with ``num_returns="dynamic"`` is :ref:`re-executed <task-retries>`, and the re-executed task yields more values than the original execution. Note that in general, Ray does not guarantee correctness for task re-execution if the task is nondeterministic, and it is recommended to set ``@ray.remote(num_retries=0)`` for such tasks.

.. literalinclude:: ../doc_code/generator.py
    :language: python
    :start-after: __generator_errors_unsupported_start__
    :end-before: __generator_errors_unsupported_end__

.. _dynamic-generators-limitation:

Limitations
-----------

Although a generator function creates ``ObjectRefs`` one at a time, currently Ray will not schedule dependent tasks until the entire task is complete and all values have been created. This is similar to the semantics used by tasks that return multiple values as a list.


.. _core-patterns-limit-pending-tasks:

Pattern: Using ray.wait to limit the number of pending tasks
============================================================

In this pattern, we use :func:`ray.wait() <ray.wait>` to limit the number of pending tasks.

If we continuously submit tasks faster than their process time, we will accumulate tasks in the pending task queue, which can eventually cause OOM.
With ``ray.wait()``, we can apply backpressure and limit the number of pending tasks so that the pending task queue won't grow indefinitely and cause OOM.

.. note::

   If we submit a finite number of tasks, it's unlikely that we will hit the issue mentioned above since each task only uses a small amount of memory for bookkeeping in the queue.
   It's more likely to happen when we have an infinite stream of tasks to run.

.. note::

   This method is meant primarily to limit how many tasks should be in flight at the same time.
   It can also be used to limit how many tasks can run *concurrently*, but it is not recommended, as it can hurt scheduling performance.
   Ray automatically decides task parallelism based on resource availability, so the recommended method for adjusting how many tasks can run concurrently is to :ref:`modify each task's resource requirements <core-patterns-limit-running-tasks>` instead.

Example use case
----------------

You have a worker actor that process tasks at a rate of X tasks per second and you want to submit tasks to it at a rate lower than X to avoid OOM.

For example, Ray Serve uses this pattern to limit the number of pending queries for each worker.

.. figure:: ../images/limit-pending-tasks.svg

    Limit number of pending tasks


Code example
------------

**Without backpressure:**

.. literalinclude:: ../doc_code/limit_pending_tasks.py
    :language: python
    :start-after: __without_backpressure_start__
    :end-before: __without_backpressure_end__

**With backpressure:**

.. literalinclude:: ../doc_code/limit_pending_tasks.py
    :language: python
    :start-after: __with_backpressure_start__
    :end-before: __with_backpressure_end__


Anti-pattern: Returning ray.put() ObjectRefs from a task harms performance and fault tolerance
==============================================================================================

**TLDR:** Avoid calling :func:`ray.put() <ray.put>` on task return values and returning the resulting ObjectRefs.
Instead, return these values directly if possible.

Returning ray.put() ObjectRefs are considered anti-patterns for the following reasons:

- It disallows inlining small return values: Ray has a performance optimization to return small (<= 100KB) values inline directly to the caller, avoiding going through the distributed object store.
  On the other hand, ``ray.put()`` will unconditionally store the value to the object store which makes the optimization for small return values impossible.
- Returning ObjectRefs involves extra distributed reference counting protocol which is slower than returning the values directly.
- It's less :ref:`fault tolerant <fault-tolerance>`: the worker process that calls ``ray.put()`` is the "owner" of the returned ``ObjectRef`` and the return value fate shares with the owner. If the worker process dies, the return value is lost.
  In contrast, the caller process (often the driver) is the owner of the return value if it's returned directly.

Code example
------------

If you want to return a single value regardless if it's small or large, you should return it directly.

.. literalinclude:: ../doc_code/anti_pattern_return_ray_put.py
    :language: python
    :start-after: __return_single_value_start__
    :end-before: __return_single_value_end__

If you want to return multiple values and you know the number of returns before calling the task, you should use the :ref:`num_returns <ray-task-returns>` option.

.. literalinclude:: ../doc_code/anti_pattern_return_ray_put.py
    :language: python
    :start-after: __return_static_multi_values_start__
    :end-before: __return_static_multi_values_end__

If you don't know the number of returns before calling the task, you should use the :ref:`dynamic generator <dynamic-generators>` pattern if possible.

.. literalinclude:: ../doc_code/anti_pattern_return_ray_put.py
    :language: python
    :start-after: __return_dynamic_multi_values_start__
    :end-before: __return_dynamic_multi_values_end__


.. _ray-pass-large-arg-by-value:

Anti-pattern: Passing the same large argument by value repeatedly harms performance
===================================================================================

**TLDR:** Avoid passing the same large argument by value to multiple tasks, use :func:`ray.put() <ray.put>` and pass by reference instead.

When passing a large argument (>100KB) by value to a task,
Ray will implicitly store the argument in the object store and the worker process will fetch the argument to the local object store from the caller's object store before running the task.
If we pass the same large argument to multiple tasks, Ray will end up storing multiple copies of the argument in the object store since Ray doesn't do deduplication.

Instead of passing the large argument by value to multiple tasks,
we should use ``ray.put()`` to store the argument to the object store once and get an ``ObjectRef``,
then pass the argument reference to tasks. This way, we make sure all tasks use the same copy of the argument, which is faster and uses less object store memory.

Code example
------------

**Anti-pattern:**

.. literalinclude:: ../doc_code/anti_pattern_pass_large_arg_by_value.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**Better approach:**

.. literalinclude:: ../doc_code/anti_pattern_pass_large_arg_by_value.py
    :language: python
    :start-after: __better_approach_start__
    :end-before: __better_approach_end__


Pattern: Using pipelining to increase throughput
================================================

If you have multiple work items and each requires several steps to complete,
you can use the `pipelining <https://en.wikipedia.org/wiki/Pipeline_(computing)>`__ technique to improve the cluster utilization and increase the throughput of your system.

.. note::

  Pipelining is an important technique to improve the performance and is heavily used by Ray libraries.
  See :ref:`Ray Data <data>` as an example.

.. figure:: ../images/pipelining.svg

Example use case
----------------

A component of your application needs to do both compute-intensive work and communicate with other processes.
Ideally, you want to overlap computation and communication to saturate the CPU and increase the overall throughput.

Code example
------------

.. literalinclude:: ../doc_code/pattern_pipelining.py

In the example above, a worker actor pulls work off of a queue and then does some computation on it.
Without pipelining, we call :func:`ray.get() <ray.get>` immediately after requesting a work item, so we block while that RPC is in flight, causing idle CPU time.
With pipelining, we instead preemptively request the next work item before processing the current one, so we can use the CPU while the RPC is in flight which increases the CPU utilization.


Anti-pattern: Fetching too many objects at once with ray.get causes failure
===========================================================================

**TLDR:** Avoid calling :func:`ray.get() <ray.get>` on too many objects since this will lead to heap out-of-memory or object store out-of-space. Instead fetch and process one batch at a time.

If you have a large number of tasks that you want to run in parallel, trying to do ``ray.get()`` on all of them at once could lead to failure with heap out-of-memory or object store out-of-space since Ray needs to fetch all the objects to the caller at the same time.
Instead you should get and process the results one batch at a time. Once a batch is processed, Ray will evict objects in that batch to make space for future batches.

.. figure:: ../images/ray-get-too-many-objects.svg

    Fetching too many objects at once with ``ray.get()``

Code example
------------

**Anti-pattern:**

.. literalinclude:: ../doc_code/anti_pattern_ray_get_too_many_objects.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**Better approach:**

.. literalinclude:: ../doc_code/anti_pattern_ray_get_too_many_objects.py
    :language: python
    :start-after: __better_approach_start__
    :end-before: __better_approach_end__

Here besides getting one batch at a time to avoid failure, we are also using ``ray.wait()`` to process results in the finish order instead of the submission order to reduce the runtime. See :doc:`ray-get-submission-order` for more details.


.. _task-pattern-nested-tasks:

Pattern: Using nested tasks to achieve nested parallelism
=========================================================

In this pattern, a remote task can dynamically call other remote tasks (including itself) for nested parallelism.
This is useful when sub-tasks can be parallelized.

Keep in mind, though, that nested tasks come with their own cost: extra worker processes, scheduling overhead, bookkeeping overhead, etc.
To achieve speedup with nested parallelism, make sure each of your nested tasks does significant work. See :doc:`too-fine-grained-tasks` for more details.

Example use case
----------------

You want to quick-sort a large list of numbers.
By using nested tasks, we can sort the list in a distributed and parallel fashion.

.. figure:: ../images/tree-of-tasks.svg

    Tree of tasks


Code example
------------

.. literalinclude:: ../doc_code/pattern_nested_tasks.py
    :language: python
    :start-after: __pattern_start__
    :end-before: __pattern_end__

We call :func:`ray.get() <ray.get>` after both ``quick_sort_distributed`` function invocations take place.
This allows you to maximize parallelism in the workload. See :doc:`ray-get-loop` for more details.

Notice in the execution times above that with smaller tasks, the non-distributed version is faster. However, as the task execution
time increases, i.e. because the lists to sort are larger, the distributed version is faster.


Pattern: Using a supervisor actor to manage a tree of actors
============================================================

Actor supervision is a pattern in which a supervising actor manages a collection of worker actors.
The supervisor delegates tasks to subordinates and handles their failures.
This pattern simplifies the driver since it manages only a few supervisors and does not deal with failures from worker actors directly.
Furthermore, multiple supervisors can act in parallel to parallelize more work.

.. figure:: ../images/tree-of-actors.svg

    Tree of actors

.. note::

    - If the supervisor dies (or the driver), the worker actors are automatically terminated thanks to actor reference counting.
    - Actors can be nested to multiple levels to form a tree.

Example use case
----------------

You want to do data parallel training and train the same model with different hyperparameters in parallel.
For each hyperparameter, you can launch a supervisor actor to do the orchestration and it will create worker actors to do the actual training per data shard.

.. note::
    For data parallel training and hyperparameter tuning, it's recommended to use :ref:`Ray Train <train-key-concepts>` (:py:class:`~ray.train.data_parallel_trainer.DataParallelTrainer` and :ref:`Ray Tune's Tuner <tune-main>`)
    which applies this pattern under the hood.

Code example
------------

.. literalinclude:: ../doc_code/pattern_tree_of_actors.py
    :language: python


Pattern: Using asyncio to run actor methods concurrently
========================================================

By default, a Ray :ref:`actor <ray-remote-classes>` runs in a single thread and
actor method calls are executed sequentially. This means that a long running method call blocks all the following ones.
In this pattern, we use ``await`` to yield control from the long running method call so other method calls can run concurrently.
Normally the control is yielded when the method is doing IO operations but you can also use ``await asyncio.sleep(0)`` to yield control explicitly.

.. note::
   You can also use :ref:`threaded actors <threaded-actors>` to achieve concurrency.

Example use case
----------------

You have an actor with a long polling method that continuously fetches tasks from the remote store and executes them.
You also want to query the number of tasks executed while the long polling method is running.

With the default actor, the code will look like this:

.. literalinclude:: ../doc_code/pattern_async_actor.py
    :language: python
    :start-after: __sync_actor_start__
    :end-before: __sync_actor_end__

This is problematic because ``TaskExecutor.run`` method runs forever and never yield the control to run other methods.
We can solve this problem by using :ref:`async actors <async-actors>` and use ``await`` to yield control:

.. literalinclude:: ../doc_code/pattern_async_actor.py
    :language: python
    :start-after: __async_actor_start__
    :end-before: __async_actor_end__

Here, instead of using the blocking :func:`ray.get() <ray.get>` to get the value of an ObjectRef, we use ``await`` so it can yield the control while we are waiting for the object to be fetched.



Anti-pattern: Processing results in submission order using ray.get increases runtime
====================================================================================

**TLDR:** Avoid processing independent results in submission order using :func:`ray.get() <ray.get>` since results may be ready in a different order than the submission order.

A batch of tasks is submitted, and we need to process their results individually once they’re done.
If each task takes a different amount of time to finish and we process results in submission order, we may waste time waiting for all of the slower (straggler) tasks that were submitted earlier to finish while later faster tasks have already finished.

Instead, we want to process the tasks in the order that they finish using :func:`ray.wait() <ray.wait>` to speed up total time to completion.

.. figure:: ../images/ray-get-submission-order.svg

    Processing results in submission order vs completion order


Code example
------------

.. literalinclude:: ../doc_code/anti_pattern_ray_get_submission_order.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

Other ``ray.get()`` related anti-patterns are:

- :doc:`unnecessary-ray-get`
- :doc:`ray-get-loop`


.. _core-patterns:

Design Patterns & Anti-patterns
===============================

This section is a collection of common design patterns and anti-patterns for writing Ray applications.

.. toctree::
    :maxdepth: 1

    nested-tasks
    generators
    limit-pending-tasks
    limit-running-tasks
    concurrent-operations-async-actor
    actor-sync
    tree-of-actors
    pipelining
    return-ray-put
    ray-get-loop
    unnecessary-ray-get
    ray-get-submission-order
    ray-get-too-many-objects
    too-fine-grained-tasks
    redefine-task-actor-loop
    pass-large-arg-by-value
    closure-capture-large-objects
    global-variables


.. _ray-get-loop:

Anti-pattern: Calling ray.get in a loop harms parallelism
=========================================================

**TLDR:** Avoid calling :func:`ray.get() <ray.get>` in a loop since it's a blocking call; use ``ray.get()`` only for the final result.

A call to ``ray.get()`` fetches the results of remotely executed functions. However, it is a blocking call, which means that it always waits until the requested result is available.
If you call ``ray.get()`` in a loop, the loop will not continue to run until the call to ``ray.get()`` is resolved.

If you also spawn the remote function calls in the same loop, you end up with no parallelism at all, as you wait for the previous function call to finish (because of ``ray.get()``) and only spawn the next call in the next iteration of the loop.
The solution here is to separate the call to ``ray.get()`` from the call to the remote functions. That way all remote functions are spawned before we wait for the results and can run in parallel in the background. Additionally, you can pass a list of object references to ``ray.get()`` instead of calling it one by one to wait for all of the tasks to finish.

Code example
------------

.. literalinclude:: ../doc_code/anti_pattern_ray_get_loop.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

.. figure:: ../images/ray-get-loop.svg

    Calling ``ray.get()`` in a loop

When calling ``ray.get()`` right after scheduling the remote work, the loop blocks until the result is received. We thus end up with sequential processing.
Instead, we should first schedule all remote calls, which are then processed in parallel. After scheduling the work, we can then request all the results at once.

Other ``ray.get()`` related anti-patterns are:

- :doc:`unnecessary-ray-get`
- :doc:`ray-get-submission-order`


Anti-pattern: Using global variables to share state between tasks and actors
============================================================================

**TLDR:** Don't use global variables to share state with tasks and actors. Instead, encapsulate the global variables in an actor and pass the actor handle to other tasks and actors.

Ray drivers, tasks and actors are running in
different processes, so they don’t share the same address space.
This means that if you modify global variables
in one process, changes are not reflected in other processes.

The solution is to use an actor's instance variables to hold the global state and pass the actor handle to places where the state needs to be modified or accessed.
Note that using class variables to manage state between instances of the same class is not supported.
Each actor instance is instantiated in its own process, so each actor will have its own copy of the class variables.

Code example
------------

**Anti-pattern:**

.. literalinclude:: ../doc_code/anti_pattern_global_variables.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**Better approach:**

.. literalinclude:: ../doc_code/anti_pattern_global_variables.py
    :language: python
    :start-after: __better_approach_start__
    :end-before: __better_approach_end__


Anti-pattern: Redefining the same remote function or class harms performance
============================================================================

**TLDR:** Avoid redefining the same remote function or class.

Decorating the same function or class multiple times using the :func:`ray.remote <ray.remote>` decorator leads to slow performance in Ray.
For each Ray remote function or class, Ray will pickle it and upload to GCS.
Later on, the worker that runs the task or actor will download and unpickle it.
Each decoration of the same function or class generates a new remote function or class from Ray's perspective.
As a result, the pickle, upload, download and unpickle work will happen every time we redefine and run the remote function or class.

Code example
------------

**Anti-pattern:**

.. literalinclude:: ../doc_code/anti_pattern_redefine_task_actor_loop.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**Better approach:**

.. literalinclude:: ../doc_code/anti_pattern_redefine_task_actor_loop.py
    :language: python
    :start-after: __better_approach_start__
    :end-before: __better_approach_end__

We should define the same remote function or class outside of the loop instead of multiple times inside a loop so that it's pickled and uploaded only once.


.. _core-patterns-limit-running-tasks:

Pattern: Using resources to limit the number of concurrently running tasks
==========================================================================

In this pattern, we use :ref:`resources <resource-requirements>` to limit the number of concurrently running tasks.

By default, Ray tasks require 1 CPU each and Ray actors require 0 CPU each, so the scheduler limits task concurrency to the available CPUs and actor concurrency to infinite.
Tasks that use more than 1 CPU (e.g., via mutlithreading) may experience slowdown due to interference from concurrent ones, but otherwise are safe to run.

However, tasks or actors that use more than their proportionate share of memory may overload a node and cause issues like OOM.
If that is the case, we can reduce the number of concurrently running tasks or actors on each node by increasing the amount of resources requested by them.
This works because Ray makes sure that the sum of the resource requirements of all of the concurrently running tasks and actors on a given node does not exceed the node's total resources.

.. note::

   For actor tasks, the number of running actors limits the number of concurrently running actor tasks we can have.

Example use case
----------------

You have a data processing workload that processes each input file independently using Ray :ref:`remote functions <ray-remote-functions>`.
Since each task needs to load the input data into heap memory and do the processing, running too many of them can cause OOM.
In this case, you can use the ``memory`` resource to limit the number of concurrently running tasks (usage of other resources like ``num_cpus`` can achieve the same goal as well).
Note that similar to ``num_cpus``, the ``memory`` resource requirement is *logical*, meaning that Ray will not enforce the physical memory usage of each task if it exceeds this amount.

Code example
------------

**Without limit:**

.. literalinclude:: ../doc_code/limit_running_tasks.py
    :language: python
    :start-after: __without_limit_start__
    :end-before: __without_limit_end__

**With limit:**

.. literalinclude:: ../doc_code/limit_running_tasks.py
    :language: python
    :start-after: __with_limit_start__
    :end-before: __with_limit_end__


.. _unnecessary-ray-get:

Anti-pattern: Calling ray.get unnecessarily harms performance
=============================================================

**TLDR:** Avoid calling :func:`ray.get() <ray.get>` unnecessarily for intermediate steps. Work with object references directly, and only call ``ray.get()`` at the end to get the final result.

When ``ray.get()`` is called, objects must be transferred to the worker/node that calls ``ray.get()``. If you don't need to manipulate the object, you probably don't need to call ``ray.get()`` on it!

Typically, it’s best practice to wait as long as possible before calling ``ray.get()``, or even design your program to avoid having to call ``ray.get()`` at all.

Code example
------------

**Anti-pattern:**

.. literalinclude:: ../doc_code/anti_pattern_unnecessary_ray_get.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

.. figure:: ../images/unnecessary-ray-get-anti.svg

**Better approach:**

.. literalinclude:: ../doc_code/anti_pattern_unnecessary_ray_get.py
    :language: python
    :start-after: __better_approach_start__
    :end-before: __better_approach_end__

.. figure:: ../images/unnecessary-ray-get-better.svg

Notice in the anti-pattern example, we call ``ray.get()`` which forces us to transfer the large rollout to the driver, then again to the *reduce* worker.

In the fixed version, we only pass the reference to the object to the *reduce* task.
The ``reduce`` worker will implicitly call ``ray.get()`` to fetch the actual rollout data directly from the ``generate_rollout`` worker, avoiding the extra copy to the driver.

Other ``ray.get()`` related anti-patterns are:

- :doc:`ray-get-loop`
- :doc:`ray-get-submission-order`


Pattern: Using an actor to synchronize other tasks and actors
=============================================================

When you have multiple tasks that need to wait on some condition or otherwise
need to synchronize across tasks & actors on a cluster, you can use a central
actor to coordinate among them.

Example use case
----------------

You can use an actor to implement a distributed ``asyncio.Event`` that multiple tasks can wait on.

Code example
------------

.. literalinclude:: ../doc_code/actor-sync.py


Anti-pattern: Closure capturing large objects harms performance
===============================================================

**TLDR:** Avoid closure capturing large objects in remote functions or classes, use object store instead.

When you define a :func:`ray.remote <ray.remote>` function or class,
it is easy to accidentally capture large (more than a few MB) objects implicitly in the definition.
This can lead to slow performance or even OOM since Ray is not designed to handle serialized functions or classes that are very large.

For such large objects, there are two options to resolve this problem:

- Use :func:`ray.put() <ray.put>` to put the large objects in the Ray object store, and then pass object references as arguments to the remote functions or classes (*"better approach #1"* below)
- Create the large objects inside the remote functions or classes by passing a lambda method (*"better approach #2"*). This is also the only option for using unserializable objects.


Code example
------------

**Anti-pattern:**

.. literalinclude:: ../doc_code/anti_pattern_closure_capture_large_objects.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**Better approach #1:**

.. literalinclude:: ../doc_code/anti_pattern_closure_capture_large_objects.py
    :language: python
    :start-after: __better_approach_1_start__
    :end-before: __better_approach_1_end__

**Better approach #2:**

.. literalinclude:: ../doc_code/anti_pattern_closure_capture_large_objects.py
    :language: python
    :start-after: __better_approach_2_start__
    :end-before: __better_approach_2_end__


Anti-pattern: Over-parallelizing with too fine-grained tasks harms speedup
==========================================================================

**TLDR:** Avoid over-parallelizing. Parallelizing tasks has higher overhead than using normal functions.

Parallelizing or distributing tasks usually comes with higher overhead than an ordinary function call. Therefore, if you parallelize a function that executes very quickly, the overhead could take longer than the actual function call!

To handle this problem, we should be careful about parallelizing too much. If you have a function or task that’s too small, you can use a technique called **batching** to make your tasks do more meaningful work in a single call.


Code example
------------

**Anti-pattern:**

.. literalinclude:: ../doc_code/anti_pattern_too_fine_grained_tasks.py
    :language: python
    :start-after: __anti_pattern_start__
    :end-before: __anti_pattern_end__

**Better approach:** Use batching.

.. literalinclude:: ../doc_code/anti_pattern_too_fine_grained_tasks.py
    :language: python
    :start-after: __batching_start__
    :end-before: __batching_end__

As we can see from the example above, over-parallelizing has higher overhead and the program runs slower than the serial version.
Through batching with a proper batch size, we are able to amortize the overhead and achieve the expected speedup.


.. _generator-pattern:

Pattern: Using generators to reduce heap memory usage
=====================================================

In this pattern, we use **generators** in Python to reduce the total heap memory usage during a task. The key idea is that for tasks that return multiple objects, we can return them one at a time instead of all at once. This allows a worker to free the heap memory used by a previous return value before returning the next one.

Example use case
----------------

You have a task that returns multiple large values. Another possibility is a task that returns a single large value, but you want to stream this value through Ray's object store by breaking it up into smaller chunks.

Using normal Python functions, we can write such a task like this. Here's an example that returns numpy arrays of size 100MB each:

.. literalinclude:: ../doc_code/pattern_generators.py
    :language: python
    :start-after: __large_values_start__
    :end-before: __large_values_end__

However, this will require the task to hold all ``num_returns`` arrays in heap memory at the same time at the end of the task. If there are many return values, this can lead to high heap memory usage and potentially an out-of-memory error.

We can fix the above example by rewriting ``large_values`` as a **generator**. Instead of returning all values at once as a tuple or list, we can ``yield`` one value at a time.

.. literalinclude:: ../doc_code/pattern_generators.py
    :language: python
    :start-after: __large_values_generator_start__
    :end-before: __large_values_generator_end__

Code example
------------

.. literalinclude:: ../doc_code/pattern_generators.py
    :language: python
    :start-after: __program_start__

.. code-block:: text

    $ RAY_IGNORE_UNHANDLED_ERRORS=1 python test.py 100

    Using normal functions...
    ... -- A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker...
    Worker failed
    Using generators...
    (large_values_generator pid=373609) yielded return value 0
    (large_values_generator pid=373609) yielded return value 1
    (large_values_generator pid=373609) yielded return value 2
    ...
    Success!


Ray Core CLI
============

.. _ray-cli:

Debugging applications
----------------------
This section contains commands for inspecting and debugging the current cluster.

.. _ray-stack-doc:

.. click:: ray.scripts.scripts:stack
   :prog: ray stack
   :show-nested:

.. _ray-memory-doc:

.. click:: ray.scripts.scripts:memory
   :prog: ray memory
   :show-nested:

.. _ray-timeline-doc:

.. click:: ray.scripts.scripts:timeline
   :prog: ray timeline
   :show-nested:

.. _ray-status-doc:

.. click:: ray.scripts.scripts:status
   :prog: ray status
   :show-nested:

.. click:: ray.scripts.scripts:debug
   :prog: ray debug
   :show-nested:


Usage Stats
-----------
This section contains commands to enable/disable :ref:`Ray usage stats <ref-usage-stats>`.

.. _ray-disable-usage-stats-doc:

.. click:: ray.scripts.scripts:disable_usage_stats
   :prog: ray disable-usage-stats
   :show-nested:

.. _ray-enable-usage-stats-doc:

.. click:: ray.scripts.scripts:enable_usage_stats
   :prog: ray enable-usage-stats
   :show-nested:

Core API
========

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ray.init
    ray.shutdown
    ray.is_initialized
    ray.job_config.JobConfig

Tasks
-----

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ray.remote
    ray.remote_function.RemoteFunction.options
    ray.cancel

Actors
------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ray.remote
    ray.actor.ActorClass.options
    ray.method
    ray.get_actor
    ray.kill

Objects
-------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ray.get
    ray.wait
    ray.put

.. _runtime-context-apis:

Runtime Context
---------------
.. autosummary::
    :nosignatures:
    :toctree: doc/

    ray.runtime_context.get_runtime_context
    ray.runtime_context.RuntimeContext
    ray.get_gpu_ids

Cross Language
--------------
.. autosummary::
    :nosignatures:
    :toctree: doc/

    ray.cross_language.java_function
    ray.cross_language.java_actor_class


Ray Core API
============

.. toctree::
    :maxdepth: 2

    core.rst
    scheduling.rst
    runtime-env.rst
    utility.rst
    exceptions.rst
    cli.rst
    ../../ray-observability/reference/cli.rst
    ../../ray-observability/reference/api.rst


Runtime Env API
===============

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ray.runtime_env.RuntimeEnvConfig
    ray.runtime_env.RuntimeEnv


Scheduling API
==============

Scheduling Strategy
-------------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy
    ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy

.. _ray-placement-group-ref:

Placement Group
---------------

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ray.util.placement_group
    ray.util.placement_group.PlacementGroup
    ray.util.placement_group_table
    ray.util.remove_placement_group
    ray.util.get_current_placement_group


Utility
=======

.. autosummary::
   :nosignatures:
   :toctree: doc/

   ray.util.ActorPool
   ray.util.queue.Queue
   ray.nodes
   ray.cluster_resources
   ray.available_resources

.. _custom-metric-api-ref:

Custom Metrics
--------------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   ray.util.metrics.Counter
   ray.util.metrics.Gauge
   ray.util.metrics.Histogram

.. _package-ref-debugging-apis:

Debugging
---------

.. autosummary::
   :nosignatures:
   :toctree: doc/

   ray.util.rpdb.set_trace
   ray.util.inspect_serializability
   ray.timeline


.. _ray-core-exceptions:

Exceptions
==========

.. autosummary::
    :nosignatures:
    :toctree: doc/

    ray.exceptions.RayError
    ray.exceptions.RayTaskError
    ray.exceptions.RayActorError
    ray.exceptions.TaskCancelledError
    ray.exceptions.TaskUnschedulableError
    ray.exceptions.ActorDiedError
    ray.exceptions.ActorUnschedulableError
    ray.exceptions.ActorUnavailableError
    ray.exceptions.AsyncioActorExit
    ray.exceptions.LocalRayletDiedError
    ray.exceptions.WorkerCrashedError
    ray.exceptions.TaskPlacementGroupRemoved
    ray.exceptions.ActorPlacementGroupRemoved
    ray.exceptions.ObjectStoreFullError
    ray.exceptions.OutOfDiskError
    ray.exceptions.ObjectLostError
    ray.exceptions.ObjectFetchTimedOutError
    ray.exceptions.GetTimeoutError
    ray.exceptions.OwnerDiedError
    ray.exceptions.PlasmaObjectNotAvailable
    ray.exceptions.ObjectReconstructionFailedError
    ray.exceptions.ObjectReconstructionFailedMaxAttemptsExceededError
    ray.exceptions.ObjectReconstructionFailedLineageEvictedError
    ray.exceptions.RuntimeEnvSetupError
    ray.exceptions.CrossLanguageError
    ray.exceptions.RaySystemError


.. _fault-tolerance-nodes:

Node Fault Tolerance
====================

A Ray cluster consists of one or more worker nodes,
each of which consists of worker processes and system processes (e.g. raylet).
One of the worker nodes is designated as the head node and has extra processes like the GCS.

Here, we describe node failures and their impact on tasks, actors, and objects.

Worker node failure
-------------------

When a worker node fails, all the running tasks and actors will fail and all the objects owned by worker processes of this node will be lost. In this case, the :ref:`tasks <fault-tolerance-tasks>`, :ref:`actors <fault-tolerance-actors>`, :ref:`objects <fault-tolerance-objects>` fault tolerance mechanisms will kick in and try to recover the failures using other worker nodes.

Head node failure
-----------------

When a head node fails, the entire Ray cluster fails.
To tolerate head node failures, we need to make :ref:`GCS fault tolerant <fault-tolerance-gcs>`
so that when we start a new head node we still have all the cluster-level data.

Raylet failure
--------------

When a raylet process fails, the corresponding node will be marked as dead and is treated the same as node failure.
Each raylet is associated with a unique id, so even if the raylet restarts on the same physical machine,
it'll be treated as a new raylet/node to the Ray cluster.


.. _fault-tolerance-actors:
.. _actor-fault-tolerance:

Actor Fault Tolerance
=====================

Actors can fail if the actor process dies, or if the **owner** of the actor
dies. The owner of an actor is the worker that originally created the actor by
calling ``ActorClass.remote()``. :ref:`Detached actors <actor-lifetimes>` do
not have an owner process and are cleaned up when the Ray cluster is destroyed.


Actor process failure
---------------------

Ray can automatically restart actors that crash unexpectedly.
This behavior is controlled using ``max_restarts``,
which sets the maximum number of times that an actor will be restarted.
The default value of ``max_restarts`` is 0, meaning that the actor won't be
restarted. If set to -1, the actor will be restarted infinitely many times.
When an actor is restarted, its state will be recreated by rerunning its
constructor.
After the specified number of restarts, subsequent actor methods will
raise a ``RayActorError``.

By default, actor tasks execute with at-most-once semantics
(``max_task_retries=0`` in the ``@ray.remote`` :func:`decorator <ray.remote>`). This means that if an
actor task is submitted to an actor that is unreachable, Ray will report the
error with ``RayActorError``, a Python-level exception that is thrown when
``ray.get`` is called on the future returned by the task. Note that this
exception may be thrown even though the task did indeed execute successfully.
For example, this can happen if the actor dies immediately after executing the
task.

Ray also offers at-least-once execution semantics for actor tasks
(``max_task_retries=-1`` or ``max_task_retries > 0``). This means that if an
actor task is submitted to an actor that is unreachable, the system will
automatically retry the task. With this option, the system will only throw a
``RayActorError`` to the application if one of the following occurs: (1) the
actor’s ``max_restarts`` limit has been exceeded and the actor cannot be
restarted anymore, or (2) the ``max_task_retries`` limit has been exceeded for
this particular task. Note that if the actor is currently restarting when a
task is submitted, this will count for one retry. The retry limit can be set to
infinity with ``max_task_retries = -1``.

You can experiment with this behavior by running the following code.

.. literalinclude:: ../doc_code/actor_restart.py
  :language: python
  :start-after: __actor_restart_begin__
  :end-before: __actor_restart_end__

For at-least-once actors, the system will still guarantee execution ordering
according to the initial submission order. For example, any tasks submitted
after a failed actor task will not execute on the actor until the failed actor
task has been successfully retried. The system will not attempt to re-execute
any tasks that executed successfully before the failure
(unless ``max_task_retries`` is nonzero and the task is needed for :ref:`object
reconstruction <fault-tolerance-objects-reconstruction>`).

.. note::

  For :ref:`async or threaded actors <async-actors>`, :ref:`tasks might be
  executed out of order <actor-task-order>`. Upon actor restart, the system
  will only retry *incomplete* tasks. Previously completed tasks will not be
  re-executed.


At-least-once execution is best suited for read-only actors or actors with
ephemeral state that does not need to be rebuilt after a failure. For actors
that have critical state, the application is responsible for recovering the
state, e.g., by taking periodic checkpoints and recovering from the checkpoint
upon actor restart.


Actor checkpointing
~~~~~~~~~~~~~~~~~~~

``max_restarts`` automatically restarts the crashed actor,
but it doesn't automatically restore application level state in your actor.
Instead, you should manually checkpoint your actor's state and recover upon actor restart.

For actors that are restarted manually, the actor's creator should manage the checkpoint and manually restart and recover the actor upon failure. This is recommended if you want the creator to decide when the actor should be restarted and/or if the creator is coordinating actor checkpoints with other execution:

.. literalinclude:: ../doc_code/actor_checkpointing.py
  :language: python
  :start-after: __actor_checkpointing_manual_restart_begin__
  :end-before: __actor_checkpointing_manual_restart_end__

Alternatively, if you are using Ray's automatic actor restart, the actor can checkpoint itself manually and restore from a checkpoint in the constructor:

.. literalinclude:: ../doc_code/actor_checkpointing.py
  :language: python
  :start-after: __actor_checkpointing_auto_restart_begin__
  :end-before: __actor_checkpointing_auto_restart_end__

.. note::

  If the checkpoint is saved to external storage, make sure
  it's accessible to the entire cluster since the actor can be restarted
  on a different node.
  For example, save the checkpoint to cloud storage (e.g., S3) or a shared directory (e.g., via NFS).


Actor creator failure
---------------------

For :ref:`non-detached actors <actor-lifetimes>`, the owner of an actor is the
worker that created it, i.e. the worker that called ``ActorClass.remote()``. Similar to
:ref:`objects <fault-tolerance-objects>`, if the owner of an actor dies, then
the actor will also fate-share with the owner.  Ray will not automatically
recover an actor whose owner is dead, even if it has a nonzero
``max_restarts``.

Since :ref:`detached actors <actor-lifetimes>` do not have an owner, they will still be restarted by Ray
even if their original creator dies. Detached actors will continue to be
automatically restarted until the maximum restarts is exceeded, the actor is
destroyed, or until the Ray cluster is destroyed.

You can try out this behavior in the following code.

.. literalinclude:: ../doc_code/actor_creator_failure.py
  :language: python
  :start-after: __actor_creator_failure_begin__
  :end-before: __actor_creator_failure_end__

Force-killing a misbehaving actor
---------------------------------

Sometimes application-level code can cause an actor to hang or leak resources.
In these cases, Ray allows you to recover from the failure by :ref:`manually
terminating <ray-kill-actors>` the actor. You can do this by calling
``ray.kill`` on any handle to the actor. Note that it does not need to be the
original handle to the actor.

If ``max_restarts`` is set, you can also allow Ray to automatically restart the actor by passing ``no_restart=False`` to ``ray.kill``.

Unavailable actors
----------------------

When an actor can't accept method calls, a ``ray.get`` on the method's returned object reference may raise
``ActorUnavailableError``. This exception indicates the actor isn't accessible at the
moment, but may recover after waiting and retrying. Typical cases include:

- The actor is restarting. For example, it's waiting for resources or running the class constructor during the restart.
- The actor is experiencing transient network issues, like connection outages.
- The actor is dead, but the death hasn't yet been reported to the system.

Actor method calls are executed at-most-once. When a ``ray.get()`` call raises the ``ActorUnavailableError`` exception, there's no guarantee on
whether the actor executed the task or not. If the method has side effects, they may or may not
be observable. Ray does guarantee that the method won't be executed twice, unless the actor or the method is configured with retries, as described in the next section.

The actor may or may not recover in the next calls. Those subsequent calls
may raise ``ActorDiedError`` if the actor is confirmed dead, ``ActorUnavailableError`` if it's
still unreachable, or return values normally if the actor recovered.

As a best practice, if the caller gets the ``ActorUnavailableError`` error, it should
"quarantine" the actor and stop sending traffic to the actor. It can then periodically ping
the actor until it raises ``ActorDiedError`` or returns OK.

If a task has ``max_task_retries > 0`` and it received ``ActorUnavailableError``, Ray will retry the task up to ``max_task_retries`` times. If the actor is restarting in its constructor, the task retry will fail, consuming one retry count. If there are still retries remaining, Ray will retry again after ``RAY_task_retry_delay_ms``, until all retries are consumed or the actor is ready to accept tasks. If the constructor takes a long time to run, consider increasing ``max_task_retries`` or increase ``RAY_task_retry_delay_ms``.

Actor method exceptions
-----------------------

Sometime you want to retry when an actor method raises exceptions. Use ``max_task_retries`` with ``retry_exceptions`` to retry.

Note that by default, retrying on user raised exceptions is disabled. To enable it, make sure the method is **idempotent**, that is, invoking it multiple times should be equivalent to invoking it only once.

You can set ``retry_exceptions`` in the `@ray.method(retry_exceptions=...)` decorator, or in the `.options(retry_exceptions=...)` in the method call.

Retry behavior depends on the value you set ``retry_exceptions`` to:
- ``retry_exceptions == False`` (default): No retries for user exceptions.
- ``retry_exceptions == True``: Ray retries a method on user exception up to ``max_task_retries`` times.
- ``retry_exceptions`` is a list of exceptions: Ray retries a method on user exception up to ``max_task_retries`` times, only if the method raises an exception from these specific classes.

``max_task_retries`` applies to both exceptions and actor crashes. A Ray actor can set this option to apply to all of its methods. A method can also set an overriding option for itself. Ray searches for the first non-default value of ``max_task_retries`` in this order:

- The method call's value, for example, `actor.method.options(max_task_retries=2)`. Ray ignores this value if you don't set it.
- The method definition's value, for example, `@ray.method(max_task_retries=2)`. Ray ignores this value if you don't set it.
- The actor creation call's value, for example, `Actor.options(max_task_retries=2)`. Ray ignores this value if you didn't set it.
- The Actor class definition's value, for example, `@ray.remote(max_task_retries=2)` decorator. Ray ignores this value if you didn't set it.
- The default value,`0`.

For example, if a method sets `max_task_retries=5` and `retry_exceptions=True`, and the actor sets `max_restarts=2`, Ray executes the method up to 6 times: once for the initial invocation, and 5 additional retries. The 6 invocations may include 2 actor crashes. After the 6th invocation, a `ray.get` call to the result Ray ObjectRef raises the exception raised in the last invocation, or `ray.exceptions.RayActorError` if the actor crashed in the last invocation.



.. _fault-tolerance-tasks:
.. _task-fault-tolerance:

Task Fault Tolerance
====================

Tasks can fail due to application-level errors, e.g., Python-level exceptions,
or system-level failures, e.g., a machine fails. Here, we describe the
mechanisms that an application developer can use to recover from these errors.

Catching application-level failures
-----------------------------------

Ray surfaces application-level failures as Python-level exceptions. When a task
on a remote worker or actor fails due to a Python-level exception, Ray wraps
the original exception in a ``RayTaskError`` and stores this as the task's
return value. This wrapped exception will be thrown to any worker that tries
to get the result, either by calling ``ray.get`` or if the worker is executing
another task that depends on the object. If the user's exception type can be subclassed,
the raised exception is an instance of both ``RayTaskError`` and the user's exception type
so the user can try-catch either of them. Otherwise, the wrapped exception is just
``RayTaskError`` and the actual user's exception type can be accessed via the ``cause``
field of the ``RayTaskError``.

.. literalinclude:: ../doc_code/task_exceptions.py
  :language: python
  :start-after: __task_exceptions_begin__
  :end-before: __task_exceptions_end__

Example code of catching the user exception type when the exception type can be subclassed:

.. literalinclude:: ../doc_code/task_exceptions.py
  :language: python
  :start-after: __catch_user_exceptions_begin__
  :end-before: __catch_user_exceptions_end__

Example code of accessing the user exception type when the exception type can *not* be subclassed:

.. literalinclude:: ../doc_code/task_exceptions.py
  :language: python
  :start-after: __catch_user_final_exceptions_begin__
  :end-before: __catch_user_final_exceptions_end__

If Ray can't serialize the user's exception, it converts the exception to a ``RayError``.

.. literalinclude:: ../doc_code/task_exceptions.py
  :language: python
  :start-after: __unserializable_exceptions_begin__
  :end-before: __unserializable_exceptions_end__

Use `ray list tasks` from :ref:`State API CLI <state-api-overview-ref>` to query task exit details:

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list tasks 

.. code-block:: bash

  ======== List: 2023-05-26 10:32:00.962610 ========
  Stats:
  ------------------------------
  Total: 3

  Table:
  ------------------------------
      TASK_ID                                             ATTEMPT_NUMBER  NAME    STATE      JOB_ID  ACTOR_ID    TYPE         FUNC_OR_CLASS_NAME    PARENT_TASK_ID                                    NODE_ID                                                   WORKER_ID                                                 ERROR_TYPE
   0  16310a0f0a45af5cffffffffffffffffffffffff01000000                 0  f       FAILED   01000000              NORMAL_TASK  f                     ffffffffffffffffffffffffffffffffffffffff01000000  767bd47b72efb83f33dda1b661621cce9b969b4ef00788140ecca8ad  b39e3c523629ab6976556bd46be5dbfbf319f0fce79a664122eb39a9  TASK_EXECUTION_EXCEPTION
   1  c2668a65bda616c1ffffffffffffffffffffffff01000000                 0  g       FAILED   01000000              NORMAL_TASK  g                     ffffffffffffffffffffffffffffffffffffffff01000000  767bd47b72efb83f33dda1b661621cce9b969b4ef00788140ecca8ad  b39e3c523629ab6976556bd46be5dbfbf319f0fce79a664122eb39a9  TASK_EXECUTION_EXCEPTION
   2  c8ef45ccd0112571ffffffffffffffffffffffff01000000                 0  f       FAILED   01000000              NORMAL_TASK  f                     ffffffffffffffffffffffffffffffffffffffff01000000  767bd47b72efb83f33dda1b661621cce9b969b4ef00788140ecca8ad  b39e3c523629ab6976556bd46be5dbfbf319f0fce79a664122eb39a9  TASK_EXECUTION_EXCEPTION

.. _task-retries:

Retrying failed tasks
---------------------

When a worker is executing a task, if the worker dies unexpectedly, either
because the process crashed or because the machine failed, Ray will rerun
the task until either the task succeeds or the maximum number of retries is
exceeded. The default number of retries is 3 and can be overridden by
specifying ``max_retries`` in the ``@ray.remote`` decorator. Specifying -1
allows infinite retries, and 0 disables retries. To override the default number
of retries for all tasks submitted, set the OS environment variable
``RAY_TASK_MAX_RETRIES``. e.g., by passing this to your driver script or by
using :ref:`runtime environments<runtime-environments>`.

You can experiment with this behavior by running the following code.

.. literalinclude:: ../doc_code/tasks_fault_tolerance.py
  :language: python
  :start-after: __tasks_fault_tolerance_retries_begin__
  :end-before: __tasks_fault_tolerance_retries_end__

When a task returns a result in the Ray object store, it is possible for the
resulting object to be lost **after** the original task has already finished.
In these cases, Ray will also try to automatically recover the object by
re-executing the tasks that created the object. This can be configured through
the same ``max_retries`` option described here. See :ref:`object fault
tolerance <fault-tolerance-objects>` for more information.

By default, Ray will **not** retry tasks upon exceptions thrown by application
code. However, you may control whether application-level errors are retried,
and even **which** application-level errors are retried, via the
``retry_exceptions`` argument. This is ``False`` by default. To enable retries
upon application-level errors, set ``retry_exceptions=True`` to retry upon any
exception, or pass a list of retryable exceptions. An example is shown below.

.. literalinclude:: ../doc_code/tasks_fault_tolerance.py
  :language: python
  :start-after: __tasks_fault_tolerance_retries_exception_begin__
  :end-before: __tasks_fault_tolerance_retries_exception_end__


Use `ray list tasks -f task_id=\<task_id\>` from :ref:`State API CLI <state-api-overview-ref>` to see task attempts failures and retries:

.. code-block:: bash

  # This API is only available when you download Ray via `pip install "ray[default]"`
  ray list tasks -f task_id=16310a0f0a45af5cffffffffffffffffffffffff01000000

.. code-block:: bash

  ======== List: 2023-05-26 10:38:08.809127 ========
  Stats:
  ------------------------------
  Total: 2

  Table:
  ------------------------------
      TASK_ID                                             ATTEMPT_NUMBER  NAME              STATE       JOB_ID  ACTOR_ID    TYPE         FUNC_OR_CLASS_NAME    PARENT_TASK_ID                                    NODE_ID                                                   WORKER_ID                                                 ERROR_TYPE
   0  16310a0f0a45af5cffffffffffffffffffffffff01000000                 0  potentially_fail  FAILED    01000000              NORMAL_TASK  potentially_fail      ffffffffffffffffffffffffffffffffffffffff01000000  94909e0958e38d10d668aa84ed4143d0bf2c23139ae1a8b8d6ef8d9d  b36d22dbf47235872ad460526deaf35c178c7df06cee5aa9299a9255  WORKER_DIED
   1  16310a0f0a45af5cffffffffffffffffffffffff01000000                 1  potentially_fail  FINISHED  01000000              NORMAL_TASK  potentially_fail      ffffffffffffffffffffffffffffffffffffffff01000000  94909e0958e38d10d668aa84ed4143d0bf2c23139ae1a8b8d6ef8d9d  22df7f2a9c68f3db27498f2f435cc18582de991fbcaf49ce0094ddb0


Cancelling misbehaving tasks
----------------------------

If a task is hanging, you may want to cancel the task to continue to make
progress. You can do this by calling ``ray.cancel`` on an ``ObjectRef``
returned by the task. By default, this will send a KeyboardInterrupt to the
task's worker if it is mid-execution.  Passing ``force=True`` to ``ray.cancel``
will force-exit the worker. See :func:`the API reference <ray.cancel>` for
``ray.cancel`` for more details.

Note that currently, Ray will not automatically retry tasks that have been
cancelled.

Sometimes, application-level code may cause memory leaks on a worker after
repeated task executions, e.g., due to bugs in third-party libraries.
To make progress in these cases, you can set the ``max_calls`` option in a
task's ``@ray.remote`` decorator. Once a worker has executed this many
invocations of the given remote function, it will automatically exit. By
default, ``max_calls`` is set to infinity.


.. _fault-tolerance-objects:
.. _object-fault-tolerance:

Object Fault Tolerance
======================

A Ray object has both data (the value returned when calling ``ray.get``) and
metadata (e.g., the location of the value). Data is stored in the Ray object
store while the metadata is stored at the object's **owner**. The owner of an
object is the worker process that creates the original ``ObjectRef``, e.g., by
calling ``f.remote()`` or ``ray.put()``. Note that this worker is usually a
distinct process from the worker that creates the **value** of the object,
except in cases of ``ray.put``.

.. literalinclude:: ../doc_code/owners.py
  :language: python
  :start-after: __owners_begin__
  :end-before: __owners_end__


Ray can automatically recover from data loss but not owner failure.

.. _fault-tolerance-objects-reconstruction:

Recovering from data loss
-------------------------

When an object value is lost from the object store, such as during node
failures, Ray will use *lineage reconstruction* to recover the object.
Ray will first automatically attempt to recover the value by looking
for copies of the same object on other nodes. If none are found, then Ray will
automatically recover the value by :ref:`re-executing <fault-tolerance-tasks>`
the task that previously created the value.  Arguments to the task are
recursively reconstructed through the same mechanism.

Lineage reconstruction currently has the following limitations:

* The object, and any of its transitive dependencies, must have been generated
  by a task (actor or non-actor). This means that **objects created by
  ray.put are not recoverable**.
* Tasks are assumed to be deterministic and idempotent. Thus,
  **by default, objects created by actor tasks are not reconstructable**. To allow
  reconstruction of actor task results, set the ``max_task_retries`` parameter
  to a non-zero value (see :ref:`actor
  fault tolerance <fault-tolerance-actors>` for more details).
* Tasks will only be re-executed up to their maximum number of retries. By
  default, a non-actor task can be retried up to 3 times and an actor task
  cannot be retried.  This can be overridden with the ``max_retries`` parameter
  for :ref:`remote functions <fault-tolerance-tasks>` and the
  ``max_task_retries`` parameter for :ref:`actors <fault-tolerance-actors>`.
* The owner of the object must still be alive (see :ref:`below
  <fault-tolerance-ownership>`).

Lineage reconstruction can cause higher than usual driver memory
usage because the driver keeps the descriptions of any tasks that may be
re-executed in case of failure. To limit the amount of memory used by
lineage, set the environment variable ``RAY_max_lineage_bytes`` (default 1GB)
to evict lineage if the threshold is exceeded.

To disable lineage reconstruction entirely, set the environment variable
``RAY_TASK_MAX_RETRIES=0`` during ``ray start`` or ``ray.init``.  With this
setting, if there are no copies of an object left, an ``ObjectLostError`` will
be raised.

.. _fault-tolerance-ownership:

Recovering from owner failure
-----------------------------

The owner of an object can die because of node or worker process failure.
Currently, **Ray does not support recovery from owner failure**. In this case, Ray
will clean up any remaining copies of the object's value to prevent a memory
leak. Any workers that subsequently try to get the object's value will receive
an ``OwnerDiedError`` exception, which can be handled manually.

Understanding `ObjectLostErrors`
--------------------------------

Ray throws an ``ObjectLostError`` to the application when an object cannot be
retrieved due to application or system error. This can occur during a
``ray.get()`` call or when fetching a task's arguments, and can happen for a
number of reasons. Here is a guide to understanding the root cause for
different error types:

- ``OwnerDiedError``: The owner of an object, i.e., the Python worker that
  first created the ``ObjectRef`` via ``.remote()`` or ``ray.put()``, has died.
  The owner stores critical object metadata and an object cannot be retrieved
  if this process is lost.
- ``ObjectReconstructionFailedError``: This error is thrown if an object, or
  another object that this object depends on, cannot be reconstructed due to
  one of the limitations described :ref:`above
  <fault-tolerance-objects-reconstruction>`.
- ``ReferenceCountingAssertionError``: The object has already been deleted,
  so it cannot be retrieved. Ray implements automatic memory management through
  distributed reference counting, so this error should not happen in general.
  However, there is a `known edge case <https://github.com/ray-project/ray/issues/18456>`_ that can produce this error.
- ``ObjectFetchTimedOutError``: A node timed out while trying to retrieve a
  copy of the object from a remote node. This error usually indicates a
  system-level bug. The timeout period can be configured using the
  ``RAY_fetch_fail_timeout_milliseconds`` environment variable (default 10
  minutes).
- ``ObjectLostError``: The object was successfully created, but no copy is
  reachable.  This is a generic error thrown when lineage reconstruction is
  disabled and all copies of the object are lost from the cluster.


.. _fault-tolerance-gcs:

GCS Fault Tolerance
===================

Global Control Service (GCS) is a server that manages cluster-level metadata.
It also provides a handful of cluster-level operations including :ref:`actor <ray-remote-classes>`, :ref:`placement groups <ray-placement-group-doc-ref>` and node management.
By default, the GCS is not fault tolerant since all the data is stored in-memory and its failure means that the entire Ray cluster fails.
To make the GCS fault tolerant, HA Redis is required.
Then, when the GCS restarts, it loads all the data from the Redis instance and resumes regular functions.

During the recovery period, the following functions are not available:

- Actor creation, deletion and reconstruction.
- Placement group creation, deletion and reconstruction.
- Resource management.
- Worker node registration.
- Worker process creation.

However, running Ray tasks and actors remain alive and any existing objects will continue to be available.

Setting up Redis
----------------

.. tab-set::

    .. tab-item:: KubeRay (officially supported)

        If you are using :ref:`KubeRay <kuberay-index>`, refer to :ref:`KubeRay docs on GCS Fault Tolerance <kuberay-gcs-ft>`.

    .. tab-item:: ray start

        If you are using :ref:`ray start <ray-start-doc>` to start the Ray head node,
        set the OS environment ``RAY_REDIS_ADDRESS`` to
        the Redis address, and supply the ``--redis-password`` flag with the password when calling ``ray start``:

        .. code-block:: shell

          RAY_REDIS_ADDRESS=redis_ip:port ray start --head --redis-password PASSWORD

    .. tab-item:: ray up

        If you are using :ref:`ray up <ray-up-doc>` to start the Ray cluster, change :ref:`head_start_ray_commands <cluster-configuration-head-start-ray-commands>` field to add ``RAY_REDIS_ADDRESS`` and ``--redis-password`` to the ``ray start`` command:

        .. code-block:: yaml

          head_start_ray_commands:
            - ray stop
            - ulimit -n 65536; RAY_REDIS_ADDRESS=redis_ip:port ray start --head --redis-password PASSWORD --port=6379 --object-manager-port=8076 --autoscaling-config=~/ray_bootstrap_config.yaml --dashboard-host=0.0.0.0

    .. tab-item:: Kubernetes

        If you are using Kubernetes but not :ref:`KubeRay <kuberay-index>`, please refer to :ref:`this doc <deploy-a-static-ray-cluster-without-kuberay>`.


Once the GCS is backed by Redis, when it restarts, it'll recover the
state by reading from Redis. When the GCS is recovering from its failed state, the raylet
will try to reconnect to the GCS.
If the raylet fails to reconnect to the GCS for more than 60 seconds,
the raylet will exit and the corresponding node fails.
This timeout threshold can be tuned by the OS environment variable ``RAY_gcs_rpc_server_reconnect_timeout_s``.

If the IP address of GCS will change after restarts, it's better to use a qualified domain name
and pass it to all raylets at start time. Raylet will resolve the domain name and connect to
the correct GCS. You need to ensure that at any time, only one GCS is alive.

.. note::

  GCS fault tolerance with external Redis is officially supported
  ONLY if you are using :ref:`KubeRay <kuberay-index>` for :ref:`Ray serve fault tolerance <serve-e2e-ft>`.
  For other cases, you can use it at your own risk and
  you need to implement additional mechanisms to detect the failure of GCS or the head node
  and restart it.


.. _ray-core-examples-tutorial:

Ray Core Examples
=================

.. toctree::
    :hidden:
    :glob:

    *

.. Organize example .rst files in the same manner as the
   .py files in ray/python/ray/train/examples.

Below are examples for using Ray Core for a variety use cases.

Beginner
--------

.. list-table::

  * - :doc:`A Gentle Introduction to Ray Core by Example <gentle_walkthrough>`
  * - :doc:`Using Ray for Highly Parallelizable Tasks <highly_parallel>`
  * - :doc:`Monte Carlo Estimation of π <monte_carlo_pi>`
  

Intermediate
------------

.. list-table::

  * - :doc:`Running a Simple MapReduce Example with Ray Core <map_reduce>`
  * - :doc:`Speed Up Your Web Crawler by Parallelizing it with Ray <web-crawler>`


Advanced
--------

.. list-table::
    
  * - :doc:`Build Simple AutoML for Time Series Using Ray <automl_for_time_series>`
  * - :doc:`Build Batch Prediction Using Ray <batch_prediction>`
  * - :doc:`Build a Simple Parameter Server Using Ray <plot_parameter_server>`
  * - :doc:`Simple Parallel Model Selection <plot_hyperparameter>`
  * - :doc:`Learning to Play Pong <plot_pong_example>`


.. _monte-carlo-pi:

Monte Carlo Estimation of π
===========================

This tutorial shows you how to estimate the value of π using a `Monte Carlo method <https://en.wikipedia.org/wiki/Monte_Carlo_method>`_
that works by randomly sampling points within a 2x2 square.
We can use the proportion of the points that are contained within the unit circle centered at the origin
to estimate the ratio of the area of the circle to the area of the square.
Given that we know the true ratio to be π/4, we can multiply our estimated ratio by 4 to approximate the value of π.
The more points that we sample to calculate this approximation, the closer the value should be to the true value of π.

.. image:: ../images/monte_carlo_pi.png

We use Ray :ref:`tasks <ray-remote-functions>` to distribute the work of sampling and Ray :ref:`actors <ray-remote-classes>` to track the progress of these distributed sampling tasks.
The code can run on your laptop and can be easily scaled to large :ref:`clusters <cluster-index>` to increase the accuracy of the estimate.

To get started, install Ray via ``pip install -U ray``. See :ref:`Installing Ray <installation>` for more installation options.

Starting Ray
------------
First, let's include all modules needed for this tutorial and start a local Ray cluster with :func:`ray.init() <ray.init>`:

.. literalinclude:: ../doc_code/monte_carlo_pi.py
    :language: python
    :start-after: __starting_ray_start__
    :end-before: __starting_ray_end__

.. note::

  In recent versions of Ray (>=1.5), ``ray.init()`` is automatically called on the first use of a Ray remote API.


Defining the Progress Actor
---------------------------
Next, we define a Ray actor that can be called by sampling tasks to update progress.
Ray actors are essentially stateful services that anyone with an instance (a handle) of the actor can call its methods.

.. literalinclude:: ../doc_code/monte_carlo_pi.py
    :language: python
    :start-after: __defining_actor_start__
    :end-before: __defining_actor_end__

We define a Ray actor by decorating a normal Python class with :func:`ray.remote <ray.remote>`.
The progress actor has ``report_progress()`` method that will be called by sampling tasks to update their progress individually
and ``get_progress()`` method to get the overall progress.

Defining the Sampling Task
--------------------------
After our actor is defined, we now define a Ray task that does the sampling up to ``num_samples`` and returns the number of samples that are inside the circle.
Ray tasks are stateless functions. They execute asynchronously, and run in parallel.

.. literalinclude:: ../doc_code/monte_carlo_pi.py
    :language: python
    :start-after: __defining_task_start__
    :end-before: __defining_task_end__

To convert a normal Python function as a Ray task, we decorate the function with :func:`ray.remote <ray.remote>`.
The sampling task takes a progress actor handle as an input and reports progress to it.
The above code shows an example of calling actor methods from tasks.

Creating a Progress Actor
-------------------------
Once the actor is defined, we can create an instance of it.

.. literalinclude:: ../doc_code/monte_carlo_pi.py
    :language: python
    :start-after: __creating_actor_start__
    :end-before: __creating_actor_end__

To create an instance of the progress actor, simply call ``ActorClass.remote()`` method with arguments to the constructor.
This creates and runs the actor on a remote worker process.
The return value of ``ActorClass.remote(...)`` is an actor handle that can be used to call its methods.

Executing Sampling Tasks
------------------------
Now the task is defined, we can execute it asynchronously.

.. literalinclude:: ../doc_code/monte_carlo_pi.py
    :language: python
    :start-after: __executing_task_start__
    :end-before: __executing_task_end__

We execute the sampling task by calling ``remote()`` method with arguments to the function.
This immediately returns an ``ObjectRef`` as a future
and then executes the function asynchronously on a remote worker process.

Calling the Progress Actor
--------------------------
While sampling tasks are running, we can periodically query the progress by calling the actor ``get_progress()`` method.

.. literalinclude:: ../doc_code/monte_carlo_pi.py
    :language: python
    :start-after: __calling_actor_start__
    :end-before: __calling_actor_end__

To call an actor method, use ``actor_handle.method.remote()``.
This invocation immediately returns an ``ObjectRef`` as a future
and then executes the method asynchronously on the remote actor process.
To fetch the actual returned value of ``ObjectRef``, we use the blocking :func:`ray.get() <ray.get>`.

Calculating π
-------------
Finally, we get number of samples inside the circle from the remote sampling tasks and calculate π.

.. literalinclude:: ../doc_code/monte_carlo_pi.py
    :language: python
    :start-after: __calculating_pi_start__
    :end-before: __calculating_pi_end__

As we can see from the above code, besides a single ``ObjectRef``, :func:`ray.get() <ray.get>` can also take a list of ``ObjectRef`` and return a list of results.

If you run this tutorial, you will see output like:

.. code-block:: text

 Progress: 0%
 Progress: 15%
 Progress: 28%
 Progress: 40%
 Progress: 50%
 Progress: 60%
 Progress: 70%
 Progress: 80%
 Progress: 90%
 Progress: 100%
 Estimated value of π is: 3.1412202


