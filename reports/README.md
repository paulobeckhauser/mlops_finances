# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```
## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 92

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s242479, S243418, s244086, s225526

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:
For our project we used TorchMetrics, which can ensure reliable monitoring and comparison
of your model's performance, such as accuracy metric, across experiments.

We tried to use the third-party framework PyTorch Forecasting in our project, but couldn't 
implement it in the correct way, since some data preprocessing is necessary. 
So it is in the backlog list to be implemented. 

PyTorch Forecasting helps in handling the time-series forecasting aspect of our work, 
which involved predicting the parity price of USD/CHF combined with economic calendar data. 
It provides powerful tools such as the Temporal Fusion Transformer (TFT) model, 
which performs well in capturing relationships across time-varying covariates and static features.


We also implemented Black, which is an open-source tool to automate the process of following the best coding practices(e.g. pep8) in the python code.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

Initially, we used Conda for managing system-level dependencies and ensuring compatibility across different platforms, particularly for libraries requiring specific configurations, such as black for formatting. But then we also created a virtual environment using .venv to isolate project-specific Python dependencies. A member of the team also used 'uv' package manager which is built in Rust, and helps to guarantee compatibility between all the libraries versions, and has great results regarding the installation time comparing to only 'pip install...'.

The process would be: 1. Install python; 2. Clone the repository with 'git clone <repository-url> ; 3. Create a virtual environment with 'python -m venv .venv'; 4. Activate virtual environment with 'source .venv/bin/activate' in a linux/macOS environment; 5. Install dependencies with 'pip install -r requirements.txt'

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

*From the cookiecutter template, we have filled out the data, dockerfiles, models, src, and test folders. The data folder was used to organize our raw and processed datasets, ensuring a structured workflow. The dockerfiles folder allowed us to set up a reproducible environment for our project using Docker, making collaboration and deployment seamless. The models folder stored our trained models and associated metadata. The src folder contained all the source code, including preprocessing scripts, model training, and evaluation pipelines. Finally, the test folder was used to ensure our code's functionality through unit and integration tests.

We debated removing the notebooks folder since we rarely used Jupyter notebooks during the project, opting for .py files for prototyping and development to maintain a consistent coding style. Additionally, we added checkpoints and logs folders. The checkpoints folder stored intermediate training states, which allowed us to resume training efficiently, while the logs folder contained experiment details, including training metrics and configurations, making it easier to monitor and reproduce results. These additions were critical for managing and tracking our experiments systematically.*

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We implemented several rules to ensure code quality and consistency. First, we adhered to the PEP 8 guidelines, which define best practices for Python code formatting, including indentation, naming conventions, and spacing. To automate this process, we used the Black library, which enforces a consistent code style by reformatting our code automatically. This helped us maintain a clean and standardized codebase, reducing potential errors caused by inconsistent formatting.

We used docstrings to describe functions, classes, and modules, detailing their purpose, parameters, and expected behavior. This practice was good for team collaboration and onboarding, as it allowed new contributors to understand the code quickly.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total, we have implemented 4 tests. The first test verifies the raw data, ensuring there are exactly two non-empty files. The second test focuses on preprocessing, validating that the data transformations work as expected. The third test evaluates the model, testing its initialization, forward pass, and training functionality. Finally, the fourth test targets the API, ensuring it handles requests and returns responses correctly. These tests cover the most critical parts of the application: data integrity, preprocessing, model performance, and API functionality.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our code is currently 23%, which includes all our source code. This number indicates that we are far from achieving 100% coverage, highlighting areas of the code that are not yet tested. Increasing test coverage is on our to-do list, as it is essential for improving the reliability and robustness of our application.

However, even with 100% code coverage, we would not fully trust the code to be error-free. Code coverage only ensures that the lines of code are executed during testing, but it does not guarantee the correctness of the logic or handle all edge cases. Tests might miss certain scenarios, or the logic might fail under specific conditions. Therefore, while striving for high coverage is important, the quality and thoroughness of the tests, combined with other strategies like code reviews and performance testing, are crucial for ensuring a robust and error-free application.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made extensive use of branches and pull requests (PRs) in our project to streamline collaboration and maintain code quality. We implemented a rule to protect the main branch, requiring that any code merged into it be reviewed and approved by at least one other team member. This ensured that only thoroughly reviewed and tested code made its way into the main branch.

Each team member worked on their own branch, typically named based on the specific feature or bug fix being implemented. This approach kept individual contributions isolated, reducing the risk of conflicts and enabling parallel development. When a feature or fix was ready, the contributor merged their branch with their local main branch to verify compatibility. Afterward, they pushed the changes to a remote branch and created a pull request (PR). The PR served as a platform for discussions, comments, and code reviews.

This workflow improved version control by providing a clear history of changes, encouraging collaboration, and minimizing the risk of introducing bugs into the main branch. It also facilitated accountability and knowledge sharing among team members, making the development process more efficient and reliable.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did not use DVC in our project, but there are many scenarios where having version control of data would be highly beneficial. For example, in machine learning projects where datasets evolve over time, DVC can help track changes to the data, such as adding new samples, correcting labels, or preprocessing adjustments. By using DVC, you can maintain a version history of your data, making it easier to roll back to previous versions or analyze the impact of specific changes on model performance.

Another use case is collaborative projects where multiple team members work with the same dataset. DVC allows syncing data across team members without manually sharing large files, ensuring everyone is working with the same version. It also integrates seamlessly with Git, enabling you to version control data alongside your code.

In large-scale projects, reproducibility is a critical factor. DVC stores metadata about which datasets, models, and configurations were used for a specific experiment, ensuring results can be replicated accurately. This makes it an invaluable tool for managing complex data pipelines and improving workflow efficiency in data-driven projects.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have organized our continuous integration (CI) setup into two separate workflows to ensure a modular and efficient pipeline. Each workflow targets a specific task: one for unit testing, one for linting, and one for deployment. This separation of concerns improves maintainability and scalability as our project grows.

For unit testing, we use pytest to validate our code functionality across multiple operating systems (Ubuntu, macOS, Windows) and Python versions (3.8 to 3.11). This ensures compatibility and robustness in diverse environments. We also implement caching for Python dependencies using GitHub’s actions/cache, which significantly reduces workflow execution time by avoiding redundant installations.

In our linting workflow, we use flake8 to enforce code style and maintain high readability and quality. This step helps catch potential issues early in the development cycle, reducing technical debt over time. This also showed a lot of improvements were needed in our code.

By leveraging caching, running workflows on multiple platforms, and testing against different Python versions, we have created a robust CI pipeline that ensures our code is reliable and production-ready. An example of one of our workflows can be found <[here](https://github.com/paulobeckhauser/mlops_finances/actions/runs/12845256661)>. This setup provides a strong foundation for continuous improvement and rapid iteration in our development process.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured our experiments using Hydra, a powerful library for managing configuration files and command-line arguments. Hydra allows us to structure experiments through YAML-based config files and override parameters dynamically from the command line. We defined our configuration in the files 'config.yaml' and 'deep_learning.yaml' in the folder 'configs'. In order to be able to run an experiment in our project, just need to run the following code, for example: 'python src/finance/main.py model.epochs=30 model.lr=0.001'

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensured reproducibility of experiments by leveraging Hydra for managing configurations and logging. All experiment parameters are defined in YAML configuration files (config.yaml and model-specific files like deep_learning.yaml), ensuring that no information is hard-coded in the scripts. This allows for easy tracking and replication of experiment settings.

Whenever an experiment is run, Hydra automatically organizes outputs into uniquely named directories based on the configuration and timestamp. These directories store logs, checkpoints, and any generated outputs, ensuring that each run is isolated and fully documented.

To reproduce an experiment, one simply needs to reference the configuration used during the run. Hydra also allows for saving and loading configurations, so experiments can be rerun with the exact same settings by pointing to the relevant configuration file or overriding parameters via the command line.

This approach guarantees that all essential details, from hyperparameters to outputs, are preserved, enabling full reproducibility and traceability of experiments.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

Docker was essential in our project for containerizing the training workflow, and ensuring reproducibility. We used Docker to isolate dependencies, configurations and source code for training a deep learning model. The trainer.dockerfile was designed to build a lightweight Python container with all required packages and the project code. To build and run the Docker image, we executed the following commands:
> 1. Build the image: docker build -t finance-trainer -f dockerfiles/trainer.dockerfile .
> 2. Run the container: docker run finance-trainer'
> This trains the model, evaluate its performance and saves the results
The final output includes metrics like accuracy (0.5484) and the trained model saved at model/model.pth. This containerization ensures that anyone with Docker can reproduce the results without additional setup.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

When encountering bugs, our primary debugging method involved using print statements to trace variables and program flow. This simple approach helped identify unexpected values or incorrect logic. For more complex issues, we utilized the debugging tools within our integrated development environment to step through the code and inspect variables at various breakpoints. While we recognize the importance of code profiling for performance optimization, we didn't prioritize it due to time constraints and our focus on achieving functional correctness. We acknowledge that our code likely has areas for improvement in terms of efficiency, but profiling wasn't a primary focus during this stage of development.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following services:
1. Google Cloud Compute Engine:  Used to run machine learning workflows and MLOps pipelines by deploying a virtual machine (VM) instance. The VM was configured with a standard persistent disk (10 GB storage), Debian 12 operating system, and Google-managed encryption keys for data security. This service provided the computational resources to preprocess data, train, and evaluate models securely and cost-effectively. 
2. Google Cloud Storage: Utilized to store raw and processed data for financial analysis. Cloud Storage ensured efficient data access and sharing during the project.
3. Artifact Registry: Used to store Docker container images required for machine learning tasks. Docker images, like the trainer, containing dependencies and configurations were built locally and pushed to Artifact Registry. This ensured seamless integration with Vertex AI for custom job execution.
4. Vertex AI: Employed to train machine learning models in the cloud. A custom job was created using a containerized training script (src/finance/main.py) hosted in Artifact Registry. The service was chosen for its scalability, integration with GCP services, and monitoring tools to train and manage large models.
5. Cloud Build: Used to automate the process of building and pushing Docker images to Artifact Registry. Build triggers ensured an efficient CI/CD pipeline, reducing manual intervention and ensuring the consistency of container images.
6. Cloud Logging: Enabled monitoring and debugging by collecting logs from Compute Engine, Vertex AI, and other resources. Logs were used to identify issues, such as incorrect Docker image URIs, and ensure smooth operations during custom job submissions.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the Google Cloud Compute Engine to run our machine learning workflows and MLOps pipelines. Specifically, we deployed a virtual machine instance with the following configuration:

Machine type: Standard persistent disk with 10 GB of storage.
Architecture: x86/64.
Zone: europe-west1-b for optimized latency and regional availability.
Operating system: Debian 12 (Bookworm), initialized from the source image debian-12-bookworm-v20250113.
This virtual machine, provided the flexibility to scale our operations and execute computational tasks efficiently. The instance was encrypted with Google-managed keys for data security, and it was configured to use a lightweight, open-source operating system, ensuring low overhead while running containerized applications. We leveraged this setup for hosting containers that facilitated data preprocessing, training, and evaluation in a secure and cost-effective manner.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![GCP Bucket](https://github.com/paulobeckhauser/mlops_finances/blob/main/reports/figures/GCP_buckets.png)
![Data Stored to GCP bucket](https://github.com/paulobeckhauser/mlops_finances/blob/main/reports/figures/GCP_bucket_data.png)



### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![GCP Artifact Registry - Repositories](https://github.com/paulobeckhauser/mlops_finances/blob/main/reports/figures/Artifact%20Registry.png)
![Docker images that were uploaded](https://github.com/paulobeckhauser/mlops_finances/blob/main/reports/figures/Docker_image_finance_project.png)

The docker image 'trainer' is the one related to our project, as is possible to see in [Trainer Dockerfile](https://github.com/paulobeckhauser/mlops_finances/blob/main/dockerfiles/trainer.dockerfile)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![GCP Cloud Build History](https://github.com/paulobeckhauser/mlops_finances/blob/main/reports/figures/GCP%20Cloud%20Build%20History.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We successfully trained our model in the cloud using Vertex AI. To accomplish this, we first containerized our training script (src/finance/main.py) by creating a Docker image containing all dependencies and configurations. The image was built locally using a Dockerfile and pushed to Artifact Registry in the europe-west1 region. We configured the necessary permissions by granting the Artifact Registry Reader role to the Vertex AI service account.

Next, we created a custom job by defining a config.yaml file that specified the machine type, region, and container image URI. The custom job was submitted using the gcloud ai custom-jobs create command. We chose Vertex AI because it provided seamless integration with GCP services, scalability, and robust monitoring tools for training large models.

During the setup, we encountered an issue where the Docker image was not found due to an incorrect URI, but this was resolved by ensuring proper regional alignment and precise image naming in the Artifact Registry. The job completed successfully, and the model was trained as expected.

![Jobs Vertex AI](https://github.com/paulobeckhauser/mlops_finances/blob/main/reports/figures/Jobs_VertexAI.png)

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We did manage to write an API for our model using FastAPI. We created the API to serve a trained deep learning model implemented with PyTorch and used FastAPI's features to show prediction functionality through HTTP endpoints. The API loads the model architecture, restores the trained weights using torch.load, and uses the model to make predictions.

To handle user inputs, the API accepts JSON data, converts it into a Pandas DataFrame for preprocessing, and then transforms it into PyTorch tensors for model inference. The /predict endpoint processes these inputs and returns both class predictions and probabilities in JSON format.

Additionally, we implemented error handling to ensure clear feedback if the model weights are missing or if the input format is invalid. 

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

Yes, we managed to deploy our API locally using FastAPI and Docker. To achieve this, we first built the FastAPI application to serve a trained PyTorch model. The API loads the model architecture and weights, sets the model to evaluation mode, and handles predictions.
We then containerized the application using a Dockerfile, ensuring that all dependencies, including the trained model, were installed in the container. The Docker image was built with the following command:
- docker build -t my_model_api -f dockerfiles/api.dockerfile .

After building the image, we ran the container locally, exposing the API on port 8000:
- docker run -d -p 8000:8000 my_model_api
The deployed API can be invoked through the /predict endpoint by sending a POST request with JSON data. For example, using curl:
- curl -X POST "http://localhost:8000/predict/" -H "Content-Type: application/json" -d '{"feature1": 1.0, "feature2": 0.5}'
  
> This returns predictions and probabilities in JSON format. The deployment was successful, and the API could be accessed at http://localhost:8000.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:



### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not manage to implement monitoring for our deployed model. Implementing monitoring would have been valuable to measure the model's performance over time, including metrics such as prediction accuracy, latency, and error rates. This information would help identify any drift in the model's behavior, especially since the data patterns in time series applications can change over time.

Additionally, monitoring could have provided insights into resource usage, such as memory and CPU consumption, helping optimize the application's efficiency and scalability. It would also allow us to set up alerts for anomalies or underperformance, enabling quick intervention to minimize disruptions for end-users.

Although monitoring was not implemented during this project, it is a critical feature for ensuring the longevity and reliability of the application and is something we plan to prioritize in future developments.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

Group member 1 used 2.37 credits, and Group member 2 used 6.45 credits, resulting in a total of 8.82 credits spent during the development of the project. The service costing the most was related to instances, primarily due to running virtual machines for extended periods to train the time series model and test the Dockerized environments. The instances were resource-intensive, as they required significant computational power and uptime for debugging and fine-tuning.

Working in the cloud was a valuable experience overall. It provided flexibility and scalability that would have been challenging to achieve with local resources. However, the cost management and configuration complexities required constant monitoring and learning. Despite these challenges, using the cloud allowed the team to collaborate efficiently and access powerful tools that significantly contributed to the project's development.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

A member used the UV Package Manager that combined with Virtual Environment helps to guarantee a convergence of versions between all the libraries that are installed in the virtual environment.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![Project Diagram](https://github.com/paulobeckhauser/mlops_finances/blob/main/reports/figures/Project_diagram.jpg)

The project utilizes Git for version control and Docker for managing system-level dependencies. A virtual environment, .venv, is employed for Python dependencies, along with the requirements.txt file for pip installations. The uv package manager enhances library compatibility. The project follows a structured codebase using a Cookiecutter template, featuring directories for data, Dockerfiles, models, src, and tests, with additional directories for checkpoints and logs.

To maintain code quality, Black enforces PEP 8 formatting, and docstrings provide documentation. Four unit tests cover data verification, preprocessing, model evaluation, and API functionality, achieving 23% code coverage.

Continuous Integration (CI)
CI workflows on GitHub Actions include unit testing with pytest and linting with flake8. Tests run across multiple operating systems (Ubuntu, macOS, Windows) and Python versions (3.8–3.11). GitHub's actions/cache speeds up dependency installation.

Experiment Tracking and Configuration
Hydra manages experiment configurations using YAML files, organizing outputs into timestamped directories for reproducibility. Weights & Biases (W&B) tracks metrics, including loss graphs and hyperparameter sweeps, for improved experiment management.

Containerization
Docker is used for containerizing workflows. The trainer.dockerfile creates a lightweight Python image for training, while api.dockerfile packages the API application. Docker commands streamline image creation and deployment.

Cloud Services
The project leverages Google Cloud Platform (GCP) for key services. A VM instance on Google Compute Engine runs workflows, while Google Cloud Storage manages raw and processed data. Artifact Registry stores Docker images, and Vertex AI facilitates cloud-based model training with containerized scripts. Cloud Build automates Docker image creation, and Cloud Logging provides monitoring and debugging tools.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- The biggest challenge in the project was understanding the infrastructure and effectively utilizing Google Cloud Platform (GCP). As we relied entirely on GCP, there was a steep learning curve in grasping its various tools, services, and configurations. Setting up the environment, managing permissions, and optimizing workflows required significant effort and time.

Another major struggle was working with Docker. While it offered a way to standardize our deployments, we faced performance issues as the containers often took too long to build and run. Debugging and optimizing the containerization process became an unexpected time sink, particularly when combined with API integration challenges. Using the correct ports for API access proved tricky, requiring several iterations and troubleshooting to ensure seamless communication between components.

Building the model itself also posed significant difficulties, as we were working with time series data. Designing a model capable of accurately capturing the patterns and nuances of time-dependent data demanded careful consideration of feature engineering, hyperparameter tuning, and architecture selection. Despite our efforts, we believe the model is underperforming, likely due to challenges in balancing complexity and overfitting while working with a relatively volatile dataset.

To address these challenges, we dedicated time to learning GCP's documentation and tutorials, sought community support for Docker optimizations, and conducted multiple tests to refine our API configurations. On the modeling side, we iteratively tested different approaches and are now focusing on further fine-tuning and exploring advanced methods to improve performance. Despite the hurdles, these experiences have provided valuable learning opportunities for future projects. ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Student s242779 was responsible for setting up the initial project infrastructure on Google Cloud Platform (GCP). This included configuring the cloud environment, managing permissions, and optimizing workflows. They also worked on troubleshooting API port configurations to ensure smooth communication between components.

Student S243418 led the development and optimization of Docker containers, ensuring standardized and functional environments for the project. They addressed performance issues with container execution and refined the containerization process to minimize delays.

Student s225526 focused on developing and training the time series model. This involved preprocessing the data, feature engineering, and iteratively tuning the model. They also analyzed the model's underperformance to identify areas for improvement. Also implemented Hydra.

Student s244086 was in charge of maintaining code quality and ensuring compliance with PEP8 standards. They performed regular code reviews, refactored sections for better readability, and implemented automated linting tools to maintain a consistent coding style across the project. Also implemented DVC.
