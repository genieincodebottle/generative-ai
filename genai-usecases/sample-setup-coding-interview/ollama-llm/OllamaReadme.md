## Use Window Powershell to run following steps

### Step 1: Install WSL(Windows Subsystem for Linux)
wsl --install

### Step 2: List the different Linux Distribution that you can choose to install 

wsl.exe -l -o

### Step 3: install Linux distribtion received from above command example as below Ubuntu-22.04

wsl --install Ubuntu-22.04

### Step 4: Install Ollama at WSL

curl https://ollama.ai/install.sh | sh

### Step 5: Run Opensource LLM a per your choice example as below using mistral LLM

ollama run mistral

### Step 5: Ask LLM question and get answer :)

Now you can ask question from LLM to get response at Windows powershell terminal.


## Step 6 - Use Ollama based open source LLM. Follow the steps given in following link

* https://python.langchain.com/v0.2/docs/integrations/llms/ollama/