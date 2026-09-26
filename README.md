# 🧩 Simple 2D Image Segmenter User Interface
This application can be used locally or in headless secure server to create a segmentation mask of 2D images. 
It includes Tkinter based windows/linux usage or 
a pipeline in streamlit to run in all platforms (incl. headless server).

Further information can be obtained by writing to Arka Bhowmik (arkabhowmik@yahoo.co.uk).

## ⊞ Windows app (follow the steps below to execute app)
![desktop_app](https://github.com/user-attachments/assets/c6722515-47a0-460f-a253-7b8bc9f323c3)
Users can use the files in `main\` and follow below steps.

#### Step I: Download and install Python
Open **Command Prompt** on Windows and run:

```cmd
winget install Python.Python.3.12
python --version
pip --version
```
Use python --version and pip --version to confirm that Python was installed correctly. 

#### Step II: Copy the downloaded files from *main* to:
```
C:\Users\YourName\Segmentation_app
```
#### Step III: create/activate a virtual environment
In Command Prompt, run:
```cmd
cd C:\Users\YourName\Segmentation_app\
python -m venv venv
venv\Scripts\activate
```
#### Step IV: Install the requirements.txt
```cmd
python -m pip install -r requirements.txt
```
#### Step V: Execution

Run any one of 1-3 options:

```cmd
# 1. opens a csv picker window
python main.py
# 2. provide the CSV directly
python main.py "C:\Users\YourName\Segmentation_app\sample_case.csv"
# 3. use directory + filename arguments
python main.py --path_to_csv "C:/Users/YourName/Segmentation_app/" --csv_file "sample_case.csv"
```


## 🐧 Linux app (follow the steps below to execute app)
![desktop_app](https://github.com/user-attachments/assets/c6722515-47a0-460f-a253-7b8bc9f323c3)
Users can use the files in `main\` and follow below steps.

#### Step I: Download and install Python
Open **Terminal** on Linux and run:

```bash
sudo apt-get install python3-tk
```

#### Step II: Copy the downloaded files from *main* to:
```
/home/YourName/Segmentation_app/
```
#### Step III: create/activate a virtual environment
In bash run:
```bash
cd /home/YourName/Segmentation_app/
python3 -m venv .venv
source .venv/bin/activate
```
#### Step IV: Install the requirements.txt
```bash
python3 -m pip install -r requirements.txt
```
#### Step V: Execution

Run any one of 1-3 options:

```bash
# 1. opens a csv picker window
python main.py
# 2. provide the CSV directly
python main.py "/home/YourName/Segmentation_app/sample_case.csv"

## ✨ Streamlit app (follow the steps below to run the app)
<img width="1464" height="1246" alt="streamlit_app" src="https://github.com/user-attachments/assets/faff7744-138f-4381-9d1c-a6449200b055" />
Users can use the files in `main\streamlit_audio2text\` and follow these steps to run the app.

#### Step I: Download, install python, activate virtual environment

Open **Command Prompt/Terminal** on Windows/Linux/Server and run:
```bash
# 1. Windows
winget install Python.Python.3.12
cd C:\Users\YourName\Segmentation_app\
python -m venv venv
venv\Scripts\activate
# 2. Linux/Mac/Headless Server
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /data/user/myenv/miniconda.sh # Change user defined directory /data/user/myenv/
bash /data/user/myenv/miniconda.sh -b -u -p /data/user/myenv/miniconda3 # Change user defined directory /data/user/myenv/
/data/user/myenv/miniconda3/bin/conda init bash # Change user defined directory /data/user/myenv/
conda create -p /data/user/myenv/venv python=3.12 # Change user defined directory /data/user/myenv/
conda activate /data/user/myenv/venv # Change user defined directory /data/user/myenv/
```

#### Step II: Copy the downloaded files from *main\* to:
```bash
# 1. windows
C:\Users\YourName\Segmentation_app\
# 2. Linux/Mac/Headless Server
/data/user/Segmentation_app/
```

#### Step III: Run the `requirements_headless.txt` file inside environment
In **Command Prompt/Terminal**, run:
```
python -m pip install -r requirements_headless.txt
```

#### Step IV: Run the `app.py` file
In **Command Prompt/Terminal**, run:
```bash
streamlit run app.py -- --path_to_csv "/data/user/Segmentation_app" --csv_file "sample_case.csv"
```
*This will provide an url (e.g., "http://localhost:8501") for the app that can be copied to the browser of local machine.* 

## Notes:
- The Help image <img width="30" height="50" alt="Help_picture" src="https://github.com/user-attachments/assets/498e33cd-80a1-4f57-a6f1-7e82978b67b6" /> saves a local copy of the bundled help PDF.
# 3. use directory + filename arguments
python main.py --path_to_csv "/home/YourName/Segmentation_app/" --csv_file "sample_case.csv"
