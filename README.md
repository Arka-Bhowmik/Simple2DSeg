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
# 3. use directory + filename arguments
python main.py --path_to_csv "/home/YourName/Segmentation_app/" --csv_file "sample_case.csv"
