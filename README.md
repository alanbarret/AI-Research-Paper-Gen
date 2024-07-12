### Hosted on Streamlit:

To use ResearchPaper, you can use the hosted version at [ai-research-paper.streamlit.app](https://ai-research-paper.streamlit.app/)

### Run locally:

Alternatively, you can run ResearchPaper locally with Streamlit.

#### Step 1
First, set your Groq API key in the environment variables:

~~~
export GROQ_API_KEY="gsk_yA..."
~~~

This is an optional step that allows you to skip setting the Groq API key later in the Streamlit app.

#### Step 2
Next, set up a virtual environment and install the dependencies.

~~~
python3 -m venv venv
~~~

~~~
source venv/bin/activate # Bash

venv\Scripts\activate.bat # Windows
~~~

~~~
pip3 install -r requirements.txt
~~~

#### Step 3 (Windows Only)
It may be required to install GTK3 for users on Windows.

~~~
https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer?tab=readme-ov-file
~~~

#### Step 4
Finally, run the Streamlit app.

~~~
python3 -m streamlit run main.py
~~~

## Details

### Technologies

- Streamlit
- Mixtral 8x7b on Groq Cloud

### Limitations

ResearchPaper may generate placeholder content or information that requires verification. It is recommended to use ResearchPaper for drafting research papers and then refine the content with verified information.
