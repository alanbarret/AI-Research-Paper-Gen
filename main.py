import streamlit as st
from groq import Groq
import json
import os
from io import BytesIO
from markdown import markdown
from weasyprint import HTML, CSS
from dotenv import load_dotenv
import fitz  # PyMuPDF
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile

# Load .env file to environment
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

if "api_key" not in st.session_state:
    st.session_state.api_key = GROQ_API_KEY

if "groq" not in st.session_state:
    if GROQ_API_KEY:
        st.session_state.groq = Groq()


class GenerationStatistics:
    def __init__(
        self,
        input_time=0,
        output_time=0,
        input_tokens=0,
        output_tokens=0,
        total_time=0,
        model_name="llama3-8b-8192",
    ):
        self.input_time = input_time
        self.output_time = output_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = (
            total_time  # Sum of queue, prompt (input), and completion (output) times
        )
        self.model_name = model_name

    def get_input_speed(self):
        """
        Tokens per second calculation for input
        """
        if self.input_time != 0:
            return self.input_tokens / self.input_time
        else:
            return 0

    def get_output_speed(self):
        """
        Tokens per second calculation for output
        """
        if self.output_time != 0:
            return self.output_tokens / self.output_time
        else:
            return 0

    def add(self, other):
        """
        Add statistics from another GenerationStatistics object to this one.
        """
        if not isinstance(other, GenerationStatistics):
            raise TypeError("Can only add GenerationStatistics objects")

        self.input_time += other.input_time
        self.output_time += other.output_time
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_time += other.total_time

    def __str__(self):
        return (
            f"\n## {self.get_output_speed():.2f} T/s ⚡\nRound trip time: {self.total_time:.2f}s  Model: {self.model_name}\n\n"
            f"| Metric          | Input          | Output          | Total          |\n"
            f"|-----------------|----------------|-----------------|----------------|\n"
            f"| Speed (T/s)     | {self.get_input_speed():.2f}            | {self.get_output_speed():.2f}            | {(self.input_tokens + self.output_tokens) / self.total_time if self.total_time != 0 else 0:.2f}            |\n"
            f"| Tokens          | {self.input_tokens}            | {self.output_tokens}            | {self.input_tokens + self.output_tokens}            |\n"
            f"| Inference Time (s) | {self.input_time:.2f}            | {self.output_time:.2f}            | {self.total_time:.2f}            |"
        )


class ResearchPaper:
    def __init__(self, paper_title, structure):
        self.paper_title = paper_title
        self.structure = structure
        self.contents = {title: "" for title in self.flatten_structure(structure)}
        self.placeholders = {title: st.empty() for title in self.flatten_structure(structure)}
        st.markdown(f"# {self.paper_title}")
        st.markdown("## Generating the following:")
        toc_columns = st.columns(4)
        self.display_toc(self.structure, toc_columns)
        st.markdown("---")

    def flatten_structure(self, structure):
        sections = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self.flatten_structure(content))
        return sections

    def update_content(self, title, new_content):
        try:
            self.contents[title] += new_content
            self.display_content(title)
        except TypeError as e:
            pass

    def display_content(self, title):
        if self.contents[title].strip():
            self.placeholders[title].markdown(f"## {title}\n{self.contents[title]}")

    def display_structure(self, structure=None, level=1):
        if structure is None:
            structure = self.structure
            
        for title, content in structure.items():
            if self.contents[title].strip():  # Only display title if there is content
                st.markdown(f"{'#' * level} {title}")
                self.placeholders[title].markdown(self.contents[title])
            if isinstance(content, dict):
                self.display_structure(content, level + 1)

    def display_toc(self, structure, columns, level=1, col_index=0):
        for title, content in structure.items():
            with columns[col_index % len(columns)]:
                st.markdown(f"{' ' * (level-1) * 2}- {title}")
            col_index += 1
            if isinstance(content, dict):
                col_index = self.display_toc(content, columns, level + 1, col_index)
        return col_index

    def get_markdown_content(self, structure=None, level=1):
        """
        Returns the markdown styled pure string with the contents.
        """
        if structure is None:
            structure = self.structure
        
        if level==1:
            markdown_content = f"# {self.paper_title}\n\n"
            
        else:
            markdown_content = ""
        
        for title, content in structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                markdown_content += f"{'#' * level} {title}\n{self.contents[title]}\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        return markdown_content


def create_markdown_file(content: str) -> BytesIO:
    """
    Create a Markdown file from the provided content.
    """
    markdown_file = BytesIO()
    markdown_file.write(content.encode("utf-8"))
    markdown_file.seek(0)
    return markdown_file


def create_pdf_file(content: str) -> str:
    """
    Create a PDF file from the provided Markdown content.
    Converts Markdown to styled HTML, then HTML to PDF.
    """
    try:
        html_content = markdown(content, extensions=["extra", "codehilite"])

        styled_html = f"""
        <html>
            <head>
                <style>
                    @page {{
                        margin: 2cm;
                    }}
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        font-size: 12pt;
                    }}
                    h1, h2, h3, h4, h5, h6 {{
                        color: #333366;
                        margin-top: 1em;
                        margin-bottom: 0.5em;
                    }}
                    p {{
                        margin-bottom: 0.5em;
                    }}
                    code {{
                        background-color: #f4f4f4;
                        padding: 2px 4px;
                        border-radius: 4px;
                        font-family: monospace;
                        font-size: 0.9em;
                    }}
                    pre {{
                        background-color: #f4f4f4;
                        padding: 1em;
                        border-radius: 4px;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }}
                    blockquote {{
                        border-left: 4px solid #ccc;
                        padding-left: 1em;
                        margin-left: 0;
                        font-style: italic;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin-bottom: 1em;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    input, textarea {{
                        border-color: #4A90E2 !important;
                    }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            HTML(string=styled_html).write_pdf(tmp_file.name)
            tmp_file.seek(0)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error creating PDF file: {e}")
        return None

def generate_paper_title(prompt: str):
    """
    Generate a research paper title using AI.
    """
    completion = st.session_state.groq.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": "Generate suitable research paper titles for the provided topics. There is only one generated paper title! Don't give any explanation or add any symbols, just write the title of the paper. The requirement for this title is that it must be between 7 and 25 words long, and it must be attractive enough!"
            },
            {
                "role": "user",
                "content": f"Generate a research paper title for the following topic. There is only one generated paper title! Don't give any explanation or add any symbols, just write the title of the paper. The requirement for this title is that it must be at least 7 words and 25 words long, and it must be attractive enough:\n\n{prompt}"
            }
        ],
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content.strip()

def generate_paper_structure(prompt: str):
    """
    Returns research paper structure content as well as total tokens and total time for generation.
    """
    completion = st.session_state.groq.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": 'Write in JSON format:\n\n{"Title of section goes here":"Description of section goes here",\n"Title of section goes here":{"Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here"}}',
            },
            {
                "role": "user",
                "content": f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a long (>300 page) research paper. It is very important that use the following subject and additional instructions to write the paper. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>",
            },
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    usage = completion.usage
    statistics_to_return = GenerationStatistics(
        input_time=usage.prompt_time,
        output_time=usage.completion_time,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_time=usage.total_time,
        model_name="llama3-70b-8192",
    )

    return statistics_to_return, completion.choices[0].message.content

def generate_section(prompt: str, additional_instructions: str):
    stream = st.session_state.groq.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are an expert writer. Generate a long, comprehensive, structured chapter for the section provided. If additional instructions are provided, consider them very important. Only output the content.",
            },
            {
                "role": "user",
                "content": f"Generate a long, comprehensive, structured chapter. Use the following section and important instructions:\n\n<section_title>{prompt}</section_title>\n\n<additional_instructions>{additional_instructions}</additional_instructions>",
            },
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in stream:
        tokens = chunk.choices[0].delta.content
        if tokens:
            yield tokens
        if x_groq := chunk.x_groq:
            if not x_groq.usage:
                continue
            usage = x_groq.usage
            statistics_to_return = GenerationStatistics(
                input_time=usage.prompt_time,
                output_time=usage.completion_time,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_time=usage.total_time,
                model_name="llama3-8b-8192",
            )
            yield statistics_to_return

def extract_text_from_pdf(file):
    """
    Extracts text from the provided PDF file.
    """
    document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text("text")
    return text

def generate_research_citations(extracted_texts):
    """
    Generate proper citations for the extracted texts.
    """
    citations = []
    for text in extracted_texts:
        citation = st.session_state.groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "Generate proper citations in APA format for the given extracted text from a research paper."
                },
                {
                    "role": "user",
                    "content": f"Generate a citation for the following text:\n\n{text}"
                }
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            stream=False,
            stop=None,
        )
        citations.append(citation.choices[0].message.content.strip())
    return citations

def preprocess_texts(texts):
    """
    Preprocess texts for indexing.
    """
    preprocessed_texts = []
    for text in texts:
        # Split text into smaller chunks for indexing
        chunks = text.split("\n\n")
        preprocessed_texts.extend(chunks)
    return preprocessed_texts

def index_texts(texts):
    """
    Index texts using FAISS.
    """
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectorizer

def retrieve_passages(query, index, vectorizer, texts, top_k=5):
    """
    Retrieve top-k passages relevant to the query.
    """
    query_vector = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vector, top_k)
    retrieved_passages = [texts[i] for i in indices[0]]
    return retrieved_passages

# Initialize
if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = False

if "button_text" not in st.session_state:
    st.session_state.button_text = "Generate"

if "statistics_text" not in st.session_state:
    st.session_state.statistics_text = ""

if 'paper_title' not in st.session_state:
    st.session_state.paper_title = ""

if 'uploaded_pdfs' not in st.session_state:
    st.session_state.uploaded_pdfs = []

if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = []

if 'citations' not in st.session_state:
    st.session_state.citations = []

if 'index' not in st.session_state:
    st.session_state.index = None

if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

if 'preprocessed_texts' not in st.session_state:
    st.session_state.preprocessed_texts = []

st.write(
    """
# GroqPaper: Write full research papers using llama3 (8b and 70b) on Groq
"""
)

def disable():
    st.session_state.button_disabled = True

def enable():
    st.session_state.button_disabled = False

def empty_st():
    st.empty()

try:
    if st.button("End Generation and Download Paper"):
        if "paper" in st.session_state:
            # Create markdown file
            markdown_file = create_markdown_file(
                st.session_state.paper.get_markdown_content()
            )
            st.download_button(
                label="Download Text",
                data=markdown_file,
                file_name=f'{st.session_state.paper_title}.txt',
                mime='text/plain'
            )

            # Create pdf file (styled)
            pdf_file_path = create_pdf_file(st.session_state.paper.get_markdown_content())
            if pdf_file_path:
                with open(pdf_file_path, "rb") as pdf_file:
                    pdf_data = pdf_file.read()
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name=f'{st.session_state.paper_title}.pdf',
                    mime='application/pdf'
                )
            else:
                st.error("Failed to generate the PDF file.")
        else:
            raise ValueError("Please generate content first before downloading the paper.")

    with st.form("groqform"):
        if not GROQ_API_KEY:
            groq_input_key = st.text_input(
                "Enter your Groq API Key (gsk_yA...):", "", type="password"
            )

        topic_text = st.text_input(
            "What do you want the research paper to be about?",
            value="",
            help="Enter the main topic or title of your research paper",
        )

        additional_instructions = st.text_area(
            "Additional Instructions (optional)",
            help="Provide any specific guidelines or preferences for the research paper's content",
            placeholder="E.g., 'Focus on beginner-friendly content', 'Include case studies', etc.",
            value="",
        )

        # Upload PDFs
        uploaded_pdfs = st.file_uploader(
            "Upload related research PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload multiple PDFs that you want to extract information from and cite in the research paper",
        )

        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                if pdf not in st.session_state.uploaded_pdfs:
                    st.session_state.uploaded_pdfs.append(pdf)
                    extracted_text = extract_text_from_pdf(pdf)
                    st.session_state.extracted_texts.append(extracted_text)
                    st.success(f"Extracted text from {pdf.name}")

            st.session_state.preprocessed_texts = preprocess_texts(st.session_state.extracted_texts)
            st.session_state.index, st.session_state.vectorizer = index_texts(st.session_state.preprocessed_texts)

        # Generate button
        submitted = st.form_submit_button(
            st.session_state.button_text,
            on_click=disable,
            disabled=st.session_state.button_disabled,
        )

        # Statistics
        placeholder = st.empty()

        def display_statistics():
            with placeholder.container():
                if st.session_state.statistics_text:
                    if (
                        "Generating structure in background"
                        not in st.session_state.statistics_text
                    ):
                        st.markdown(
                            st.session_state.statistics_text + "\n\n---\n"
                        )  # Format with line if showing statistics
                    else:
                        st.markdown(st.session_state.statistics_text)
                else:
                    placeholder.empty()

        if submitted:
            if len(topic_text) < 10:
                raise ValueError("Research paper topic must be at least 10 characters long")

            st.session_state.button_disabled = True
            st.session_state.statistics_text = "Generating research paper title and structure in background...."
            display_statistics()

            if not GROQ_API_KEY:
                st.session_state.groq = Groq(api_key=groq_input_key)

            large_model_generation_statistics, paper_structure = generate_paper_structure(
                topic_text
            )
            # Generate AI research paper title
            st.session_state.paper_title = generate_paper_title(topic_text)
            st.write(f"## {st.session_state.paper_title}")

            large_model_generation_statistics, paper_structure = generate_paper_structure(topic_text)

            total_generation_statistics = GenerationStatistics(
                model_name="llama3-8b-8192"
            )

            try:
                paper_structure_json = json.loads(paper_structure)
                paper = ResearchPaper(st.session_state.paper_title, paper_structure_json)
                
                if 'paper' not in st.session_state:
                    st.session_state.paper = paper

                # Print the paper structure to the terminal to show structure
                print(json.dumps(paper_structure_json, indent=2))

                st.session_state.paper.display_structure()
    
                def stream_section_content(sections):
                    for title, content in sections.items():
                        if isinstance(content, str):
                            content_stream = generate_section(
                                title + ": " + content, additional_instructions
                            )
                            for chunk in content_stream:
                                # Check if GenerationStatistics data is returned instead of str tokens
                                chunk_data = chunk
                                if type(chunk_data) == GenerationStatistics:
                                    total_generation_statistics.add(chunk_data)

                                    st.session_state.statistics_text = str(
                                        total_generation_statistics
                                    )
                                    display_statistics()

                                elif chunk != None:
                                    st.session_state.paper.update_content(title, chunk)
                        elif isinstance(content, dict):
                            stream_section_content(content)

                stream_section_content(paper_structure_json)

                # Append extracted texts with citations
                citations = generate_research_citations(st.session_state.extracted_texts)
                st.session_state.citations = citations
                st.session_state.paper.update_content(
                    "References", "\n\n".join(st.session_state.citations)
                )

            except json.JSONDecodeError:
                st.error("Failed to decode the research paper structure. Please try again.")

            enable()

except Exception as e:
    st.session_state.button_disabled = False
    st.error(e)

    if st.button("Clear"):
        st.rerun()
