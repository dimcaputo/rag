from deepeval.test_case import LLMTestCase

test_cases = [
    LLMTestCase(
        input="What programming languages and frameworks does Dimitri Caputo have experience with?",
        actual_output="",
        expected_output="Python, Pandas, Numpy, Matplotlib, Scikit-learn, Keras, Tensorflow, PyTorch, Docker, PySpark, Ansible",
        retrieval_context=""
    ),
    LLMTestCase(
        input="Which machine learning libraries has Dimitri used for graph neural networks?",
        actual_output="",
        expected_output="Scikit-learn, Keras, Tensorflow, PyTorch, PyTorch-Geometric, and DGL",
        retrieval_context=""
    ),
    LLMTestCase(
        input="What is Dimitri Caputo’s educational background in chemistry?",
        actual_output="",
        expected_output="PhD in Organic Chemistry from University of Oxford and Diplôme d'Ingénieur Chimiste from ENSCM Montpellier",
        retrieval_context=""
    ),
    LLMTestCase(
        input="Which languages does Dimitri speak and at what proficiency?",
        actual_output="",
        expected_output="English (proficient), French (mother tongue), Italian (beginner), German (beginner)",
        retrieval_context=""
    ),
    LLMTestCase(
        input="What roles did Dimitri hold as a Medicinal Chemist?",
        actual_output="",
        expected_output="Medicinal Chemist at Evotec SAS (Toulouse, FR) and Sygnature Discovery (Nottingham, UK)",
        retrieval_context=""
    ),
    LLMTestCase(
        input="What leadership or management responsibilities has Dimitri undertaken?",
        actual_output="",
        expected_output="Supervised up to 5 chemists, line-managed two lab technicians, supervised 4 master's students",
        retrieval_context=""
    ),
    LLMTestCase(
        input="Which cities or locations does Dimitri prefer for work?",
        actual_output="",
        expected_output="Grenoble, Chambéry, Lyon, Paris, Montpellier, Toulouse, or full remote",
        retrieval_context=""
    ),
    LLMTestCase(
        input="What online courses has Dimitri completed related to data science?",
        actual_output="",
        expected_output="Applied Data Science in Python (University of Michigan, Coursera) and Python 3 Programming (University of Michigan, Coursera)",
        retrieval_context=""
    ),
    LLMTestCase(
        input="What kind of projects did Dimitri work on during his apprenticeship at Ygreky?",
        actual_output="",
        expected_output="Implemented graph neural networks for automatic detection of software vulnerabilities",
        retrieval_context=""
    ),
    LLMTestCase(
        input="What teamwork and collaboration tools is Dimitri familiar with?",
        actual_output="",
        expected_output="VSCode, Jupyter, Kate, Git, GitHub, GitLab",
        retrieval_context=""
    ),
]
