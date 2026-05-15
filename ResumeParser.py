import re
import json
from pathlib import Path

import pypdf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for pkg in ('punkt', 'stopwords', 'wordnet', 'punkt_tab'):
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(nltk_stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

SKILLS_DB = {
    'programming': {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby',
        'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'php',
    },
    'web': {
        'html', 'css', 'react.js', 'angular', 'vue', 'node.js', 'django',
        'flask', 'fastapi', 'spring', 'express', 'graphql', 'rest api',
        'tailwind', 'sass', 'webpack', 'react native', 'flutter', 'express.js',
    },
    'data_ml': {
        'machine learning', 'deep learning', 'artificial intelligence',
        'data science', 'data analysis', 'data engineering', 'nlp',
        'computer vision', 'pandas', 'numpy', 'scipy', 'matplotlib',
        'seaborn', 'tensorflow', 'keras', 'pytorch', 'scikit-learn',
        'xgboost', 'spark', 'hadoop', 'tableau', 'power bi',
    },
    'databases': {
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
        'cassandra', 'dynamodb', 'nosql', 'sqlite', 'oracle',
    },
    'devops_cloud': {
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
        'ansible', 'jenkins', 'ci/cd', 'linux', 'bash', 'git', 'github',
        'gitlab', 'bitbucket', 'prometheus', 'grafana',
    },
    'soft_skills': {
        'communication', 'leadership', 'project management', 'agile',
        'scrum', 'kanban', 'problem solving', 'critical thinking',
        'teamwork', 'mentoring', 'stakeholder management', 'presentation',
    },
}

ALL_SKILLS     = {s for skills in SKILLS_DB.values() for s in skills}
SKILL_CATEGORY = {s: cat for cat, skills in SKILLS_DB.items() for s in skills}


ALIASES: dict[str, str] = {
    'react':          'react.js',
    'reactjs':        'react.js',
    'react js':       'react.js',
    'nodejs':         'node.js',
    'node js':        'node.js',
    'node':           'node.js',
    'expressjs':      'express.js',
    'express js':     'express.js',
    'vue.js':         'vue',
    'vuejs':          'vue',
    'angular.js':     'angular',
    'angularjs':      'angular',
    'rest':           'rest api',
    'restful':        'rest api',
    'restful api':    'rest api',
    'postgres':       'postgresql',
    'mongo':          'mongodb',
    'elastic':        'elasticsearch',
    'dynamo':         'dynamodb',
    'sqlite3':        'sqlite',
    'sklearn':        'scikit-learn',
    'scikit learn':   'scikit-learn',
    'tf':             'tensorflow',
    'pytorch':        'pytorch',
    'ai':             'artificial intelligence',
    'ml':             'machine learning',
    'nlp':            'nlp',
    'cv':             'computer vision',
    'amazon web services': 'aws',
    'google cloud':   'gcp',
    'google cloud platform': 'gcp',
    'k8s':            'kubernetes',
    'ci cd':          'ci/cd',
    'cicd':           'ci/cd',
    'github actions': 'ci/cd',
    'c plus plus':    'c++',
    'csharp':         'c#',
    'c sharp':        'c#',
    'golang':         'go',
    'js':             'javascript',
    'ts':             'typescript',
    'problem-solving': 'problem solving',
    'critical-thinking': 'critical thinking',
}

def read_resume(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume not found: {file_path}")
    if path.suffix.lower() == '.pdf':
        reader = pypdf.PdfReader(str(path))
        return '\n'.join(page.extract_text() or '' for page in reader.pages)
    return path.read_text(encoding='utf-8')


def read_job_description(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Job description not found: {file_path}")
    return path.read_text(encoding='utf-8')

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.#+]', ' ', text)   # keep . # + for c#, c++, node.js
    tokens = word_tokenize(text)
    return ' '.join(
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 1
    )


def _apply_aliases(text: str) -> str:
    for alias, canonical in sorted(ALIASES.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'\b' + re.escape(alias) + r'\b', canonical, text)
    return text


def extract_skills(processed_text: str, original_text: str) -> set[str]:
    sources = [
        _apply_aliases(processed_text),
        _apply_aliases(original_text.lower()),
    ]
    found: set[str] = set()
    for skill in ALL_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if any(re.search(pattern, src) for src in sources):
            found.add(skill)
    return found


def skill_recall(resume_skills: set, job_skills: set) -> float:
    """Fraction of required job skills present in the resume."""
    if not job_skills:
        return 1.0          # no skills required → perfect match by default
    return len(resume_skills & job_skills) / len(job_skills)


def tfidf_similarity(text_a: str, text_b: str) -> float:
    try:
        vect   = TfidfVectorizer()
        matrix = vect.fit_transform([text_a, text_b])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except ValueError:
        return 0.0


def combined_score(recall: float, tfidf: float,
                   w_recall: float = 0.65, w_tfidf: float = 0.35) -> float:
    return w_recall * recall + w_tfidf * tfidf


def skills_by_category(skills: set) -> dict[str, set]:
    breakdown: dict[str, set] = {cat: set() for cat in SKILLS_DB}
    for skill in skills:
        cat = SKILL_CATEGORY.get(skill)
        if cat:
            breakdown[cat].add(skill)
    return {k: v for k, v in breakdown.items() if v}


def analyze_resume(resume_path: str, job_description_path: str) -> dict:
    resume_raw = read_resume(resume_path)
    job_raw    = read_job_description(job_description_path)

    resume_proc = preprocess_text(resume_raw)
    job_proc    = preprocess_text(job_raw)

    resume_skills = extract_skills(resume_proc, resume_raw)
    job_skills    = extract_skills(job_proc, job_raw)

    recall = skill_recall(resume_skills, job_skills)
    tfidf  = tfidf_similarity(resume_proc, job_proc)
    final  = combined_score(recall, tfidf)

    matched = resume_skills & job_skills
    missing = job_skills    - resume_skills
    extra   = resume_skills - job_skills

    return {
        'scores': {
            'final':  round(final,  4),
            'recall': round(recall, 4),
            'tfidf':  round(tfidf,  4),
        },
        'skills': {
            'resume':  resume_skills,
            'job':     job_skills,
            'matched': matched,
            'missing': missing,
            'extra':   extra,
        },
        'categories': {
            'resume':  skills_by_category(resume_skills),
            'job':     skills_by_category(job_skills),
            'missing': skills_by_category(missing),
        },
    }


def print_report(result: dict) -> None:
    scores = result['scores']
    skills = result['skills']
    cats   = result['categories']
    print(f"{'  RESUME ANALYSIS REPORT  ':^52}")
    print("\nMATCH SCORES")
    print(f"  Final (blended)   : {scores['final']:.1%}")
    print(f"  Skill Recall      : {scores['recall']:.1%}")
    print(f"  TF-IDF Cosine     : {scores['tfidf']:.1%} ")

    pct   = scores['final']
    label = ('Excellent ' if pct >= 0.75 else
             'Good      ' if pct >= 0.55 else
             'Fair      '  if pct >= 0.35 else
             'Needs Work ')
    print(f"\nAssessment        : {label}")

    print(f"\nMATCHED SKILLS ({len(skills['matched'])})")
    print(f"  {', '.join(sorted(skills['matched'])) or 'None'}")

    print(f"\nMISSING SKILLS ({len(skills['missing'])})  ← add these to your resume")
    print(f"  {', '.join(sorted(skills['missing'])) or 'None'}")

    print(f"\nEXTRA SKILLS ({len(skills['extra'])})  ← you have but JD didn't ask")
    print(f"  {', '.join(sorted(skills['extra'])) or 'None'}")

    if cats['missing']:
        print(f"\nMISSING BY CATEGORY")
        for cat, s in sorted(cats['missing'].items()):
            print(f"  {cat:<22}: {', '.join(sorted(s))}")

if __name__ == '__main__':
    resume_path   = input("Enter the path to your resume file (PDF or TXT): ").strip()
    job_desc_path = input("Enter the path to the job description file (TXT): ").strip()

    try:
        result = analyze_resume(resume_path, job_desc_path)
        print_report(result)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        raise