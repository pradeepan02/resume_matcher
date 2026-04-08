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

STOP_WORDS  = set(nltk_stopwords.words('english'))
LEMMATIZER  = WordNetLemmatizer()
SKILLS_DB = {
    'programming': {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby',
        'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'php',
    },
    'web': {
        'html', 'css', 'react.js', 'angular', 'vue', 'node.js', 'django',
        'flask', 'fastapi', 'spring', 'express', 'graphql', 'rest api',
        'tailwind', 'sass', 'webpack','react native', 'flutter','express.js',
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
ALL_SKILLS      = {s for skills in SKILLS_DB.values() for s in skills}
SKILL_CATEGORY  = {s:cat for cat, skills in SKILLS_DB.items() for s in skills}
def read_resume(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume not found: {file_path}")

    if path.suffix.lower() == '.pdf':
        reader = pypdf.PdfReader(str(path))
        return '\n'.join(page.extract_text() or '' for page in reader.pages)

    return path.read_text(encoding='utf-8')
def read_job_description(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Job description not found: {file_path}")
    return path.read_text(encoding='utf-8')
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    return ' '.join(
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    )
def extract_skills(processed_text, original_text):
    found   = set()
    sources = [processed_text, original_text.lower()]
    for skill in ALL_SKILLS:
        if any(skill in src for src in sources):
            found.add(skill)
    return found
def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
def tfidf_similarity(text_a, text_b):
    try:
        vect   = TfidfVectorizer()
        matrix = vect.fit_transform([text_a, text_b])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except ValueError:
        return 0.0
def combined_score(jaccard, tfidf,w_jaccard, w_tfidf):
    return w_jaccard * jaccard + w_tfidf * tfidf
def skills_by_category(skills):
    breakdown: dict[str, set] = {cat: set() for cat in SKILLS_DB}
    for skill in skills:
        cat = SKILL_CATEGORY.get(skill)
        if cat:
            breakdown[cat].add(skill)
    return {k: v for k, v in breakdown.items() if v}   
def analyze_resume(resume_path, job_description_path):
 
    resume_raw  = read_resume(resume_path)
    job_raw     = read_job_description(job_description_path)

    resume_proc = preprocess_text(resume_raw)
    job_proc    = preprocess_text(job_raw)

    resume_skills = extract_skills(resume_proc, resume_raw)
    job_skills    = extract_skills(job_proc,    job_raw)


    jaccard = jaccard_similarity(resume_skills, job_skills)
    tfidf   = tfidf_similarity(resume_proc, job_proc)
    final = combined_score(jaccard, tfidf, w_jaccard=0.6, w_tfidf=0.4)


    matched = resume_skills & job_skills
    missing = job_skills    - resume_skills
    extra   = resume_skills - job_skills

    return {
        'scores': {
            'final':   round(final,   4),
            'jaccard': round(jaccard, 4),
            'tfidf':   round(tfidf,   4),
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
def print_report(result):
    scores    = result['scores']
    skills    = result['skills']
    cats      = result['categories']
    print("  RESUME ANALYSIS REPORT")
    print(f"{'MATCH SCORES':}")
    print(f"  Final (blended) : {scores['final']:.1%}")
    print(f"  Skill Jaccard   : {scores['jaccard']:.1%}")
    print(f"  TF-IDF Cosine   : {scores['tfidf']:.1%}")

    pct = scores['final']
    label = ('Excellent' if pct >= 0.75 else
             'Good'      if pct >= 0.55 else
             'Fair'      if pct >= 0.35 else 'Needs Work')
    print(f"Assessment      : {label}")

    print(f"{'MATCHED SKILLS'} ({len(skills['matched'])})")
    print(f"  {', '.join(sorted(skills['matched'])) or 'None'}")

    print(f"{'MISSING SKILLS'} ({len(skills['missing'])})  ← add these to your resume")
    print(f"  {', '.join(sorted(skills['missing'])) or 'None'}")

    print(f"{'EXTRA SKILLS'} ({len(skills['extra'])})  ← you have but JD didn't ask")
    print(f"  {', '.join(sorted(skills['extra'])) or 'None'}")

    if cats['missing']:
        print(f"{'MISSING BY CATEGORY'}")
        for cat, s in sorted(cats['missing'].items()):
            print(f"  {cat:<20}: {', '.join(sorted(s))}")
if __name__ == '__main__':
    
    resume_path = input("Enter the path to your resume file (PDF or TXT): ").strip()
    job_desc_path = input("Enter the path to the job description file (TXT): ").strip()
    
    try:
        result = analyze_resume(resume_path, job_desc_path)
        print_report(result)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        raise
