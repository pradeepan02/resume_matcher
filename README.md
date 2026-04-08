# Resume Analyzer & Job Matcher

A smart Python tool that analyzes resumes and matches them with job descriptions to help fresh graduates and job seekers tailor their applications and improve their chances of getting shortlisted.

---

##  Overview

This Resume Analyzer uses Natural Language Processing (NLP) and machine learning techniques to:
- Extract key skills and competencies from resumes
- Compare resume skills with job requirements
- Generate a comprehensive match score
- Identify missing skills that should be added to your resume
- Categorize skills by expertise area (Programming, Web, Data/ML, Databases, DevOps/Cloud, Soft Skills)

---

## Key Features

1. **Multi-Format Resume Support**: Read resumes in PDF or plain text format
2. **Advanced Text Processing**: 
   - Lemmatization and tokenization using NLTK
   - Stop word removal for focused analysis
   - Case-insensitive matching
3. **Intelligent Skill Matching**:
   - Recognizes 80+ technical and soft skills across 6 categories
   - Supports skill synonyms and variations (e.g., "React.js", "node.js", "rest api")
4. **Multiple Scoring Metrics**:
   - **Jaccard Similarity**: Skill set overlap analysis
   - **TF-IDF Cosine Similarity**: Content relevance scoring
   - **Combined Score**: Blended metric (60% Jaccard + 40% TF-IDF)
5. **Actionable Insights**:
   - Shows matched skills
   - Lists missing skills by category
   - Displays extra skills you have
   - Provides assessment labels (Excellent, Good, Fair, Needs Work)

---

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

---


## Usage

### Running the Tool

1. **Execute the script** from the PYTHON directory:
   ```bash
   python ResumeParser.py
   ```

2. **Enter file paths when prompted**:
   ```
   Enter the path to your resume file (PDF or TXT): path/to/your/resume.pdf
   Enter the path to the job description file (TXT): path/to/job_description.txt
   ```

3. **View the comprehensive report** with match scores and recommendations

### Resume Format

**Supported Formats:**
- PDF files (.pdf)
- Text files (.txt)
- Any plain text file

**Requirements:**
- Readable text content
- Should include sections like: Experience, Skills, Education, Projects
- Use clear skill names from the supported list

### Job Description Format

**Supported Format:**
- Text file (.txt)

**Requirements:**
- Plain text format
- Should clearly state required and preferred skills
- Can include job title, qualifications, responsibilities, etc.

---
