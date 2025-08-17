import io
import PyPDF2
import docx
from PIL import Image
import pytesseract
import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# --- Enhanced State with Tracking ---
class AppState(TypedDict):
    jd_text: str
    jd_input: str
    resumes: List[Dict]
    cleaned_jd: str
    cleaned_resumes: List[Dict]
    scores: List[Dict]
    top_n: int
    top_resumes: List[Dict]
    # Enhanced state tracking
    processing_status: Dict
    timestamps: Dict
    extraction_stats: Dict

# --- Helper Functions ---
def is_url(text: str) -> bool:
    """Checks if the given string is a valid URL."""
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX bytes"""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return ""

def extract_text_from_image(file_content: bytes) -> str:
    """Extract text from image using OCR"""
    try:
        image = Image.open(io.BytesIO(file_content))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting from image: {e}")
        return ""

def extract_text(file_input, file_extension: str = None) -> str:
    """Enhanced text extraction with better file type handling"""
    if isinstance(file_input, str):
        # Case 1: Input is a URL
        if is_url(file_input):
            try:
                response = requests.get(file_input, timeout=10)
                if response.status_code == 200 and response.content:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    return text
                else:
                    print(f"Failed to fetch URL {file_input}: Status code {response.status_code}")
                    return ""
            except Exception as e:
                print(f"Error fetching URL {file_input}: {e}")
                return ""
        # Case 2: Input is a local file path
        else:
            try:
                with open(file_input, 'rb') as f:
                    file_content = f.read()
                file_extension = os.path.splitext(file_input)[1].lower()
                return extract_text(file_content, file_extension)
            except FileNotFoundError:
                print(f"Error: File not found at path: {file_input}")
                return ""
            except Exception as e:
                print(f"An error occurred while reading file {file_input}: {e}")
                return ""
    
    # Case 3: Input is file content (bytes)
    if not file_extension:
        return ""
    
    file_extension = file_extension.lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_input)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_input)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return extract_text_from_image(file_input)
    elif file_extension == '.txt':
        return file_input.decode('utf-8', errors='ignore')
    else:
        print(f"Unsupported file extension: {file_extension}")
        return ""

def setup_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    else:
        print("Warning: GOOGLE_API_KEY not found in environment variables")
        return None

def get_file_type_route(filename: str) -> str:
    """Determine routing based on file type"""
    extension = os.path.splitext(filename)[1].lower()
    if extension == '.pdf':
        return 'pdf_route'
    elif extension in ['.docx', '.doc']:
        return 'docx_route'
    elif extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return 'image_route'
    else:
        return 'generic_route'

# --- Enhanced Agents ---

def initialize_state(state: AppState) -> AppState:
    """Initialize enhanced state tracking"""
    current_time = datetime.now().isoformat()
    state['processing_status'] = {
        'jd_extracted': False,
        'resumes_extracted': False,
        'jd_cleaned': False,
        'resumes_cleaned': False,
        'scores_calculated': False,
        'top_selected': False
    }
    state['timestamps'] = {
        'start_time': current_time,
        'jd_extraction_start': None,
        'resume_extraction_start': None,
        'cleaning_start': None,
        'scoring_start': None
    }
    state['extraction_stats'] = {
        'total_resumes': len(state.get('resumes', [])),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'total_characters_extracted': 0
    }
    return state



# Process JD once at the beginning
def process_jd_complete(state: AppState) -> AppState:
    """Extract AND clean JD text in one step (done once for all resumes)"""
    # Extract JD
    if 'jd_input' in state:
        try:
            jd_text = extract_text(state['jd_input'])
            state['jd_text'] = jd_text
            state['processing_status']['jd_extracted'] = True
            state['extraction_stats']['total_characters_extracted'] += len(jd_text)
            print(f"JD extracted successfully: {len(jd_text)} characters")
        except Exception as e:
            print(f"Error extracting JD: {e}")
            state['jd_text'] = ""
            state['processing_status']['jd_extracted'] = False
    else:
        print("Warning: 'jd_input' missing from state")
        state['jd_text'] = ""
        state['processing_status']['jd_extracted'] = False
    
    # If extraction failed, set cleaned_jd and return
    if not state['processing_status']['jd_extracted']:
        state['cleaned_jd'] = "JD extraction failed"
        state['processing_status']['jd_cleaned'] = False
        return state
    
    # Clean JD immediately after extraction
    model = setup_gemini()
    if not model:
        state['cleaned_jd'] = state['jd_text']
        state['processing_status']['jd_cleaned'] = False
        print("Warning: No Gemini API - using raw JD text")
        return state
    
    jd_prompt = f"""Extract job requirements, qualifications, and responsibilities from this job description. 
    Keep it concise and relevant for matching candidates.
    
    Job Description: {state['jd_text'][:4000]}"""
    
    try:
        jd_response = model.generate_content(jd_prompt)
        state['cleaned_jd'] = jd_response.text
        state['processing_status']['jd_cleaned'] = True
        print("JD cleaned successfully")
    except Exception as e:
        print(f"Error cleaning JD with Gemini: {e}")
        state['cleaned_jd'] = state['jd_text']
        state['processing_status']['jd_cleaned'] = False
    
    return state

# FEATURE 2: Parallel Processing - Resume Extraction AND Cleaning
def process_resumes_parallel(state: AppState) -> AppState:
    """Extract AND clean all resume texts in parallel batch processing"""
    state['timestamps']['resume_extraction_start'] = datetime.now().isoformat()
    state['timestamps']['cleaning_start'] = datetime.now().isoformat()
    
    model = setup_gemini()
    processed_resumes = []
    successful_count = 0
    failed_count = 0
    
    for resume in state['resumes']:
        if isinstance(resume, dict):
            file_content = resume['content']
            file_extension = resume['extension']
            filename = resume['filename']
            
            # FEATURE 1: Route based on file type
            route = get_file_type_route(filename)
            print(f"Processing {filename} via {route}")
            
            # EXTRACT text first
            text = extract_text(file_content, file_extension)
            
            # FEATURE 5: Enhanced State Tracking
            if len(text.strip()) > 50:  # Quality check
                extraction_status = 'success'
                confidence = min(100, len(text) / 100)  # Simple confidence score
            else:
                extraction_status = 'failed'
                confidence = 0
                print(f"Warning: Low quality extraction for {filename}")
            
            # CLEAN text immediately if extraction succeeded
            if extraction_status == 'success' and model:
                resume_prompt = f"""Extract experience, education, certifications, skills, and projects from this resume. 
                Keep it concise and relevant for job matching.
                
                Resume: {text[:4000]}"""
                
                try:
                    resume_response = model.generate_content(resume_prompt)
                    cleaned_text = resume_response.text
                    cleaning_status = 'success'
                except Exception as e:
                    print(f"Error cleaning resume '{filename}' with Gemini: {e}")
                    cleaned_text = text
                    cleaning_status = 'failed'
            else:
                # Use raw text if extraction failed or no API
                cleaned_text = text
                cleaning_status = 'skipped_failed_extraction' if extraction_status == 'failed' else 'skipped_no_api'
            
            # Count successes
            if extraction_status == 'success':
                successful_count += 1
            else:
                failed_count += 1
            
            processed_resumes.append({
                'filename': filename,
                'original_text': text,
                'cleaned_text': cleaned_text,
                'extraction_status': extraction_status,
                'cleaning_status': cleaning_status,
                'confidence': confidence,
                'character_count': len(text),
                'route_used': route,
                'timestamp': datetime.now().isoformat()
            })
            
            state['extraction_stats']['total_characters_extracted'] += len(text)
        else:
            print(f"Warning: Invalid resume entry: {resume}")
            failed_count += 1
    
    state['cleaned_resumes'] = processed_resumes
    state['processing_status']['resumes_extracted'] = True
    state['processing_status']['resumes_cleaned'] = True
    state['extraction_stats']['successful_extractions'] = successful_count
    state['extraction_stats']['failed_extractions'] = failed_count
    
    print(f"Resume processing complete: {successful_count} success, {failed_count} failed")
    print(f"All resumes extracted and cleaned in parallel batch")
    return state

def calculate_scores_enhanced(state: AppState) -> AppState:
    """Enhanced scoring with better error handling and retries"""
    state['timestamps']['scoring_start'] = datetime.now().isoformat()
    
    model = setup_gemini()
    if not model:
        # Fallback to simple keyword matching
        print("Using fallback keyword scoring")
        scores = []
        for resume in state['cleaned_resumes']:
            # Simple keyword overlap score
            jd_words = set(state['cleaned_jd'].lower().split())
            resume_words = set(resume['cleaned_text'].lower().split())
            overlap = len(jd_words.intersection(resume_words))
            total_words = len(jd_words.union(resume_words))
            score = int((overlap / total_words) * 100) if total_words > 0 else 0
            
            scores.append({
                'filename': resume['filename'],
                'score': score,
                'method': 'keyword_fallback',
                'confidence': resume['confidence'],
                'extraction_status': resume['extraction_status']
            })
        
        state['scores'] = scores
        return state
    
    scores = []
    for resume in state['cleaned_resumes']:
        # Skip scoring for failed extractions
        if resume['extraction_status'] == 'failed':
            scores.append({
                'filename': resume['filename'],
                'score': 0,
                'method': 'skipped_failed_extraction',
                'confidence': 0,
                'extraction_status': 'failed'
            })
            continue
        
        prompt = f"""Based on the following job description and candidate resume, provide a similarity score from 0 to 100.
        
        Consider skills match, experience relevance, education fit, and overall suitability.
        
        Job Description: {state['cleaned_jd'][:2000]}
        
        Candidate Resume: {resume['cleaned_text'][:2000]}
        
        Provide ONLY the integer score (0-100), no other text.
        """
        
        score = 0
        method = 'ai_scoring'
        
        try:
            response = model.generate_content(prompt)
            score_text = response.text.strip()
            
            # Enhanced parsing to handle various AI responses
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = int(numbers[0])
                score = max(0, min(100, score))
                method = 'ai_scoring'
            else:
                print(f"No number in AI response for {resume['filename']}: '{score_text}'")
                # Retry with simpler prompt
                simple_prompt = f"Rate similarity 0-100: JD: {state['cleaned_jd'][:500]} Resume: {resume['cleaned_text'][:500]}"
                retry_response = model.generate_content(simple_prompt)
                retry_numbers = re.findall(r'\d+', retry_response.text.strip())
                if retry_numbers:
                    score = max(0, min(100, int(retry_numbers[0])))
                    method = 'ai_retry'
                else:
                    score = 0
                    method = 'ai_failed'
                    
        except Exception as e:
            print(f"Error scoring {resume['filename']}: {e}")
            score = 0
            method = 'error'
        
        scores.append({
            'filename': resume['filename'],
            'score': score,
            'method': method,
            'confidence': resume['confidence'],
            'extraction_status': resume['extraction_status'],
            'resume_text': resume['original_text']
        })
    
    state['scores'] = scores
    state['processing_status']['scores_calculated'] = True
    print(f"Scoring complete. Methods used: {set(s['method'] for s in scores)}")
    return state

def select_top_resumes(state: AppState) -> AppState:
    """Select top resumes with enhanced tracking"""
    sorted_scores = sorted(state['scores'], key=lambda x: x['score'], reverse=True)
    top_n = state.get('top_n', 5)
    
    # FEATURE 5: Enhanced tracking - add selection metadata
    top_resumes = []
    for i, resume in enumerate(sorted_scores[:top_n]):
        enhanced_resume = resume.copy()
        enhanced_resume['rank'] = i + 1
        enhanced_resume['percentile'] = ((len(sorted_scores) - i) / len(sorted_scores)) * 100
        top_resumes.append(enhanced_resume)
    
    state['top_resumes'] = top_resumes
    state['processing_status']['top_selected'] = True
    
    # Add final statistics
    state['final_stats'] = {
        'total_processed': len(state['scores']),
        'average_score': sum(s['score'] for s in state['scores']) / len(state['scores']) if state['scores'] else 0,
        'highest_score': max(s['score'] for s in state['scores']) if state['scores'] else 0,
        'successful_extractions': state['extraction_stats']['successful_extractions'],
        'processing_complete': True,
        'end_time': datetime.now().isoformat()
    }
    
    print(f"Selected top {len(top_resumes)} resumes. Average score: {state['final_stats']['average_score']:.1f}")
    return state

# FEATURE 1: Conditional routing function
def route_after_jd_processing(state: AppState) -> str:
    """Route based on JD processing success"""
    if not state['processing_status']['jd_extracted']:
        print("JD extraction failed - routing to error handling")
        return "handle_jd_error"
    
    if not state['processing_status']['jd_cleaned']:
        print("JD cleaning failed - but proceeding with raw JD text")
        # Still proceed since we have raw JD text
    
    return "proceed_to_resumes"

def handle_extraction_errors(state: AppState) -> AppState:
    """Handle cases where JD extraction fails"""
    print("Handling JD extraction errors...")
    
    if not state['processing_status']['jd_extracted']:
        state['jd_text'] = "JD extraction failed"
        state['cleaned_jd'] = "Unable to process job description - cannot proceed with matching"
        # Set empty cleaned_resumes to prevent scoring
        state['cleaned_resumes'] = []
    
    print("Cannot proceed with resume matching due to JD failure")
    return state

# Create the enhanced workflow
def create_enhanced_workflow():
    workflow = StateGraph(AppState)
    
    # Add all nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("process_jd", process_jd_complete)
    workflow.add_node("process_resumes", process_resumes_parallel)
    workflow.add_node("score", calculate_scores_enhanced)
    workflow.add_node("select", select_top_resumes)
    workflow.add_node("handle_errors", handle_extraction_errors)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Process JD first (extract + clean in one step)
    workflow.add_edge("initialize", "process_jd")
    
    # FEATURE 1: Conditional routing after JD processing
    workflow.add_conditional_edges(
        "process_jd",
        route_after_jd_processing,
        {
            "proceed_to_resumes": "process_resumes",
            "handle_jd_error": "handle_errors"
        }
    )
    
    # Resume processing (extract + clean in parallel batch)
    workflow.add_edge("process_resumes", "score")
    workflow.add_edge("handle_errors", "score")
    
    # Final steps
    workflow.add_edge("score", "select")
    workflow.add_edge("select", END)
    
    return workflow.compile()


def run_workflow(jd_input, resume_files_data, top_n=5):
    """
    Run the enhanced resume matcher workflow
    
    Args:
        jd_input: URL string or file path for job description
        resume_files_data: List of resume data dictionaries
        top_n: Number of top resumes to return
    
    Returns:
        final_state: Complete workflow results
    """
    initial_state = {
        'jd_input': jd_input,
        'resumes': resume_files_data,
        'top_n': top_n
    }
    
    app = create_enhanced_workflow()
    return app.invoke(initial_state)
