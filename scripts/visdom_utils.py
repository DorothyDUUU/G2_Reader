import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import traceback
import logging
from io import BytesIO
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from llm_client import get_default_model, get_openai_client

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using OCR if needed.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of text from each page
    """
    try:
        # First try regular PDF extraction
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file, strict=False)
            pages = [page.extract_text() for page in reader.pages]
            
        # If any page has no text, use OCR
        # if any(not page.strip() for page in pages):
        #     logger.info(f"Using OCR for {pdf_path} as some pages have no text")
        #     pages = []
        #     pdf_images = convert_from_path(pdf_path)
        #     for page_num, page_img in enumerate(pdf_images):
        #         text = pytesseract.image_to_string(page_img)
        #         pages.append(f"--- Page {page_num + 1} ---\n{text}\n")
        
        return pages
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        traceback.print_exc()
        return []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)

def split_text(text):
    """
    Split text into chunks.
    
    Args:
        text (str): Text to split
        
    Returns:
        list: List of text chunks
    """
    return text_splitter.split_text(text)


def extract_images_from_pdf(pdf_path):
    """
    Extract images from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of images
    """
    return convert_from_path(pdf_path)


def encode_image(pil_image):
    """Encode a PIL image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")    
    return img_str

#pages = extract_text_from_pdf("/home/ubuntu/songjunru/VisDoM/spiqa/docs/1603.00286v5.pdf")
#chunks = split_text(pages[1])
#images = extract_images_from_pdf("/home/ubuntu/songjunru/VisDoM/spiqa/docs/1603.00286v5.pdf")
#encoded_images = [encode_image(image) for image in images]

client = get_openai_client()
DEFAULT_MODEL = get_default_model()

def analyze_content(content, visual, model=DEFAULT_MODEL):
    """Analyze content to extract keywords, context, and other metadata"""

    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                            },
                            "required": ["keywords", "context", "tags"],
                            "additionalProperties": False
                        },
                        "strict": True
                }
            }

    if visual==False:
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
                temperature=0.7,
                max_tokens=2048,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error analyzing content (response_format not supported?): {str(e)}")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You must respond with a JSON object."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e2:
                print(f"Error analyzing content: {str(e2)}")
                return {
                    "keywords": [],
                    "context": "General",
                    "category": "Uncategorized",
                    "tags": [],
                }

    else:
        prompt = """Generate a structured analysis of the provided image (which is a page of a scientific paper) by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // A summary of:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                    // - Detailed descriptions of visual elements within the image, such as tables and charts (including the index, such as Table 1, Figure 1, etc.)
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }
            """

        messages = [
            {"role": "system", "content": "You must respond with a JSON object."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content}"}},
                ],
            },
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=0.7,
                max_tokens=2048,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error analyzing content (response_format not supported?): {str(e)}")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e2:
                print(f"Error analyzing content: {str(e2)}")
                return {
                    "keywords": [],
                    "context": "General",
                    "category": "Uncategorized",
                    "tags": [],
                }
