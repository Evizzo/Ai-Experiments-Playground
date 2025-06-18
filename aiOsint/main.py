from typing import Dict, List, Any
import re
import os
import logging
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import whois21
from dotenv import load_dotenv
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('osint_investigation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

load_dotenv()

class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = "gpt-4.1-nano-2025-04-14"  
    REQUEST_TIMEOUT: int = 10

    @classmethod
    def validate_config(cls) -> None:
        if not cls.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        logger.info("Configuration validated successfully")

class InvestigationRequest(BaseModel):
    domain: str = Field(..., description="Target domain for investigation")

class InvestigationStep(BaseModel):
    thought: str = Field(..., description="The LLM's thought process for this step")
    action: str = Field(..., description="The action taken (function called)")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the function")
    result: Any = Field(default_factory=dict, description="Result of the function call (can be dict, list, or string)")

class InvestigationResponse(BaseModel):
    domain: str = Field(..., description="The domain that was investigated")
    steps: List[InvestigationStep] = Field(default_factory=list, description="List of investigation steps taken")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of findings")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Investigation timestamp")

def getDomainInfo(domain: str) -> Dict[str, Any]:
    """Retrieve domain registration information using multiple sources.
    
    Args:
        domain: The domain name to look up (e.g., 'example.com')
        
    Returns:
        A dictionary containing domain information including registrar, dates, and contact details
    """
    logger.info(f"Fetching domain information for: {domain}")
    result = {
        "domain": domain,
        "registrar": "Unknown",
        "creationDate": "Unknown",
        "expirationDate": "Unknown",
        "emails": [],
        "nameServers": [],
        "ip_address": "Unknown",
        "dns_records": [],
        "ssl_info": {},
        "headers": {}
    }
    
    try:
        whoisClient = whois21.WHOIS(domain)
        if whoisClient.success:
            result.update({
                "registrar": getattr(whoisClient, "registrar", "Unknown"),
                "creationDate": getattr(whoisClient, "creationDate", "Unknown"),
                "expirationDate": getattr(whoisClient, "expirationDate", "Unknown"),
                "emails": getattr(whoisClient, "emails", []),
                "nameServers": getattr(whoisClient, "nameServers", [])
            })
        
        try:
            import socket
            ip_address = socket.gethostbyname(domain)
            result["ip_address"] = ip_address
            
            import dns.resolver
            for record_type in ['A', 'MX', 'NS', 'TXT']:
                try:
                    answers = dns.resolver.resolve(domain, record_type)
                    result["dns_records"].append({
                        "type": record_type,
                        "values": [str(rdata) for rdata in answers]
                    })
                except Exception as e:
                    logger.debug(f"Could not get {record_type} records: {str(e)}")
        except Exception as e:
            logger.debug(f"Could not resolve domain: {str(e)}")
        
        try:
            import ssl
            import socket
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443)) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    result["ssl_info"] = {
                        "issuer": dict(x[0] for x in cert['issuer']),
                        "subject": dict(x[0] for x in cert['subject']),
                        "version": cert['version'],
                        "notBefore": cert['notBefore'],
                        "notAfter": cert['notAfter']
                    }
        except Exception as e:
            logger.debug(f"Could not get SSL info: {str(e)}")
        
        try:
            response = requests.get(f"https://{domain}", timeout=Config.REQUEST_TIMEOUT, verify=False)
            result["headers"] = dict(response.headers)
        except Exception as e:
            logger.debug(f"Could not get HTTP headers: {str(e)}")
        
        logger.info(f"Successfully retrieved domain information for {domain}")
        return result
    except Exception as e:
        logger.error(f"Error in domain lookup for {domain}: {str(e)}")
        return result

def extractEmailAddresses(url: str) -> List[str]:
    """Extract email addresses from a webpage and its linked pages.
    
    Args:
        url: The URL to extract emails from
        
    Returns:
        A list of unique email addresses found on the page and its links
    """
    logger.info(f"Extracting emails from URL: {url}")
    found_emails = set()
    visited_urls = set()
    
    def extract_from_url(current_url: str, depth: int = 0) -> None:
        if depth > 2 or current_url in visited_urls: 
            return
        
        visited_urls.add(current_url)
        try:
            response = requests.get(current_url, timeout=Config.REQUEST_TIMEOUT, verify=False)
            response.raise_for_status()
            
            page_content = BeautifulSoup(response.text, "html.parser")
            text_content = page_content.get_text()
            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            emails = set(re.findall(email_pattern, text_content))
            found_emails.update(emails)
            
            if depth < 2:
                for link in page_content.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('/'):
                        href = f"{url.rstrip('/')}{href}"
                    elif not href.startswith(('http://', 'https://')):
                        continue
                    
                    if href.startswith(url) and href not in visited_urls:
                        extract_from_url(href, depth + 1)
                        
        except Exception as e:
            logger.debug(f"Error processing {current_url}: {str(e)}")
    
    try:
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        
        extract_from_url(url)
        logger.info(f"Found {len(found_emails)} unique email addresses at {url}")
        return list(found_emails)
    except Exception as e:
        logger.error(f"Error extracting emails from {url}: {str(e)}")
        return []

def getBreachInfoFromBreachDirectory(email: str) -> Dict[str, Any]:
    """Check for data breaches using BreachDirectory API.
    
    Args:
        email: The email address to check for breaches
        
    Returns:
        A dictionary containing breach information if found
    """
    logger.info(f"Checking breach information for email: {email}")
    result = {
        "email": email,
        "breaches": [],
        "total_breaches": 0,
        "status": "Not checked"
    }
    
    try:
        url = f"https://breachdirectory.org/api?email={email}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://breachdirectory.org",
            "Referer": "https://breachdirectory.org/"
        }
        response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
        response.raise_for_status()
        breach_data = response.json()
        
        if breach_data.get("success"):
            result["breaches"] = breach_data.get("result", [])
            result["total_breaches"] = len(result["breaches"])
            result["status"] = "Breaches found" if result["total_breaches"] > 0 else "No breaches found"
        else:
            result["status"] = "No breaches found"
            
        logger.info(f"Found {result['total_breaches']} breaches for {email}")
        return result
        
    except requests.RequestException as e:
        logger.error(f"Request failed for breach check of {email}: {str(e)}")
        result["status"] = "Error checking breaches"
        result["error"] = str(e)
        return result
    except Exception as e:
        logger.error(f"Unexpected error in breach check of {email}: {str(e)}")
        result["status"] = "Error checking breaches"
        result["error"] = str(e)
        return result

def analyzeSocialProfile(url: str) -> Dict[str, str]:
    """Analyze social media profile content.
    
    Args:
        url: The URL of the social media profile
        
    Returns:
        A dictionary containing profile analysis results
    """
    logger.info(f"Analyzing social profile at URL: {url}")
    try:
        response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
        response.raise_for_status()
        pageContent = BeautifulSoup(response.text, "html.parser").get_text()
        result = {
            "url": url,
            "contentPreview": pageContent[:200]
        }
        logger.info(f"Successfully analyzed social profile at {url}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing social profile at {url}: {str(e)}")
        return {"url": url, "contentPreview": "Error analyzing profile"}

def generatePhishingEmailTemplate(profile: Dict[str, Any] = None) -> str:
    """Generate a phishing email template based on profile information using LLM.
    
    Args:
        profile: Dictionary containing profile information
        
    Returns:
        A string containing the generated phishing email template
    """
    logger.info("Generating phishing email template using LLM")
    try:
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        domain = ""
        content_preview = ""
        
        if profile is None:
            profile = {}
        
        if isinstance(profile, dict):
            if "domain" in profile:
                domain = profile.get("domain", "")
            elif "url" in profile:
                domain = profile.get("url", "").split("//")[-1].split("/")[0]
            
            content_preview = profile.get("contentPreview", "")
            
            if not domain and "ip_address" in profile:
                domain = profile.get("domain", "unknown-domain.com")
        
        if not domain:
            domain = "target-domain.com"
        
        prompt = f"""
        Based on the following information, generate a realistic phishing email template:
        
        Target Domain: {domain}
        Additional Context: {content_preview}
        
        Generate a phishing email that appears to come from a legitimate source related to this domain.
        The email should be convincing and use social engineering techniques.
        Focus on creating urgency or authority to prompt action.
        Include elements like:
        - Fake security alerts
        - Account verification requests
        - Urgent action required
        - Professional language and formatting
        
        Return only the email content without any additional formatting or explanations.
        """
        
        response = client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert at creating convincing phishing email templates for security research purposes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        template = response.choices[0].message.content.strip()
        logger.info(f"Generated phishing template for domain: {domain}")
        return template
    except Exception as e:
        logger.error(f"Error generating phishing template: {str(e)}")
        return "Error generating template"

def parseThoughtProcess(content: str) -> str:
    """Parse the thought process from the LLM's message content.
    
    Args:
        content: The message content from the LLM
        
    Returns:
        The extracted thought process or a default message
    """
    if not content:
        return "No explicit reasoning provided"
    
    thought_match = re.search(r'THOUGHT:(.*?)(?:ACTION:|$)', content, re.DOTALL)
    if thought_match:
        return thought_match.group(1).strip()
    
    return content.strip()

app = FastAPI()

@app.on_event("startup")
async def startupEvent():
    logger.info("Starting OSINT investigation service")
    Config.validate_config()
    client = OpenAI(api_key=Config.OPENAI_API_KEY)

@app.post("/investigate", response_model=InvestigationResponse)
async def investigateDomain(request: InvestigationRequest) -> InvestigationResponse:
    logger.info(f"Received investigation request for domain: {request.domain}")
    
    try:
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        domain = request.domain.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
        
        functions = [
            {
                "name": "getDomainInfo",
                "description": "Retrieve domain registration information using WHOIS",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "The domain name to look up (e.g., 'example.com')"
                        }
                    },
                    "required": ["domain"]
                }
            },
            {
                "name": "extractEmailAddresses",
                "description": "Extract email addresses from a webpage",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to extract emails from"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "getBreachInfoFromBreachDirectory",
                "description": "Check for data breaches using multiple sources",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "The email address to check for breaches"
                        }
                    },
                    "required": ["email"]
                }
            },
            {
                "name": "analyzeSocialProfile",
                "description": "Analyze social media profile content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the social media profile"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "generatePhishingEmailTemplate",
                "description": "Generate a phishing email template based on profile information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "description": "Dictionary containing profile information"
                        }
                    },
                    "required": ["profile"]
                }
            }
        ]

        messages = [
            {
                "role": "system", 
                "content": """You are Elliot Alderson, a master hacker conducting OSINT investigations. 
                You have access to several functions to gather intelligence about domains and their associated entities.

AVAILABLE FUNCTIONS:
- getDomainInfo: Get WHOIS, DNS, and SSL information about a domain
- extractEmailAddresses: Find email addresses from websites
- getBreachInfoFromBreachDirectory: Check if emails have been involved in data breaches
- analyzeSocialProfile: Analyze social media profiles for intelligence
- generatePhishingEmailTemplate: Create realistic phishing templates based on gathered information

EXPECTED INVESTIGATION RESULTS:
The user expects you to gather information for ALL of the following areas:
- domain_info: Complete domain registration and technical details
- discovered_emails: Email addresses found on the target website
- breach_data: Data breach information for discovered emails
- social_profile: Analysis of any social media profiles found
- phishing_template: A convincing phishing email template based on gathered intel
- final_summary: Comprehensive analysis with domain overview, security assessment, contact points, technical footprint, and recommendations

INVESTIGATION APPROACH:
- Use your expertise to determine the most effective investigation strategy
- Aim to gather information for ALL expected result areas above
- IMPORTANT: Always generate a phishing template using the generatePhishingEmailTemplate function
- Build upon findings from each function to guide your next actions
- Ensure thorough coverage of the target's digital footprint

You have 5 attempts to conduct a comprehensive investigation covering all expected areas.
"""
            },
            {"role": "user", "content": f"Investigate domain {domain}"}
        ]

        investigation = InvestigationResponse(
            domain=domain,
            steps=[],
            summary={}
        )
        
        max_iterations = 5
        current_iteration = 0
        
        while current_iteration < max_iterations:
            current_iteration += 1
            logger.info(f"Starting investigation iteration {current_iteration}")
            
            response = client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                functions=functions,
                function_call="auto",
                temperature=0.1
            )

            message = response.choices[0].message
            
            if message.content:
                logger.info(f"LLM Thought Process: {message.content}")
            
            if message.function_call:
                function_name = message.function_call.name
                arguments = eval(message.function_call.arguments)
                
                logger.info(f"LLM decided to call function: {function_name}")
                logger.info(f"Function arguments: {arguments}")

                result = None
                if function_name == "getDomainInfo":
                    result = getDomainInfo(**arguments)
                elif function_name == "extractEmailAddresses":
                    result = extractEmailAddresses(**arguments)
                elif function_name == "getBreachInfoFromBreachDirectory":
                    result = getBreachInfoFromBreachDirectory(**arguments)
                elif function_name == "analyzeSocialProfile":
                    result = analyzeSocialProfile(**arguments)
                elif function_name == "generatePhishingEmailTemplate":
                    if not arguments or "profile" not in arguments:
                        domain_info = next((step.result for step in investigation.steps if step.action == "getDomainInfo"), {})
                        arguments = {"profile": domain_info}
                    result = generatePhishingEmailTemplate(**arguments)

                logger.info(f"Function execution result: {result}")

                thought = parseThoughtProcess(message.content)
                logger.info(f"Parsed thought: {thought}")
                
                investigation.steps.append(InvestigationStep(
                    thought=thought,
                    action=function_name,
                    arguments=arguments,
                    result=result
                ))

                messages.append(message)
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": str(result)
                })
            else:
                thought = parseThoughtProcess(message.content)
                logger.info(f"Parsed thought (no function call): {thought}")
                
                investigation.steps.append(InvestigationStep(
                    thought=thought,
                    action="no_action",
                    arguments={},
                    result={}
                ))
                
                messages.append(message)
                break

        summary_messages = messages + [
            {
                "role": "system",
                "content": """Provide a comprehensive final summary of your investigation findings. Format the response as a JSON object with the following structure:
{
    "domain_overview": "Brief overview of the domain and its infrastructure",
    "security_assessment": "Analysis of security posture and vulnerabilities",
    "contact_points": "Summary of discovered communication channels",
    "technical_footprint": "Details about technology stack and infrastructure",
    "recommendations": "List of security improvements and best practices"
}
Ensure each field contains properly formatted text with newlines represented as '\\n'."""
            }
        ]
        
        summary_response = client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=summary_messages,
            temperature=0.1
        )
        
        try:
            final_summary = eval(summary_response.choices[0].message.content)
        except:
            final_summary = {
                "domain_overview": summary_response.choices[0].message.content,
                "security_assessment": "Not available",
                "contact_points": "Not available",
                "technical_footprint": "Not available",
                "recommendations": "Not available"
            }
        
        logger.info(f"Final summary from LLM: {final_summary}")

        summary = {
            "domain_info": next((step.result for step in investigation.steps if step.action == "getDomainInfo"), {}),
            "discovered_emails": next((step.result for step in investigation.steps if step.action == "extractEmailAddresses"), []),
            "breach_data": next((step.result for step in investigation.steps if step.action == "getBreachInfoFromBreachDirectory"), {}),
            "social_profile": next((step.result for step in investigation.steps if step.action == "analyzeSocialProfile"), {}),
            "phishing_template": next((step.result for step in investigation.steps if step.action == "generatePhishingEmailTemplate"), ""),
            "final_summary": final_summary
        }
        investigation.summary = summary

        logger.info(f"Investigation completed for domain: {domain}")
        return investigation

    except Exception as e:
        logger.error(f"Error during investigation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
