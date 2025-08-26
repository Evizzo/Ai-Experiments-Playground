## AI-Powered Security Analysis

The AI assistant can autonomously conduct security research using these capabilities:

- `getDomainInfo`: Analyzes domain registration data, DNS records, SSL certificates, and server configurations
- `extractEmailAddresses`: Identifies publicly available contact information for legitimate security research
- `getBreachInfoFromBreachDirectory`: Checks for known data breaches affecting the domain
- `generateSecurityReport`: Creates comprehensive security assessment reports

## AI Features

- **Intelligent Security Research**
  - Makes data-driven decisions about which security checks to perform
  - Adapts research strategy based on findings
  - Maintains context throughout the security assessment
  - Provides detailed reasoning for each security check

- **Comprehensive Security Analysis**
  - Correlates security data from multiple sources
  - Identifies potential security vulnerabilities
  - Generates detailed security reports
  - Provides actionable security recommendations

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
echo "OPENAI_API_KEY=your_key" > .env

# Run
uvicorn main:app --reload
```

## Example Request

```bash
curl -X POST "http://localhost:8000/investigate" \
     -H "Content-Type: application/json" \
     -d '{"domain": "example.com"}'
```

## AI Response Format

```json
{
    "domain": "example.com",
    "steps": [
        {
            "thought": "AI's reasoning for this security check",
            "action": "security_tool_called",
            "arguments": {"param": "value"},
            "result": {}
        }
    ],
    "summary": {
        "domain_info": {},
        "discovered_emails": [],
        "breach_data": {},
        "social_profile": {},
        "security_report": "",
        "final_summary": {
            "domain_overview": "AI security analysis",
            "security_assessment": "AI vulnerability evaluation",
            "contact_points": "AI discovery",
            "technical_footprint": "AI security analysis",
            "recommendations": "AI security recommendations"
        }
    }
}
```