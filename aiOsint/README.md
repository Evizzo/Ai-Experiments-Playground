# AI OSINT Investigation Tool

An AI-powered OSINT tool that embodies Elliot Alderson's persona to conduct autonomous domain investigations. Using ChatGPT, it intelligently calls various tools to gather and analyze domain information.

## AI Tools & Capabilities

The AI can autonomously call these tools:

- `getDomainInfo`: Retrieves WHOIS data, DNS records, SSL info, and server headers
- `extractEmailAddresses`: Crawls websites and extracts contact information
- `getBreachInfoFromBreachDirectory`: Checks for data breaches
- `analyzeSocialProfile`: Analyzes social media presence
- `generatePhishingEmailTemplate`: Creates context-aware phishing templates

## AI Features

- **Autonomous Investigation**
  - Makes intelligent decisions about which tools to call
  - Adapts investigation strategy based on findings
  - Maintains context between tool calls
  - Provides detailed thought process for each action

- **Intelligent Analysis**
  - Correlates data from multiple sources
  - Identifies security vulnerabilities
  - Generates comprehensive reports
  - Makes actionable recommendations

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
            "thought": "AI's reasoning",
            "action": "tool_called",
            "arguments": {"param": "value"},
            "result": {}
        }
    ],
    "summary": {
        "domain_info": {},
        "discovered_emails": [],
        "breach_data": {},
        "social_profile": {},
        "phishing_template": "",
        "final_summary": {
            "domain_overview": "AI analysis",
            "security_assessment": "AI evaluation",
            "contact_points": "AI discovery",
            "technical_footprint": "AI analysis",
            "recommendations": "AI recommendations"
        }
    }
}
```

## Security & Ethics

- For legitimate security research only
- Ethical AI usage required
- Privacy-aware investigation
- Responsible data handling

## License

MIT License

## Disclaimer

This tool is for educational and legitimate security research purposes only. Users are responsible for ensuring their use complies with applicable laws and regulations.
