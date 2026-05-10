# AI Assistance Log

This document lists how Claude Code (claude-sonnet-4-6) was used during development,
as requested in the case deliverable instructions.

## What was generated / assisted

| Area | Assistance Level | Notes |
|------|-----------------|-------|
| Project architecture | Full design | Proposed module boundaries, SOLID layer split |
| `app/core/config.py` | Generated | pydantic-settings wiring |
| `app/core/exceptions.py` | Generated | Domain exception hierarchy |
| `app/schemas/retention.py` | Generated | Pydantic response models |
| `app/api/v1/retention.py` | Generated | FastAPI route + exception mapping |
| `app/api/v1/health.py` | Generated | Liveness + readiness probes |
| `app/dependencies.py` | Generated | FastAPI Depends() providers |

## Prompts used

Debugging and fixing issues were handled interactively through a Claude Code session running in the       
  project directory. The assistant read the provided CSV and .pkl files to understand the data schema and
  model interface before generating any service code. All prompts were conversational and iterative — no    
  large prompt templates were written in advance

## Human decisions

 - Designed the full architecture and workflow of the project, defining what each layer should own before  
  any code was written.                                                                                   
  - Chose gpt-4o-mini as the default LLM for cost/latency balance in a PoC context (swappable via .env).    
  - Chose to surface customer_id (not a real name) in personalisation since the customer database has no name column.                                                                                              
  - Reviewed and approved the 7-rule guardrail set as appropriate to Vodafone brand and compliance          
  guidelines.                                                                                               
  - Decided to run the service in demo mode (no OpenAI key) for the submission, so reviewers can test the 
  full pipeline without API credentials.                                                                    
  - Diagram images were created using a GPT-based model separately, chosen for its superior rendering of  
  colourful visual diagrams.                                                                                
  - README structure, section order, and screenshot placement were directed by the engineer; half of the content was  
  drafted by the assistant.        
