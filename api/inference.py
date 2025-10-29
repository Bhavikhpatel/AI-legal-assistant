from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import re

class LegalInference:
    """Advanced LLM inference with clean, BNS-only output"""
    
    def __init__(self, model="llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.llm = ChatGroq(model=model, api_key=api_key, temperature=0.3)
        print(f"✅ LLM initialized: {model}")
        
        # Query understanding prompt (unchanged)
        self.query_understanding_prompt = ChatPromptTemplate.from_template("""
You are a legal query analyzer for Bharatiya Nyaya Sanhita (BNS).

Extract the core legal intent from the user's query and output ONLY a JSON object.

**Rules:**
1. Identify the PRIMARY OFFENSE (murder, theft, rape, assault, etc.)
2. Extract KEY LEGAL TERMS (punishment, section, imprisonment, fine, etc.)
3. Use standard legal terminology, not colloquial language
4. If multiple offenses, list in order of relevance

**Examples:**

User: "what is the punishment for theft"
Output: {{"primary_offense": "theft", "keywords": ["punishment", "theft", "penalty"], "intent": "punishment_inquiry"}}

User: "someone stole my phone, what can I do legally?"
Output: {{"primary_offense": "theft", "keywords": ["theft", "stolen property", "legal remedy"], "intent": "legal_remedy"}}

User: "murder punishment in BNS"
Output: {{"primary_offense": "murder", "keywords": ["murder", "punishment", "section 103"], "intent": "punishment_inquiry"}}

User: "what is section 303 about"
Output: {{"primary_offense": "theft", "keywords": ["section 303", "theft"], "intent": "section_inquiry"}}

User: "is hurting someone illegal?"
Output: {{"primary_offense": "voluntarily causing hurt", "keywords": ["hurt", "assault", "injury"], "intent": "offense_definition"}}

---

User Query: "{query}"

Output JSON:""")
        
        # IMPROVED: System prompt that enforces BNS-only, clean output
        self.system_prompt = """You are a legal assistant helping citizens understand Bharatiya Nyaya Sanhita (BNS) - India's new criminal code.

**CRITICAL RULES:**
1. ONLY reference BNS sections (NEVER mention IPC or Indian Penal Code)
2. Use simple, clear language for non-lawyers
3. Be concise - no lengthy legal explanations
4. Structure output with clear sections ONLY
5. DO NOT include sections titled "Explanation", "Complete Citations", or "Applicability"

**Output Structure (MANDATORY):**
- **Chapter:** [Name only]
- **Section:** [Number only from BNS]
- **Punishment:** [Clear bullet points]
- **What This Means:** [1-2 simple sentences]

**STRICTLY FORBIDDEN:**
- ❌ IPC references
- ❌ "Explanation:" sections
- ❌ "Complete Citations:" sections  
- ❌ "Applicability to query" discussions
- ❌ "Distinctions between" explanations
- ❌ Long paragraphs of legal reasoning"""

        # IMPROVED: Ultra-clean output template
        self.user_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """User Question: {original_query}

Matched Offense: {offense}
Confidence: {confidence_level} ({confidence_score:.1%})
{confidence_warning}

Legal Information from BNS Database:
{context}

---

Provide a CLEAN, STRUCTURED answer with ONLY these sections (NO other sections allowed):

**Chapter:** [Chapter name from BNS]

**Section:** [BNS section number - NOT IPC]

**Punishment:**
• [Imprisonment details]
• [Fine details]
• [Any alternatives]

**What This Means:**
[1-2 simple sentences explaining the offense in plain language]

---

CRITICAL REQUIREMENTS:
1. Use ONLY information from the context provided above
2. Reference ONLY BNS sections (never IPC)
3. DO NOT add "Explanation:" section
4. DO NOT add "Complete Citations:" section
5. DO NOT add "Applicability" discussions
6. Keep it short and user-friendly

Generate the answer NOW:""")
        ])

    def understand_query(self, user_query):
        """Extract legal intent and keywords from user query"""
        try:
            print(f"🧠 Understanding query: '{user_query}'")
            
            formatted_prompt = self.query_understanding_prompt.format(query=user_query)
            response = self.llm.invoke(formatted_prompt)
            
            response_text = response.content.strip()
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            query_analysis = json.loads(response_text)
            
            primary = query_analysis.get("primary_offense", "")
            keywords = query_analysis.get("keywords", [])
            
            reformulated = f"{primary} {' '.join(keywords[:5])}"
            query_analysis["reformulated_query"] = reformulated
            
            print(f"  ✅ Primary Offense: {primary}")
            print(f"  📝 Keywords: {', '.join(keywords)}")
            
            return query_analysis
            
        except Exception as e:
            print(f"⚠️ Query understanding failed: {e}")
            return {
                "primary_offense": user_query,
                "keywords": user_query.split(),
                "intent": "general",
                "reformulated_query": user_query
            }

    def generate_interpretation(self, context, offense_name, confidence_level="MEDIUM", 
                               confidence_score=0.7, confidence_warning=None, original_query=""):
        """Generate clean, BNS-only legal interpretation"""
        try:
            warning_text = f"\n⚠️ **{confidence_warning}**" if confidence_warning else ""
            
            formatted_prompt = self.user_prompt_template.format_messages(
                original_query=original_query or "N/A",
                offense=offense_name,
                context=context,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                confidence_warning=warning_text
            )
            
            response = self.llm.invoke(formatted_prompt)
            output = response.content
            
            # STEP 1: Remove unwanted sections
            output = self._remove_unwanted_sections(output)
            
            # STEP 2: Replace IPC with BNS
            output = self._replace_ipc_with_bns(output)
            
            # STEP 3: Clean up formatting
            output = self._clean_formatting(output)
            
            # STEP 4: Add LOW confidence disclaimer if needed
            if confidence_level == "LOW":
                disclaimer = "⚠️ **Please Note:** This match has low confidence. Consult a legal professional for accurate advice.\n\n---\n\n"
                output = disclaimer + output
            
            # STEP 5: Validate BNS sections
            output = self._validate_bns_sections(output, context)
            
            return output
            
        except Exception as e:
            print(f"❌ LLM error: {e}")
            return f"Error generating interpretation: {str(e)}"
    
    def _remove_unwanted_sections(self, output):
        """Remove Explanation and Complete Citations sections"""
        # Patterns to remove
        unwanted_patterns = [
            # Remove "Explanation:" sections
            r'(?i)\*\*Explanation:?\*\*.*?(?=\n\*\*|\n\n\*\*|\Z)',
            r'(?i)Explanation:.*?(?=\n\n|\Z)',
            
            # Remove "Complete Citations:" sections
            r'(?i)\*\*Complete Citations:?\*\*.*?(?=\n\*\*|\n\n\*\*|\Z)',
            r'(?i)Complete Citations:.*?(?=\n\n|\Z)',
            
            # Remove "Applicability" sections
            r'(?i)\*\*Applicability.*?\*\*.*?(?=\n\*\*|\n\n\*\*|\Z)',
            r'(?i)Applicability.*?(?=\n\n|\Z)',
            
            # Remove long explanation paragraphs starting with "The provisions of"
            r'(?i)The provisions of.*?(?=\n\n|\Z)',
            
            # Remove "It is essential to note" paragraphs
            r'(?i)It is essential to note.*?(?=\n\n|\Z)',
            
            # Remove "Related offenses" mentions
            r'(?i)Related offenses.*?are mentioned.*?(?=\n\n|\Z)',
            
            # Remove IPC section lists
            r'(?i)Sections? \d+.*?(?:IPC|Indian Penal Code).*?(?=\n\n|\Z)',
        ]
        
        for pattern in unwanted_patterns:
            output = re.sub(pattern, '', output, flags=re.DOTALL)
        
        return output
    
    def _replace_ipc_with_bns(self, output):
        """Replace all IPC references with BNS"""
        # Replace full forms
        output = re.sub(r'\bIndian Penal Code\b', 'Bharatiya Nyaya Sanhita', output, flags=re.IGNORECASE)
        output = re.sub(r'\bIPC\b', 'BNS', output)
        
        # Replace "Section XXX of the IPC" with "Section XXX of BNS"
        output = re.sub(r'(Section\s+\d+)\s+of\s+(?:the\s+)?IPC', r'\1 of BNS', output, flags=re.IGNORECASE)
        
        return output
    
    def _clean_formatting(self, output):
        """Clean up extra whitespace and formatting"""
        # Remove multiple blank lines
        output = re.sub(r'\n{3,}', '\n\n', output)
        
        # Remove trailing whitespace
        output = '\n'.join(line.rstrip() for line in output.split('\n'))
        
        # Remove any remaining "---" dividers that separate unwanted sections
        output = re.sub(r'\n---\n\s*\n', '\n\n', output)
        
        return output.strip()
    
    def _validate_bns_sections(self, output, context):
        """Ensure only BNS sections from context are mentioned"""
        # Extract section numbers from output
        output_sections = set(re.findall(r'(?:Section|section)\s+(\d+)', output))
        context_sections = set(re.findall(r'(?:Section|section|number)[\s:]+(\d+)', context))
        
        # Check for hallucinated sections
        hallucinated_sections = output_sections - context_sections
        if hallucinated_sections and len(hallucinated_sections) > 0:
            # Only warn if the sections are significantly different
            warning = f"\n\n⚠️ **Note:** Please verify Section(s) {', '.join(sorted(hallucinated_sections))} against official BNS document."
            output = output + warning
        
        return output
