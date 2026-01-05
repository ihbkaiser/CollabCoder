from typing import List, Optional, Dict, Tuple
import tiktoken
import os
import json
import re
import sys
import time
import random
from datetime import datetime
from copy import deepcopy
from .Base import BaseStrategy # Assuming this is in your project structure
from models.Base import BaseModel # Adjust paths as needed
from datasets.Dataset import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.MBPPDataset import MBPPDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset
from datasets.LCBDataset import *
from results.Results import Results
from evaluations.func_evaluate import evaluate_io
import numpy as np
from forms import * # Assuming this imports your form models like PlanOutput, etc.
from multi_thread import multi_thread_task_dict

def manual_parse_to_dict_4_consistency(text):
    clean_text = text.replace('\n', ' ').strip()
    
    result = {"consistency_scores": {}}
    sections = re.findall(r'"(update plan|update code only)":\s*{(.*?}})', clean_text)
    
    for section_name, section_content in sections:
        result["consistency_scores"][section_name] = {}
        
        sub_blocks = re.findall(r'"(plan-code|plan-content|code-content)":\s*{\s*"consistency":\s*([\d.]+),\s*"reasoning":\s*"(.*?)"\s*}', section_content)
        
        for sub_name, conf_val, reason_val in sub_blocks:
            result["consistency_scores"][section_name][sub_name] = {
                "consistency": float(conf_val),
                "reasoning": reason_val
            }
            
    return result

def manual_parse_to_dict_4_confidence(text):
    clean_text = text.replace('\n', ' ').strip()

    result = {"confidence_scores": {}}
    
    sections = re.findall(r'"(update plan|update code only)":\s*{(.*?}})', clean_text)
    
    for section_name, section_content in sections:
        result["confidence_scores"][section_name] = {}
        

        sub_blocks = re.findall(r'"(plan|code|content)":\s*{\s*"confidence":\s*([\d.]+),\s*"reasoning":\s*"(.*?)"\s*}', section_content)
        
        for sub_name, conf_val, reason_val in sub_blocks:
            result["confidence_scores"][section_name][sub_name] = {
                "confidence": float(conf_val),
                "reasoning": reason_val
            }
            
    return result

class AnalysisReflection:
    """
    Paper-faithful RT: maintains a persistent debugging strategy R(t) across iterations.
    Updates every iteration based on:
      - R(t-1)
      - diagnosis E_X(t) for selected target X(t) (plan or code)
      - problem P
      - current target state X(t) (plan or code)
      - failure log F(t)
    """
    def __init__(self):
        self.strategy = ""          # R(t)
        self.historical_data = {}   # optional logs

    def update_historical_data(self, iteration: int, data: Dict):
        self.historical_data[iteration] = data

    def update_strategy(
        self,
        iteration: int,
        target: str,                 # "plan" or "code"
        diagnosis: Dict,             # E_X(t) (insights + optional simulation)
        problem: str,
        target_state: str,           # π(t) or c(t)
        failure_log: str,
        gpt_chat,
        max_attempts: int = 1,
        verbose: bool = True
    ) -> Tuple[str, int, int]:
        """
        Returns updated R(t) and token usage.
        """
        prev = self.strategy if self.strategy else "No previous strategy."
        insights = diagnosis.get("insights", "")
        simulation = diagnosis.get("simulation", "")

        prompt = f"""
You are maintaining a **stateful debugging strategy** across iterations for a competitive programming solver.

## Problem
{problem}

## Current refinement target
Target = {target}

## Current target state (the thing to be refined)
{target_state}

## Failure log (always correct)
{failure_log}

## Current diagnosis for this target (E_X)
### Simulation (may be empty)
{simulation}

### Insights
{insights}

## Previous debugging strategy R(t-1)
{prev}

# Task
Write an **updated debugging strategy R(t)** that:
- Incorporates the new diagnosis and failure evidence
- Builds on prior strategy but does NOT repeat it verbatim
- States concrete next actions/hypotheses to try (bullet points are fine)
- Avoids ineffective repeated fixes
- Does NOT generate code or a new plan

Return ONLY the updated strategy text.
""".strip()

        pr_tok = 0
        com_tok = 0
        for attempt in range(max_attempts):
            try:
                if verbose:
                    print(f"[RT] Updating strategy, attempt {attempt+1}, target={target}")
                resp, p, c = gpt_chat([{"role": "user", "content": prompt}])
                pr_tok += p
                com_tok += c
                self.strategy = resp.strip()
                # save logs
                self.update_historical_data(iteration, {
                    "target": target,
                    "diagnosis": diagnosis,
                    "failure_log": failure_log,
                    "strategy": self.strategy
                })
                return self.strategy, pr_tok, com_tok
            except Exception as e:
                if verbose:
                    print(f"[RT] Error updating strategy: {e}")
                if attempt == max_attempts - 1:
                    # keep previous strategy if update fails
                    return self.strategy, pr_tok, com_tok
class CollabCoder(BaseStrategy):
    def __init__(
        self,
        k: int = 1,
        t: int = 5,
        max_attempts: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.top_plan = 1
        self.t = t
        self.number_of_code_per_plan = 1
        self.trust_weights = {
            'plan': 0.3,
            'code': 0.4,
            'content': 0.3
        }
        self.analysis_meaning = {
            "plan": "Identifies errors or problems in the planning approach.",
            "code": "Identifies errors or problems in the code implementation.",
            "content": "Identifies mismatches between problem, plan, and code."
        }
        self.history = []
        self.max_attempts = max_attempts
        self.verbose = True
        self.rt = AnalysisReflection() # Initialize AnalysisReflection for debugging guidance
    def _extract_json_string(self, text: str) -> Optional[str]:
        m = re.search(r'```json\s*({[\s\S]*?})\s*```', text, re.DOTALL)
        if not m:
            m = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text, re.DOTALL)
        if not m:
            m = re.search(r'({[\s\S]*})', text, re.DOTALL)
        return m.group(1) if m else None
    def _fix_invalid_escapes(self, json_str: str) -> str:
        json_str = json_str.replace('\b', '\\b').replace('\f', '\\f').replace('\r', '\\r').replace('\t', '\\t')
        json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
        return json_str
    def parse_key_from_md(self, text: str, key: str) -> str:
        """
        Extracts the content under a markdown heading matching the key (e.g., ## key or ### key).
        Supports varying heading levels (#, ##, ###). If no match, returns the whole text as fallback.
        """
        # Flexible pattern for any number of # followed by the key, then content until next heading or end
        # pattern = re.compile(r'#+\s*' + re.escape(key) + r'\s*(.*?)(?=#+\s*|\Z)', re.DOTALL | re.IGNORECASE | re.MULTILINE)
        pattern = re.compile(r'#+\s*' + re.escape(key) + r'\s*(.*?)(?=#+\s*|\Z)', re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        else:
            return text.strip()
    def parse_code(self, response: str) -> str:
        if self.verbose:
            print("Step: Parsing code")
            print(f"Input response: {response}...")
        if "```" not in response:
            return response
        code_pattern = r'```((.|\n)*?)```'
        languages = ['Python', 'python', 'Python3', 'python3', 'C', 'c', 'C++', 'c++', 'Java', 'java', 'Node', 'node', 'Rust', 'rust', 'PHP', 'php', 'Go', 'go', 'Ruby', 'ruby', 'C#', 'c#', 'csharp']
        for lang in languages:
            if f"```{lang}" in response:
                code_pattern = r'```' + lang + r'((.|\n)*?)```'
                break
        code_blocks = re.findall(code_pattern, response, re.DOTALL)
        if code_blocks:
            code_str = code_blocks[-1][0] if isinstance(code_blocks[-1], tuple) else code_blocks[-1]
        else:
            code_str = response
        parsed_code = code_str.strip()
        if self.verbose:
            print("Step: Code parsing successful")
            print(f"Parsed code: {parsed_code}...")
        return parsed_code
    def get_sample_io_str(self, item) -> str:
        import pdb
        # pdb.set_trace()
        if self.verbose:
            print("Step: Getting sample I/O string")
        if isinstance(self.data, XCodeDataset):
            sample_io = f"Input:\n{item['sample_inputs']}\nExpected output:\n{item['sample_outputs']}"
        elif isinstance(self.data, LCBDataset):
            return self.data.get_sample_io(item)
        else:
            sample_io_list = item.get('sample_io', [])
            if sample_io_list:
                if isinstance(sample_io_list[0], str):
                    sample_io = "\n".join(io for io in sample_io_list)
                elif isinstance(sample_io_list[0], dict):
                    sample_io = "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io_list])
            else:
                sample_io = ''
        if self.verbose:
            print("Step: Sample I/O retrieved")
            print(f"Sample I/O: {sample_io}...")
        return sample_io
    def get_problem_understanding(self, item) -> Tuple[str, int, int]:
        
        if self.verbose:
            print("Step: Generating problem understanding")
        problem_text = self.data.get_prompt(item)
        input_for_understanding = [
        {
        "role": "user",
        "content": f"""
        **You are a code generation assistant tasked with analyzing a programming problem.**
        ## Problem Description
        {problem_text}
        ## Sample Input/Output
        {self.get_sample_io_str(item)}
        ## Guidelines (IMPORTANT)
        - Clarify the **requirements** and **objectives** of the problem, give a concise understanding of the problem
        - Outline edge cases and important things to consider
        - Do **not** provide code, algorithms, or full solutions.
        - Clearly highlight what the problem is asking the solver to achieve.
        """
        }
        ]
        import pdb
        # pdb.set_trace()
        try:
            if self.verbose:
                print("Step: Making API call for understanding")
            understanding, pr_tok, com_tok = self.gpt_chat(processed_input=input_for_understanding)
            item['api_calls'] += 1
            if self.verbose:
                print("Step: Understanding parsed")
                print(f"Understanding: {understanding}...")
            return understanding, pr_tok, com_tok
        except Exception as e:
            print(f"Error in get_problem_understanding: {e}")
            return "", 0, 0
    def generate_code_from_plan(self, item, planning: str, problem_text: str, sample_io_prompt: str, previous_codes: str = "", understanding: str = "") -> Tuple[List[Tuple[str, float, str]], int, int]:
        if self.verbose:
            print("Step: Generating code from plan")
            print(f"Plan: {planning}...")
        codes_with_scores = []
        pr_tok = 0
        com_tok = 0
        std_input_prompt = """
    - Strictly follow the sample input and output format.
    - The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take the input using `input()` function then call the function with specified parameters and finally print the output of the function.
    - For array input parse the array then pass it to the function. Parsing technique is given in the sample input output format section.
    - Do not add extra print statement otherwise it will failed the test cases.
         """ if isinstance(self.data, (APPSDataset, CodeContestDataset, XCodeDataset, LCBDataset)) else ""
        context = f"# Problem:\n {problem_text}\n"
        # context += understanding if understanding else ""
        for c_idx in range(1, self.number_of_code_per_plan + 1):
            if self.verbose:
                print(f"Step: Generating code variant {c_idx}")
            diversity_prompt = "" if c_idx == 1 else f"""
**Generate a distinct implementation** from previous ones: {previous_codes}. Use a unique approach, such as alternative data structures (e.g., list vs. dictionary, array vs. set in {self.language}), varied coding patterns (e.g., functional vs. imperative style).
Ensure the implementation strictly follows the provided plan and solves the problem correctly.
"""
            input_for_code_generation = [
                {
                    "role": "user",
                    "content": f"""# Task
**You are a programmer** tasked with solving a given problem using the **{self.language}** programming language. See the plan to solve the plan and implement code to solve it.
{context}
# Planning
{planning}
# Sample Test Cases
{sample_io_prompt}
{diversity_prompt}
# Instructions
- The generated **{self.language}** code must be inside a triple backtick (```) code block.
- Do not add extra explanation or words.
- Do not add assert statements in your code.
{std_input_prompt}
"""
                }
            ]
            try:
                if self.verbose:
                    print("Step: Making API call for code generation")
                code_response, pr_tok_1, com_tok_1 = self.gpt_chat(processed_input=input_for_code_generation)
                pr_tok += pr_tok_1
                com_tok += com_tok_1
                item['api_calls'] += 1
                code = self.parse_code(code_response)
                if self.verbose:
                    print(f"Generated code variant {c_idx}: {code}")
                # Evaluate the code
                try:
                    passed, test_log = self.data.evaluate_sample_io(item, code, self.language)
                    score = 1.0 if passed else 0.0
                except Exception as e:
                    print(f"Error evaluating code: {e}")
                    score = 0.0
                    test_log = f"Evaluation failed: {e}"
                codes_with_scores.append((code, score, test_log))
                previous_codes += f"\n- {code}"
            except Exception as e:
                print(f"Error generating code {c_idx}: {e}")
        if self.verbose:
            print(f"Step: {len(codes_with_scores)} code variants generated and evaluated")
        return codes_with_scores, pr_tok, com_tok
    def generate_plans(self, item, problem_understanding=None) -> Tuple[List[Tuple[str, float]], int, int]:
        if self.verbose:
            print("Step: Starting plan generation")
        plans_with_scores = []
        pr_tok = 0
        com_tok = 0
        previous_approaches = ""
        problem_text = self.data.get_prompt(item)
        sample_io_prompt = self.get_sample_io_str(item)
        if problem_understanding is None:
            problem_understanding, pr_u, com_u = self.get_problem_understanding(item)
            pr_tok += pr_u
            com_tok += com_u
        max_plans = self.k
        for t in range(1, max_plans + 1):
            if self.verbose:
                print(f"Step: Generating plan variant {t}")
            diff_prompt = "" if t == 1 else f", different from the following previous approaches: {previous_approaches}"
            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": (
                        f"You are a programmer tasked with generating appropriate plan to solve a given problem using the **{self.language}** programming language."
                        f"**# Target Problem:**\n{problem_text}\n\n"
                        f"**# Target Problem Understanding:**\n{problem_understanding}\n\n"
                        f"**## Sample Test Cases:**\n{sample_io_prompt}"
                        "**Expected Output Structure:**"
                        "### Recall Example Problem"
                        "**Recall a relevant and distinct problems** (different from problem mentioned above) and"
                        "- **Describe it**"
                        f"- **Generate {self.language} code** step by step to solve that problem"
                        "- **Discuss the algorithm** to solve this problem"
                        "- **Finally generate a planning** to solve that problem"
                        "### Algorithm to solve the original problem"
                        "- **Write down the algorithm** that is well suited for the original problem"
                        "- **Give some tutorials** about the algorithm for example:"
                        " - How to approach this type of algorithm"
                        " - Important things to consider"
                        "### Plan"
                        "- **Write down a detailed, step-by-step plan** to solve the **original problem**."
                        "--------"
                        "**IMPORTANT:**"
                        "- **Strictly follow** the instructions."
                        "- **DO NOT generate** code in your response."
                    ),
                },
            ]
            for attempt in range(self.max_attempts):
                if self.verbose:
                    print(f"Step: Planning generation attempt {attempt + 1} for variant {t}")
                try:
                    planning_resp, pr_tok_temp, com_tok_temp = self.gpt_chat(input_for_problem_planning)
                    planning = self.parse_key_from_md(planning_resp, "Plan")
                    pr_tok += pr_tok_temp
                    com_tok += com_tok_temp
                    item['api_calls'] += 1
                    break
                except Exception as e:
                    print(f"Error in planning attempt {attempt + 1}: {e}")
                    if attempt == self.max_attempts - 1:
                        continue
            llm_score = 1 # Placeholder, as verification is commented out
            plans_with_scores.append((planning, llm_score))
            previous_approaches += f"\n- {planning}"
            if self.verbose:
                print(f"Step: Plan variant {t} completed")
                print(f"LLM score: {llm_score}")
        if len(plans_with_scores) < self.k:
            print(f"Warning: Only {len(plans_with_scores)}/{self.k} valid plans generated")
        if self.verbose:
            print(f"Step: {len(plans_with_scores)} plans generated")
        return plans_with_scores, pr_tok, com_tok
    def merged_analyses(self, plan: str, code: str, test_log: str, problem: str, problem_understanding: str) -> Dict:
        if self.verbose:
            print("Step: Performing merged analyses (plan + code + content in one API call)")
        code_prompt_section = f"### Code\n{code}\n"
        input_prompt = [
            {
                "role": "user",
                "content": f"""
**You are a code generation assistant tasked with analyzing a programming problem in debugging, logical reasoning, and assessing solution alignments.**
---
## Context:
### Problem Description
{problem}
### Proposed Plan
{plan}
{code_prompt_section}
### Test Log (failing input/output)
{test_log}
---
## Response Structure
Your response must be structured with the following sections only:
### Plan Analysis
#### Simulation
Provide a detailed **step-by-step simulation** of the plan on the failing test cases, highlighting where divergences occur.
#### Insight
- Based on this simulation detect any of the following cases:
    - Plan is wrong
    - Plan is correct but plan to code generation is wrong.
- Finally, discuss how to correct this plan.

### Code Analysis
#### Simulation
Provide a detailed **line-by-line simulation** of the code on the failing test cases, highlighting divergences and errors.
#### Insight
Based on the simulation, provide concise insights on how to correct this code.

### Content Analysis
Provide a **single concise insight** (4-5 sentences) that includes:
* A **detailed evaluation** of the alignment between the plan and the code
* A **conclusion** on which component(s) should be updated
  (e.g., update the plan, update the code, update both, or no updates needed)
* Brief suggestions on how to improve alignment if necessary

---
## IMPORTANT
- **Strictly follow** the instructions and structure.
- The **test log is always true**. Do not modify or doubt it.
- Do not be overconfident. The **current plan or code has issues**.
- Do **not** generate new code.
- For content analysis, **focus only** on alignment issues with a concise insight.
- Do **NOT** introduce new solutions or rewrite the code/plan.
"""
            },
        ]
        pr_tok = 0
        com_tok = 0
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Merged analyses attempt {attempt + 1}")
            try:
                response, pr_tok_temp, com_tok_temp = self.gpt_chat(input_prompt)
                print(f"Response from merged analyses: {response}")

                pr_tok += pr_tok_temp
                com_tok += com_tok_temp
                
                # import pdb
                # # pdb.set_trace()
                
                # plan_simulation = self.parse_key_from_md(response, " Plan Analysis").split("#### Insight")[0].split("#### Simulation")[1].strip() if "#### Insight" in self.parse_key_from_md(response, "Plan Analysis") else ""
                # plan_insight = self.parse_key_from_md(response, "Plan Analysis").split("#### Insight")[1].strip() if "#### Insight" in self.parse_key_from_md(response, "Plan Analysis") else self.parse_key_from_md(response, "Plan Analysis")
                # code_simulation = self.parse_key_from_md(response, "Code Analysis").split("#### Insight")[0].split("#### Simulation")[1].strip() if "#### Insight" in self.parse_key_from_md(response, "Code Analysis") else ""
                # code_insight = self.parse_key_from_md(response, "Code Analysis").split("#### Insight")[1].strip() if "#### Insight" in self.parse_key_from_md(response, "Code Analysis") else self.parse_key_from_md(response, "Code Analysis")
                # content_insight = self.parse_key_from_md(response, "Content Analysis")
                try:
                    plan_simulation = response.split("### Plan Analysis")[1].split("#### Simulation")[1].split("#### Insight")[0].strip("\n")
                except:
                    plan_simulation = None
                try:
                    plan_insight = response.split("### Plan Analysis")[1].split("#### Simulation")[1].split("#### Insight")[1].split("### Code Analysis")[0].strip("\n")
                except:
                    plan_insight = None
                try:
                    code_simulation = response.split("### Code Analysis")[1].split("#### Simulation")[1].split("#### Insight")[0].strip("\n")
                except:
                    code_simulation = None
                try:
                    code_insight = response.split("### Code Analysis")[1].split("#### Simulation")[1].split("#### Insight")[1].split("### Content Analysis")[0].strip("\n")
                except:
                    code_insight = None
                try:
                    content_insight = response.split("### Content Analysis")[1].strip("\n")
                except:
                    content_insight = None

                analysis_result = {
                    'plan_analysis': {'simulation': plan_simulation, 'insights': plan_insight},
                    'code_analysis': {'simulation': code_simulation, 'insights': code_insight},
                    'content_analysis': {'insights': content_insight},
                    'pr_tok': pr_tok,
                    'com_tok': com_tok
                }
                if self.verbose:
                    print("Step: Merged analyses successful")
                    print(f"Plan insights: {analysis_result['plan_analysis']['insights']}...")
                    print(f"Code insights: {analysis_result['code_analysis']['insights']}...")
                    print(f"Content insights: {analysis_result['content_analysis']['insights']}...")
                return analysis_result
            except Exception as e:
                print(f"Error in merged_analyses attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    return {
                        'plan_analysis': {'insights': ''},
                        'code_analysis': {'insights': ''},
                        'content_analysis': {'insights': ''},
                        'pr_tok': pr_tok,
                        'com_tok': com_tok
                    }
    def get_all_scores(self, decisions: List[str], analyses: Dict[str, Dict]) -> Tuple[Dict[str, Dict[str, ConfidenceOutput]], Dict[str, Dict[str, ConsistencyOutput]]]:
        """
        Computes all confidence and consistency scores in a single API call.
        Returns: (confidence_scores, consistency_scores)
        """
        if self.verbose:
            print("Step: Computing all confidence and consistency scores in a single API call")
        ANALYSES_ORDER = ["plan", "code", "content"]
        analysis_names = [n for n in ANALYSES_ORDER if n in analyses]
        if not analysis_names:
            return (
                {d: {n: ConfidenceOutput() for n in ANALYSES_ORDER} for d in decisions},
                {d: {} for d in decisions}
            )
        # Generate pairs for consistency
        pairs = []
        for i, n1 in enumerate(analysis_names):
            for n2 in analysis_names[i+1:]:
                pairs.append((n1, n2))
        agent_descriptions = {
            "plan": "Plan Analyst: finds logical flaws, missing steps, or edge cases in the plan.",
            "code": "Code Analyst: finds implementation bugs, logic mistakes, or I/O handling issues.",
            "content": "Content Evaluator: checks misalignment among problem, plan, and code."
        }
        analysis_meanings = {
            "plan": "Evaluates planning approach quality.",
            "code": "Evaluates code implementation quality.",
            "content": "Evaluates alignment between problem, plan, and code."
        }
        packed_analyses = [
            {
                "name": name,
                "role": agent_descriptions.get(name, ""),
                "purpose": analysis_meanings.get(name, ""),
                "insights": analyses.get(name, {}).get("insights", "")
            }
            for name in analysis_names
        ]
        user_content = (
            "**You are a senior competitive programming reviewer.** "
            "**Evaluate** confidence (how strongly each analysis supports/refutes each decision) and consistency (how much analysis pairs agree/disagree on each decision).\n\n"
            f"**Decisions:**\n{json.dumps(decisions)}\n\n"
            "**Decision meanings:**\n"
            "- **update code only:** The plan that generates the code is correct, but the code is wrong (e.g., implementation errors, bugs in code).\n"
            "- **update plan:** Both plan and code are wrong, but should fix the plan because the error is more serious (e.g., wrong approach, misunderstanding of the problem).\n\n"
            f"**Analysis Types and Insights:**\n{json.dumps(packed_analyses, ensure_ascii=False, indent=2)}\n\n"
            f"**Analysis Pairs for Consistency:**\n{json.dumps(pairs)}\n\n"
            "**Confidence Scoring rules** (in [0.0, 1.0]):\n"
            "- **1.0** = strongly supports with clear, direct evidence.\n"
            "- **0.7-0.9** = supports with mostly relevant reasoning, minor gaps.\n"
            "- **0.4-0.6** = weak/partial support; relevant but missing key links.\n"
            "- **0.1-0.3** = minimal/unclear relevance.\n"
            "- **0.0** = no relevance or contradicts the decision.\n\n"
            "**Consistency Scoring rules** (in [0.0, 1.0]):\n"
            "- **1.0** = both clearly support or both clearly refute the decision with aligned reasoning.\n"
            "- **0.7-0.9** = generally agree with minor differences in focus.\n"
            "- **0.4-0.6** = mixed/partial agreement; some overlap but notable differences.\n"
            "- **0.1-0.3** = mostly disagree with conflicting reasoning.\n"
            "- **0.0** = fully contradictory or unrelated conclusions.\n\n"
            "**Instructions:**\n"
            "1) For each **<decision> — <analysis_type>**, judge confidence with brief reasoning (1-3 sentences).\n"
            "2) For each **<decision> — <analysis1-analysis2>**, judge consistency with brief reasoning (1-3 sentences).\n"
            "3) If insights contradict, set low scores.\n\n"
            "**Output JSON ONLY** (no extra text, no markdown):\n"
            "{\n"
            ' "confidence_scores": {\n'
            ' "<decision>": {\n'
            ' "<analysis_type>": {\n'
            ' "confidence": float,\n'
            ' "reasoning": str\n'
            " }\n"
            " }\n"
            " },\n"
            ' "consistency_scores": {\n'
            ' "<decision>": {\n'
            ' "<analysis1>-<analysis2>": {\n'
            ' "consistency": float,\n'
            ' "reasoning": str\n'
            " }\n"
            " }\n"
            " }\n"
            "}"
        )
        messages = [{"role": "user", "content": user_content}]
        confidence_result: Dict[str, Dict[str, ConfidenceOutput]] = {}
        consistency_result: Dict[str, Dict[str, ConsistencyOutput]] = {}
        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"Step: Scores API call attempt {attempt + 1}")
            try:
                response, pr_tok, com_tok = self.gpt_chat(messages)
                # item['api_calls'] += 1
                json_str = self._extract_json_string(response)
                if not json_str:
                    if self.verbose:
                        print(f"Invalid output: No JSON found\nResponse head: {response}...")
                    continue
                json_str = self._fix_invalid_escapes(json_str)
                data = json.loads(json_str)
                
                # import pdb
                # # pdb.set_trace()
                
                confidence_scores = data.get("confidence_scores", {})
                consistency_scores = data.get("consistency_scores", {})
                for d in decisions:
                    confidence_result[d] = {}
                    for name in analysis_names:
                        item = confidence_scores.get(d, {}).get(name, {})
                        confidence_result[d][name] = ConfidenceOutput(
                            confidence=float(item.get("confidence", 0.0) or 0.0),
                            reasoning=str(item.get("reasoning", "") or "")
                        )
                    consistency_result[d] = {}
                    for n1, n2 in pairs:
                        try:
                            key = f"{n1}-{n2}"
                            item = consistency_scores.get(d, {}).get(key, {})
                            consistency_result[d][key] = ConsistencyOutput(
                                consistency=float(item.get("consistency", 0.0) or 0.0),
                                reasoning=str(item.get("reasoning", "") or "")
                            )
                        except:
                            key = f"{n2}-{n1}"
                            item = consistency_scores.get(d, {}).get(key, {})
                            consistency_result[d][key] = ConsistencyOutput(
                                consistency=float(item.get("consistency", 0.0) or 0.0),
                                reasoning=str(item.get("reasoning", "") or "")
                            )
                if self.verbose:
                    print("Step: All scores calculated")
                    for d in decisions:
                        for name in analysis_names:
                            print(f"[CONF] {d}/{name}: {confidence_result[d][name].confidence:.3f}")
                        for key, obj in consistency_result[d].items():
                            print(f"[CONS] {d}/{key}: {obj.consistency:.3f}")
                return confidence_result, consistency_result, pr_tok, com_tok
            except Exception as e:
                print(f"Error in get_all_scores attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    print("Step: Max attempts reached, returning default scores")
                    return (
                        {d: {n: ConfidenceOutput() for n in analysis_names} for d in decisions},
                        {d: {f"{n1}-{n2}": ConsistencyOutput() for n1, n2 in pairs} for d in decisions},
                        0,0
                    )
    def fast_collaborative_decision(self, plan: str, code: str, outcomes: str, item) -> str:
        """
        Updated collaborative_decision to use merged analyses and combined scores.
        """
        if self.verbose:
            print("Step: Starting collaborative decision with merged analysis")
        merged_result = {
            'plan_analysis': {'insights': ''},
            'code_analysis': {'insights': ''},
            'content_analysis': {'insights': ''},
            'pr_tok': 0,
            'com_tok': 0
        }
        try:
            problem_understanding, _, _ = self.get_problem_understanding(item)
            problem_text = self.data.get_prompt(item)
        
            # Perform merged analyses in one API call
            merged_result = self.merged_analyses(plan, code, outcomes, problem_text, problem_understanding)
            item['api_calls'] += 1
            # Extract analysis results
            analyses = {
                'plan': merged_result['plan_analysis'],
                'code': merged_result['code_analysis'],
                'content': merged_result['content_analysis']
            }
        
            decisions = ['update plan', 'update code only']
        
            # Compute all confidence and consistency scores in one API call
            confidence_scores, consistency_scores, pr_tok, com_tok = self.get_all_scores(decisions, analyses)
            merged_result["pr_tok"] += pr_tok
            merged_result["com_tok"] += com_tok
            item['api_calls'] += 1
        
            scores = {}
            for decision in decisions:
                if self.verbose:
                    print(f"Step: Scoring decision '{decision}'")
                total = 0.0
                for name in analyses.keys():
                    w = self.trust_weights[name]
                    conf = confidence_scores[decision][name].confidence
                    cons_prod = 1.0
                    for name2 in analyses.keys():
                        if name2 != name:
                            pair_key_1 = f"{name}-{name2}"
                            pair_key_2 = f"{name2}-{name}"
                            try:
                                cons = consistency_scores[decision][pair_key_1].consistency
                            except KeyError:
                                cons = consistency_scores[decision][pair_key_2].consistency
                            cons_prod *= cons
                    total += w * conf * cons_prod
                scores[decision] = total
                if self.verbose:
                    print(f"Step: Score for '{decision}': {total}")
        
            max_score = max(scores.values())
            candidates = [k for k, v in scores.items() if v == max_score]
            if len(candidates) > 1:
                return "update code only", merged_result
            decision = candidates[0]
            if self.verbose:
                print("Step: Decision made")
                print(f"Decision: {decision}")
            return decision, merged_result
     
        except Exception as e:
            print(f"Error in collaborative_decision: {e}")
            return "update code only", merged_result
    def debug_plan(self, iteration: int, plan: str, diagnosis: Dict, problem: str, decision: str, failure_log: str):
        if self.verbose:
            print(f"Step: Debugging plan at iteration {iteration}")
    
        # Update RT strategy for target=plan (paper Eq.7)
        pr0=0
        com0=0
        R_t, pr1, com1 = self.rt.update_strategy(
            iteration=iteration,
            target="plan",
            diagnosis=diagnosis,
            problem=problem,
            target_state=plan,
            failure_log=failure_log,
            gpt_chat=self.gpt_chat,
            max_attempts=self.max_attempts,
            verbose=self.verbose
        )
    
        # Minimal change: add Current Debugging Strategy
        update_prompt = [
            {
                'role': 'user',
                'content': f"""You are a programmer tasked with generating appropriate plan to solve a given problem using the **{self.language}** programming language. You already have a wrong plan. Correct it so that it can generate correct plan.
    
    ## Problem
    {problem}
    
    ## Current Debugging Strategy
    {R_t}
    
    ## Plan Critique
    {diagnosis.get('insights', '')}
    
    ## Current Test Log
    {failure_log}
    
    Your response must be structured as follows:
    ## New Plan
    - Write down a detailed, step-by-step modified plan to solve the **original problem**.
    - Ensure each step logically follows from the previous one.
    
    **IMPORTANT Instruction:**
    - Your response must contain only the plan.
    - Do not add any explanation.
    - Do not generate code.
    """
            }
        ]
    
        try:
            if self.verbose:
                print("Step: Making API call for plan update")
            updated_response, pr2, com2 = self.gpt_chat(update_prompt)
            revised_plan = updated_response.strip()
        except Exception as e:
            print(f"Error debugging plan: {e}")
            revised_plan = plan
            pr0 = 0
            com0 = 0
        pr0 += (pr1+pr2)
        com0 += (com1+com2)
        return revised_plan, pr0, com0
    def debug_code(
    self,
    iteration: int,
    plan: str,
    code: str,
    diagnosis: Dict,
    problem: str,
    decision: str,
    failure_log: str,
):
        if self.verbose:
            print(f"Step: Debugging code at iteration {iteration}")
    
        # Update RT strategy for target=code (paper Eq.7)
        R_t, pr2, com2 = self.rt.update_strategy(
            iteration=iteration,
            target="code",
            diagnosis=diagnosis,
            problem=problem,
            target_state=code,
            failure_log=failure_log,
            gpt_chat=self.gpt_chat,
            max_attempts=self.max_attempts,
            verbose=self.verbose,
        )
        pr0=0
        com0=0
        std_input_prompt = (
            """..."""
            if isinstance(
                self.data, (APPSDataset, CodeContestDataset, XCodeDataset, LCBDataset)
            )
            else ""
        )
    
        insights = diagnosis.get("insights", "No insights provided")
    
        code_prompt = [
            {
                "role": "user",
                "content": f"""You are a programmer who has received a solution of a problem written in **{self.language}** that fails to pass certain test cases. Your task is to modify the code in such a way so that it can pass all the test cases. Do not generate same code.
    
    ## Problem:
    {problem}
    
    ## Current Plan
    {plan}
    
    ## Current Debugging Strategy
    {R_t}
    
    ## Buggy Code
    ```{self.language}
    {code}
    ````
    
    ## Test Log
    
    {failure_log}
    
    ## Code Critique
    
    {insights}
    
    **Task:** Using the provided code critique and test log, **refine the code** to correct the identified issues.
    
    **IMPORTANT:** Your response must contain **only the {self.language} code** to solve this problem:
    
    # Your corrected code, with comments explaining each correction.
    
    **Important Instructions:**
    
    * Strictly follow the instructions.
    * Do not add testing code for example assert statement in your code.
    * Do not be overconfident that the generated code is correct. It is wrong.
    * The modified **{self.language}** code must be enclosed within triple backticks
      {std_input_prompt}
      """,
            }
        ]
        try:
            if self.verbose:
                print("Step: Making API call for code update")
                updated_response, pr1, com1 = self.gpt_chat(code_prompt)
                revised_code = self.parse_code(updated_response)
                
        except Exception as e:
            print(f"Error debugging code: {e}")
            revised_code = code
            pr0=0
            com0=0
        pr0+=(pr1+pr2)
        com0+=(com1+com2)
    
        return revised_code, pr0,com0

    def _inner_run(self, item):
        self.rt.historical_data = {}
        if self.verbose:
            print("Step: Starting inner run")
        pr_tok = 0
        com_tok = 0
        all_codes_with_scores = [] # List to collect (code, score)

        try:
            problem_understanding, pr_tok_u, com_tok_u = self.get_problem_understanding(item)
            pr_tok += pr_tok_u
            com_tok += com_tok_u
        except Exception as e:
            print(f"Error getting problem understanding: {e}")
            problem_understanding = ""
        try:
            plans_with_scores, pr_tok_p, com_tok_p = self.generate_plans(item, problem_understanding)
            pr_tok += pr_tok_p
            com_tok += com_tok_p
            if self.verbose:
                print("Step: Plans generated")
                print(f"Number of plans: {len(plans_with_scores)}")
        except Exception as e:
            print(f"Error generating plans: {e}")
            plans_with_scores = []
        if not plans_with_scores:
            print("Warning: No valid plans generated. Returning default code.")
            return "# No valid solution generated", pr_tok, com_tok
        problem_text = self.data.get_prompt(item)
        sample_io_prompt = self.get_sample_io_str(item)
        for plan_idx, (planning, plan_score) in enumerate(plans_with_scores, 1):
            if self.verbose:
                print(f"Step: Processing plan {plan_idx}")
            try:
                codes_with_scores, pr_tok_code, com_tok_code = self.generate_code_from_plan(
                    item, planning, problem_text, sample_io_prompt, "", problem_understanding
                )
                pr_tok += pr_tok_code
                com_tok += com_tok_code
            except Exception as e:
                print(f"Error generating codes for plan {plan_idx}: {e}")
                continue
            for code_idx, (code, code_score, test_log) in enumerate(codes_with_scores, 1):
                all_codes_with_scores.append((code, code_score))
                if self.verbose:
                    print(f"Step: Added initial code for plan {plan_idx}, code {code_idx} - Score: {code_score}")
                passed = code_score == 1.0
                if passed:
                    if self.verbose:
                        print(f"Step: Code passed samples for plan {plan_idx}, code {code_idx}")
                    return code, pr_tok, com_tok
                current_planning = planning
                current_code = code
                current_test_log = test_log
                current_code_score = code_score
                for i in range(1, self.t + 1):
                    if self.verbose:
                        print(f"Step: Iteration {i} for plan {plan_idx}")

                    # 1) Make a decision (try), otherwise fallback safely
                    merged_result = {
                        'plan_analysis': {'insights': '', 'simulation': ''},
                        'code_analysis': {'insights': '', 'simulation': ''},
                        'content_analysis': {'insights': ''},
                        'pr_tok': 0,
                        'com_tok': 0
                    }
                    try:
                        decision, merged_result = self.fast_collaborative_decision(
                            current_planning, current_code, current_test_log, item
                        )
                        if self.verbose:
                            print(f"Step: Decision made: {decision}")
                    except Exception as e:
                        print(f"Error in decision: {e}")
                        decision = "update code only"  # safest default

                    # 2) Apply the decision (this MUST happen on the normal path too)
                    try:
                        if decision == "update plan":
                            diagnosis = merged_result.get("plan_analysis", {})

                            item['num_of_plan_update'] += 1

                            # RT updated + plan refined
                            revised_plan, pr_0, com_0 = self.debug_plan(
                                iteration=i,
                                plan=current_planning,
                                diagnosis=diagnosis,
                                problem=problem_text,
                                decision=decision,
                                failure_log=current_test_log,
                            )
                            pr_tok += pr_0
                            com_tok += com_0
                            current_planning = revised_plan

                            # After plan update, regenerate code from updated plan
                            codes_with_scores, pr_tok_code, com_tok_code = self.generate_code_from_plan(
                                item, current_planning, problem_text, sample_io_prompt, "", problem_understanding
                            )
                            pr_tok += pr_tok_code
                            com_tok += com_tok_code

                            if codes_with_scores:
                                best_new = max(codes_with_scores, key=lambda x: x[1])
                                current_code, current_code_score, current_test_log = best_new
                                all_codes_with_scores.append((current_code, current_code_score))

                                if current_code_score == 1.0:
                                    return current_code, pr_tok, com_tok
                            else:
                                # if regeneration fails, fall back to updating code next loop
                                decision = "update code only"

                        else:  # "update code only" (default)
                            diagnosis = merged_result.get("code_analysis", {})

                            item['num_of_code_update'] += 1

                            revised_code, pr_0, com_0 = self.debug_code(
                                iteration=i,
                                plan=current_planning,
                                code=current_code,
                                diagnosis=diagnosis,
                                problem=problem_text,
                                decision=decision,
                                failure_log=current_test_log,
                            )
                            pr_tok += pr_0
                            com_tok += com_0

                            # Evaluate updated code
                            try:
                                passed, new_test_log = self.data.evaluate_sample_io(item, revised_code, self.language)
                                new_score = 1.0 if passed else 0.0
                            except Exception as e:
                                new_test_log = f"Evaluation failed: {e}"
                                new_score = 0.0

                            current_code = revised_code
                            current_code_score = new_score
                            current_test_log = new_test_log
                            all_codes_with_scores.append((current_code, current_code_score))

                            if current_code_score == 1.0:
                                return current_code, pr_tok, com_tok

                    except Exception as e:
                        print(f"Error applying update at iteration {i}: {e}")
                        # keep current_* as-is and continue
                        continue


        # At the end, select the code with the highest score
        if all_codes_with_scores:
            best_code, best_score = max(all_codes_with_scores, key=lambda x: x[1])
            if self.verbose:
                print(f"Step: Selected best code with score: {best_score}")
            return best_code, pr_tok, com_tok
        else:
            print("Warning: No codes generated. Returning default.")
            return "# No valid solution generated", pr_tok, com_tok
    def run_single_pass(self, item: dict):
        if self.verbose:
            print("Step: Starting single pass run")
        max_retries = 1
        for attempt in range(1, max_retries + 1):
            if self.verbose:
                print(f"Step: Run attempt {attempt}")
            try:
                item['api_calls'] = item.get('api_calls', 0)
                item['num_of_plan_update'] = item.get('num_of_plan_update', 0)
                item['num_of_code_update'] = item.get('num_of_code_update', 0)
                result = self._inner_run(item)
                if self.verbose:
                    print("Step: Run successful")
                return result
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    return "No_solution_found", 0, 0
