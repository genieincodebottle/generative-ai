"""
LangGraph - Intelligent Task Planning & Execution System

Real-world task planning with AI-powered features:
- Intelligent project analysis and task breakdown
- Smart dependency detection and critical path analysis
- Resource optimization and realistic timeline estimation
- Risk assessment and mitigation planning
- Progress tracking with adaptive replanning
- Multi-domain project templates (software, marketing, research, events)
- Human-in-the-loop approval and refinement
"""

import asyncio
import streamlit as st
import json
import os
from typing import Annotated, TypedDict, Literal, List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# LangGraph 2025 imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Pydantic for structured outputs
from pydantic import BaseModel, Field

load_dotenv()

# State Management for Task Planning
class TaskPlanningState(TypedDict):
    messages: Annotated[list, add_messages]
    main_goal: str
    task_plan: Dict[str, Any]
    current_task: Optional[str]
    completed_tasks: List[str]
    failed_tasks: List[str]
    available_resources: Dict[str, Any]
    execution_context: Dict[str, Any]
    planning_stage: str
    next_action: str

# Data Models
class Task(BaseModel):
    """Individual task structure"""
    task_id: str = Field(description="Unique task identifier")
    title: str = Field(description="Task title")
    description: str = Field(description="Detailed task description")
    priority: Literal["low", "medium", "high", "critical"] = Field(description="Task priority")
    estimated_duration: int = Field(description="Estimated duration in minutes")
    dependencies: List[str] = Field(default=[], description="Task dependencies")
    resources_required: List[str] = Field(default=[], description="Required resources")
    status: Literal["pending", "in_progress", "completed", "failed", "blocked"] = Field(default="pending")
    assigned_agent: Optional[str] = Field(description="Assigned agent or executor")

class TaskPlan(BaseModel):
    """Complete task plan structure"""
    plan_id: str = Field(description="Unique plan identifier")
    main_objective: str = Field(description="Main goal/objective")
    tasks: List[Task] = Field(description="List of tasks")
    execution_order: List[str] = Field(description="Recommended execution order")
    total_estimated_time: int = Field(description="Total estimated time in minutes")
    critical_path: List[str] = Field(description="Critical path task IDs")
    risk_factors: List[str] = Field(default=[], description="Identified risks")

# Intelligent Task Planning Tools
@tool
def intelligent_task_analyzer(main_goal: str, project_context: str = "") -> str:
    """Analyze project goal and intelligently break it down into actionable tasks"""
    try:
        # Smart project classification and analysis
        goal_lower = main_goal.lower()
        context_lower = project_context.lower()

        # Determine project domain and complexity
        project_analysis = {
            "domain": "general",
            "complexity": "medium",
            "estimated_duration": "4-6 weeks",
            "key_skills_required": [],
            "potential_challenges": [],
            "success_metrics": []
        }

        # Software/Tech Projects
        if any(keyword in goal_lower for keyword in ["app", "software", "website", "api", "system", "platform", "code", "develop"]):
            project_analysis.update({
                "domain": "software_development",
                "key_skills_required": ["programming", "testing", "design", "deployment"],
                "potential_challenges": ["technical debt", "scope creep", "integration issues", "performance bottlenecks"],
                "success_metrics": ["functionality delivered", "performance benchmarks", "user adoption", "bug count"]
            })

        # Marketing/Business Projects
        elif any(keyword in goal_lower for keyword in ["marketing", "campaign", "brand", "launch", "promotion", "sales"]):
            project_analysis.update({
                "domain": "marketing",
                "key_skills_required": ["content creation", "analytics", "design", "market research"],
                "potential_challenges": ["audience engagement", "budget constraints", "competition", "market timing"],
                "success_metrics": ["reach and impressions", "conversion rates", "ROI", "brand awareness"]
            })

        # Research/Analysis Projects
        elif any(keyword in goal_lower for keyword in ["research", "study", "analyze", "investigate", "report", "analysis"]):
            project_analysis.update({
                "domain": "research",
                "key_skills_required": ["data analysis", "research methodology", "writing", "critical thinking"],
                "potential_challenges": ["data quality", "research bias", "time constraints", "resource access"],
                "success_metrics": ["research quality", "insights generated", "actionable recommendations", "stakeholder satisfaction"]
            })

        # Event/Project Management
        elif any(keyword in goal_lower for keyword in ["event", "conference", "workshop", "meeting", "organize"]):
            project_analysis.update({
                "domain": "event_management",
                "key_skills_required": ["project management", "vendor coordination", "logistics", "communication"],
                "potential_challenges": ["venue availability", "vendor reliability", "weather/external factors", "attendance"],
                "success_metrics": ["attendance rate", "participant satisfaction", "budget adherence", "objectives met"]
            })

        # Assess complexity based on keywords
        high_complexity_indicators = ["enterprise", "complex", "advanced", "integration", "multiple", "large-scale", "global"]
        low_complexity_indicators = ["simple", "basic", "small", "prototype", "mvp", "pilot"]

        if any(indicator in goal_lower + context_lower for indicator in high_complexity_indicators):
            project_analysis["complexity"] = "high"
            project_analysis["estimated_duration"] = "8-12 weeks"
        elif any(indicator in goal_lower + context_lower for indicator in low_complexity_indicators):
            project_analysis["complexity"] = "low"
            project_analysis["estimated_duration"] = "2-4 weeks"

        return json.dumps(project_analysis, indent=2)

    except Exception as e:
        return f"Project analysis error: {str(e)}"

@tool
def smart_task_generator(main_goal: str, project_analysis: str, team_size: int = 3) -> str:
    """Generate realistic, actionable tasks based on project analysis"""
    try:
        analysis = json.loads(project_analysis)
        domain = analysis.get("domain", "general")
        complexity = analysis.get("complexity", "medium")

        # Domain-specific task templates with realistic breakdowns
        task_templates = {
            "software_development": [
                {
                    "phase": "Planning & Analysis",
                    "tasks": [
                        "Requirements gathering and stakeholder interviews",
                        "Technical architecture design and technology stack selection",
                        "Database schema design and API specification",
                        "UI/UX wireframes and user journey mapping",
                        "Development environment setup and CI/CD pipeline"
                    ]
                },
                {
                    "phase": "Development",
                    "tasks": [
                        "Backend API development and database implementation",
                        "Frontend user interface development",
                        "User authentication and authorization system",
                        "Core business logic implementation",
                        "Third-party integrations and external API connections"
                    ]
                },
                {
                    "phase": "Testing & Quality",
                    "tasks": [
                        "Unit testing and automated test suite development",
                        "Integration testing and API testing",
                        "User acceptance testing and bug fixes",
                        "Performance testing and optimization",
                        "Security audit and vulnerability assessment"
                    ]
                },
                {
                    "phase": "Deployment & Launch",
                    "tasks": [
                        "Production environment setup and configuration",
                        "Database migration and data setup",
                        "Application deployment and monitoring setup",
                        "Documentation and user guides creation",
                        "Launch coordination and post-launch monitoring"
                    ]
                }
            ],
            "marketing": [
                {
                    "phase": "Research & Strategy",
                    "tasks": [
                        "Target audience research and persona development",
                        "Competitive analysis and market positioning",
                        "Campaign objectives and KPI definition",
                        "Channel strategy and budget allocation",
                        "Content strategy and messaging framework"
                    ]
                },
                {
                    "phase": "Content Creation",
                    "tasks": [
                        "Brand assets and visual identity development",
                        "Website and landing page optimization",
                        "Social media content calendar creation",
                        "Email marketing templates and automation setup",
                        "Video and multimedia content production"
                    ]
                },
                {
                    "phase": "Campaign Execution",
                    "tasks": [
                        "Paid advertising campaigns setup (Google, Facebook, etc.)",
                        "Social media posting and community management",
                        "Email marketing campaigns and newsletters",
                        "Influencer outreach and partnership coordination",
                        "PR and media relations activities"
                    ]
                },
                {
                    "phase": "Analytics & Optimization",
                    "tasks": [
                        "Analytics tracking and reporting dashboard setup",
                        "A/B testing implementation and analysis",
                        "Campaign performance monitoring and reporting",
                        "ROI analysis and budget optimization",
                        "Strategy refinement and future planning"
                    ]
                }
            ],
            "research": [
                {
                    "phase": "Research Design",
                    "tasks": [
                        "Research questions and hypothesis formulation",
                        "Literature review and theoretical framework",
                        "Methodology selection and research design",
                        "Data collection strategy and tools selection",
                        "Ethics approval and compliance requirements"
                    ]
                },
                {
                    "phase": "Data Collection",
                    "tasks": [
                        "Survey design and questionnaire development",
                        "Participant recruitment and screening",
                        "Data collection execution and quality control",
                        "Interview conduct and transcript preparation",
                        "Secondary data gathering and validation"
                    ]
                },
                {
                    "phase": "Analysis & Insights",
                    "tasks": [
                        "Data cleaning and preprocessing",
                        "Statistical analysis and pattern identification",
                        "Qualitative analysis and thematic coding",
                        "Results interpretation and insight generation",
                        "Cross-validation and reliability testing"
                    ]
                },
                {
                    "phase": "Reporting & Dissemination",
                    "tasks": [
                        "Research report writing and structure",
                        "Executive summary and key findings presentation",
                        "Peer review and expert validation",
                        "Final report formatting and publication",
                        "Stakeholder presentation and knowledge transfer"
                    ]
                }
            ],
            "event_management": [
                {
                    "phase": "Planning & Design",
                    "tasks": [
                        "Event objectives and success criteria definition",
                        "Target audience analysis and registration strategy",
                        "Budget planning and financial management",
                        "Venue research, selection, and booking",
                        "Event timeline and logistics planning"
                    ]
                },
                {
                    "phase": "Content & Programming",
                    "tasks": [
                        "Speaker recruitment and coordination",
                        "Agenda development and session planning",
                        "Workshop and breakout session design",
                        "Entertainment and networking activity planning",
                        "Catering and hospitality arrangements"
                    ]
                },
                {
                    "phase": "Marketing & Registration",
                    "tasks": [
                        "Event website and registration system setup",
                        "Marketing campaign development and execution",
                        "Social media promotion and community building",
                        "Partnership and sponsorship coordination",
                        "Attendee communication and engagement"
                    ]
                },
                {
                    "phase": "Execution & Follow-up",
                    "tasks": [
                        "Event setup and technical equipment coordination",
                        "Registration and check-in management",
                        "Live event coordination and troubleshooting",
                        "Post-event survey and feedback collection",
                        "Event wrap-up, analysis, and reporting"
                    ]
                }
            ]
        }

        # Get appropriate template or create generic one
        template = task_templates.get(domain, [
            {
                "phase": "Planning",
                "tasks": ["Project initiation and scope definition", "Resource planning and team assembly", "Timeline and milestone planning"]
            },
            {
                "phase": "Execution",
                "tasks": ["Core work execution", "Progress monitoring and adjustments", "Quality assurance and review"]
            },
            {
                "phase": "Delivery",
                "tasks": ["Final deliverables preparation", "Stakeholder review and approval", "Project closure and documentation"]
            }
        ])

        # Generate tasks with realistic estimates
        all_tasks = []
        task_id_counter = 1

        for phase_info in template:
            phase = phase_info["phase"]
            phase_tasks = phase_info["tasks"]

            for task_title in phase_tasks:
                # Estimate duration based on complexity and team size
                base_duration = 8  # hours
                if complexity == "high":
                    base_duration = 16
                elif complexity == "low":
                    base_duration = 4

                # Adjust for team size (more people = potentially faster, but with coordination overhead)
                if team_size > 5:
                    duration_modifier = 0.8  # Slight efficiency loss due to coordination
                elif team_size < 2:
                    duration_modifier = 1.5  # Single person takes longer
                else:
                    duration_modifier = 1.0

                estimated_hours = int(base_duration * duration_modifier)

                # Determine priority based on phase and position
                if phase in ["Planning", "Research Design", "Planning & Analysis"]:
                    priority = "high"
                elif "Testing" in phase or "Quality" in phase:
                    priority = "high"
                else:
                    priority = "medium"

                # Determine required skills based on task content
                required_skills = analysis.get("key_skills_required", ["general"])[:2]
                task_lower = task_title.lower()

                if "deployment" in task_lower or "production" in task_lower:
                    required_skills = ["deployment", "technical architecture"]
                elif "database" in task_lower or "migration" in task_lower:
                    required_skills = ["database design", "programming"]
                elif "documentation" in task_lower or "user guides" in task_lower:
                    required_skills = ["writing", "documentation"]
                elif "monitoring" in task_lower:
                    required_skills = ["deployment", "monitoring"]
                elif "testing" in task_lower:
                    required_skills = ["testing", "quality assurance"]

                task = {
                    "task_id": f"task_{task_id_counter:03d}",
                    "title": task_title,
                    "description": f"{task_title} - {main_goal}",
                    "phase": phase,
                    "priority": priority,
                    "estimated_hours": estimated_hours,
                    "status": "pending",
                    "dependencies": [],
                    "resources_required": required_skills,
                    "deliverables": [f"Completed {task_title.lower()}"],
                    "acceptance_criteria": [
                        f"{task_title} meets quality standards",
                        f"Deliverables approved by stakeholders",
                        f"Documentation updated"
                    ]
                }
                all_tasks.append(task)
                task_id_counter += 1

        # Set up dependencies (tasks in later phases depend on earlier phases)
        for i, task in enumerate(all_tasks):
            if i > 0:
                # Current task depends on previous task in same phase, or last task of previous phase
                prev_task = all_tasks[i-1]
                if prev_task["phase"] != task["phase"]:
                    # Find the last task of the previous phase
                    prev_phase_tasks = [t for t in all_tasks[:i] if t["phase"] == prev_task["phase"]]
                    if prev_phase_tasks:
                        task["dependencies"] = [prev_phase_tasks[-1]["task_id"]]
                else:
                    task["dependencies"] = [prev_task["task_id"]]

        # Calculate total time and create plan
        total_hours = sum(task["estimated_hours"] for task in all_tasks)
        execution_order = [task["task_id"] for task in all_tasks]

        # Identify critical path (typically the longest sequential path)
        critical_path = []
        for phase_info in template:
            phase_tasks = [t for t in all_tasks if t["phase"] == phase_info["phase"]]
            if phase_tasks:
                critical_path.append(phase_tasks[0]["task_id"])  # First task of each phase

        task_plan = {
            "plan_id": f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "main_objective": main_goal,
            "project_domain": domain,
            "complexity_level": complexity,
            "tasks": all_tasks,
            "execution_order": execution_order,
            "total_estimated_hours": total_hours,
            "estimated_weeks": round(total_hours / (team_size * 40), 1),  # Assuming 40 hours/week per person
            "critical_path": critical_path,
            "phases": [phase_info["phase"] for phase_info in template],
            "risk_factors": analysis.get("potential_challenges", []),
            "success_metrics": analysis.get("success_metrics", [])
        }

        return json.dumps(task_plan, indent=2)

    except Exception as e:
        return f"Smart task generation error: {str(e)}"

@tool
def intelligent_resource_allocator(task_plan: str, team_info: str) -> str:
    """Intelligently allocate resources based on skills, availability, and task requirements"""
    try:
        plan_data = json.loads(task_plan)
        team_data = json.loads(team_info)

        tasks = plan_data.get("tasks", [])
        team_members = team_data.get("team_members", [])

        allocation_results = {
            "task_assignments": {},
            "resource_utilization": {},
            "timeline_optimization": {},
            "skill_gaps": [],
            "recommendations": [],
            "workload_balance": {}
        }

        # Initialize team member workloads
        for member in team_members:
            member_id = member.get("id", member.get("name", "unknown"))
            allocation_results["resource_utilization"][member_id] = {
                "total_hours": 0,
                "skills_used": set(),
                "assigned_tasks": [],
                "availability": member.get("availability", 40)  # hours per week
            }

        # Smart task assignment based on skills and availability
        unassigned_tasks = []

        for task in tasks:
            task_id = task["task_id"]
            required_skills = task.get("resources_required", [])
            estimated_hours = task.get("estimated_hours", 8)
            priority = task.get("priority", "medium")

            best_match = None
            best_score = 0

            # Find best team member for this task
            for member in team_members:
                member_id = member.get("id", member.get("name", "unknown"))
                member_skills = set(member.get("skills", []))
                current_workload = allocation_results["resource_utilization"][member_id]["total_hours"]
                max_capacity = member.get("availability", 40)

                # Calculate skill match score
                skill_overlap = len(set(required_skills) & member_skills)
                skill_score = skill_overlap / max(len(required_skills), 1)

                # Calculate availability score
                remaining_capacity = max_capacity - current_workload
                if remaining_capacity >= estimated_hours:
                    availability_score = 1.0
                elif remaining_capacity > 0:
                    availability_score = remaining_capacity / estimated_hours
                else:
                    availability_score = 0.0

                # Calculate priority bonus
                priority_bonus = {"high": 0.3, "medium": 0.1, "low": 0.0}.get(priority, 0.0)

                # Overall score
                total_score = (skill_score * 0.6) + (availability_score * 0.3) + priority_bonus

                if total_score > best_score and availability_score > 0:
                    best_match = member_id
                    best_score = total_score

            # Assign task or mark as unassigned
            if best_match:
                allocation_results["task_assignments"][task_id] = {
                    "assigned_to": best_match,
                    "match_score": round(best_score, 2),
                    "required_skills": required_skills,
                    "estimated_hours": estimated_hours,
                    "priority": priority,
                    "phase": task.get("phase", "Unknown")
                }

                # Update workload
                allocation_results["resource_utilization"][best_match]["total_hours"] += estimated_hours
                allocation_results["resource_utilization"][best_match]["skills_used"].update(required_skills)
                allocation_results["resource_utilization"][best_match]["assigned_tasks"].append(task_id)
            else:
                unassigned_tasks.append({
                    "task_id": task_id,
                    "title": task.get("title", ""),
                    "required_skills": required_skills,
                    "reason": "No available team member with matching skills"
                })

        # Convert sets to lists for JSON serialization
        for member_id in allocation_results["resource_utilization"]:
            allocation_results["resource_utilization"][member_id]["skills_used"] = list(
                allocation_results["resource_utilization"][member_id]["skills_used"]
            )

        # Analyze workload balance
        workloads = [info["total_hours"] for info in allocation_results["resource_utilization"].values()]
        if workloads:
            avg_workload = sum(workloads) / len(workloads)
            max_workload = max(workloads)
            min_workload = min(workloads)

            allocation_results["workload_balance"] = {
                "average_hours": round(avg_workload, 1),
                "max_hours": max_workload,
                "min_hours": min_workload,
                "balance_ratio": round(min_workload / max_workload if max_workload > 0 else 1, 2)
            }

        # Identify skill gaps
        all_required_skills = set()
        for task in tasks:
            all_required_skills.update(task.get("resources_required", []))

        available_skills = set()
        for member in team_members:
            available_skills.update(member.get("skills", []))

        missing_skills = all_required_skills - available_skills
        allocation_results["skill_gaps"] = list(missing_skills)

        # Generate recommendations
        recommendations = []

        if unassigned_tasks:
            recommendations.append(f"{len(unassigned_tasks)} tasks remain unassigned due to skill/capacity constraints")

        if allocation_results["workload_balance"]["balance_ratio"] < 0.7:
            recommendations.append("Workload is imbalanced - consider redistributing tasks")

        if missing_skills:
            recommendations.append(f"Consider hiring contractors or training for: {', '.join(missing_skills)}")

        if any(info["total_hours"] > info["availability"] for info in allocation_results["resource_utilization"].values()):
            recommendations.append("Some team members are over-allocated - extend timeline or add resources")

        allocation_results["recommendations"] = recommendations
        allocation_results["unassigned_tasks"] = unassigned_tasks

        # Timeline optimization suggestions
        critical_path_tasks = plan_data.get("critical_path", [])
        parallel_opportunities = []

        # Find tasks that can be done in parallel
        for task in tasks:
            if not task.get("dependencies") and task["task_id"] not in critical_path_tasks:
                parallel_opportunities.append(task["task_id"])

        allocation_results["timeline_optimization"] = {
            "parallel_opportunities": parallel_opportunities,
            "critical_path_focus": critical_path_tasks,
            "estimated_project_weeks": plan_data.get("estimated_weeks", "Unknown")
        }

        return json.dumps(allocation_results, indent=2)

    except Exception as e:
        return f"Resource allocation error: {str(e)}"

@tool
def dependency_analyzer(tasks: str) -> str:
    """Analyze task dependencies and create optimal execution sequence"""
    try:
        tasks_data = json.loads(tasks)
        tasks_list = tasks_data.get("tasks", [])

        analysis_results = {
            "dependency_graph": {},
            "execution_sequence": [],
            "parallel_groups": [],
            "bottlenecks": [],
            "critical_path": [],
            "estimated_completion": ""
        }

        # Build dependency graph
        for task in tasks_list:
            task_id = task["task_id"]
            dependencies = task.get("dependencies", [])
            analysis_results["dependency_graph"][task_id] = {
                "depends_on": dependencies,
                "title": task["title"],
                "duration": task.get("estimated_hours", 8),
                "priority": task.get("priority", "medium")
            }

        # Calculate execution sequence (topological sort simulation)
        remaining_tasks = set(task["task_id"] for task in tasks_list)
        execution_sequence = []
        parallel_groups = []

        while remaining_tasks:
            # Find tasks with no dependencies or completed dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                dependencies = analysis_results["dependency_graph"][task_id]["depends_on"]
                if all(dep in execution_sequence for dep in dependencies):
                    ready_tasks.append(task_id)

            if ready_tasks:
                # Group tasks that can run in parallel
                if len(ready_tasks) > 1:
                    parallel_groups.append(ready_tasks)

                execution_sequence.extend(ready_tasks)
                remaining_tasks -= set(ready_tasks)
            else:
                # Handle circular dependencies
                if remaining_tasks:
                    remaining_task = next(iter(remaining_tasks))
                    execution_sequence.append(remaining_task)
                    remaining_tasks.remove(remaining_task)

        analysis_results["execution_sequence"] = execution_sequence
        analysis_results["parallel_groups"] = parallel_groups

        # Identify critical path (longest path through dependencies)
        critical_path = []
        max_duration = 0

        for task_id in execution_sequence:
            task_info = analysis_results["dependency_graph"][task_id]
            if task_info["priority"] in ["high", "critical"]:
                critical_path.append(task_id)
                max_duration += task_info["duration"]

        analysis_results["critical_path"] = critical_path

        # Estimate completion time
        total_duration = max_duration
        completion_date = datetime.now() + timedelta(minutes=total_duration)
        analysis_results["estimated_completion"] = completion_date.strftime("%Y-%m-%d %H:%M")

        # Identify bottlenecks
        for task_id in execution_sequence:
            dependencies = analysis_results["dependency_graph"][task_id]["depends_on"]
            if len(dependencies) > 2:
                analysis_results["bottlenecks"].append(task_id)

        return json.dumps(analysis_results, indent=2)

    except Exception as e:
        return f"Dependency analysis error: {str(e)}"

@tool
def progress_tracker(completed_tasks: str, all_tasks: str) -> str:
    """Track progress and provide status updates"""
    try:
        completed = json.loads(completed_tasks) if completed_tasks else []
        all_tasks_data = json.loads(all_tasks)

        progress_report = {
            "completion_percentage": 0,
            "tasks_completed": len(completed),
            "tasks_remaining": 0,
            "estimated_time_remaining": 0,
            "current_status": "not_started",
            "next_tasks": [],
            "blocked_tasks": [],
            "recommendations": []
        }

        tasks_list = all_tasks_data.get("tasks", [])
        total_tasks = len(tasks_list)

        if total_tasks > 0:
            progress_report["completion_percentage"] = (len(completed) / total_tasks) * 100
            progress_report["tasks_remaining"] = total_tasks - len(completed)

            # Calculate remaining time
            remaining_time = 0
            for task in tasks_list:
                if task["task_id"] not in completed:
                    remaining_time += task.get("estimated_hours", 8)

            progress_report["estimated_time_remaining"] = remaining_time

            # Determine current status
            if len(completed) == 0:
                progress_report["current_status"] = "not_started"
            elif len(completed) == total_tasks:
                progress_report["current_status"] = "completed"
            else:
                progress_report["current_status"] = "in_progress"

            # Find next available tasks
            for task in tasks_list:
                if task["task_id"] not in completed:
                    dependencies = task.get("dependencies", [])
                    if all(dep in completed for dep in dependencies):
                        progress_report["next_tasks"].append({
                            "task_id": task["task_id"],
                            "title": task["title"],
                            "priority": task.get("priority", "medium")
                        })

            # Find blocked tasks
            for task in tasks_list:
                if task["task_id"] not in completed:
                    dependencies = task.get("dependencies", [])
                    if dependencies and not all(dep in completed for dep in dependencies):
                        missing_deps = [dep for dep in dependencies if dep not in completed]
                        progress_report["blocked_tasks"].append({
                            "task_id": task["task_id"],
                            "title": task["title"],
                            "missing_dependencies": missing_deps
                        })

            # Generate recommendations
            completion_rate = progress_report["completion_percentage"]
            if completion_rate < 25:
                progress_report["recommendations"].append("Focus on completing initial setup tasks")
            elif completion_rate < 75:
                progress_report["recommendations"].append("Maintain steady progress on core tasks")
            else:
                progress_report["recommendations"].append("Prioritize final deliverables and quality checks")

        return json.dumps(progress_report, indent=2)

    except Exception as e:
        return f"Progress tracking error: {str(e)}"

# Task Planning Functions
def create_planning_llm(provider: str, model: str, **kwargs):
    """Create LLM instance for task planning with configurable parameters"""
    temperature = kwargs.get('temperature', 0.3)

    if provider == "Ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=kwargs.get('base_url', "http://localhost:11434"),
            timeout=kwargs.get('timeout', 120)
        )
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature
        )
    elif provider == "Groq":
        return ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature
        )
    elif provider == "Anthropic":
        return ChatAnthropic(
            model=model,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature
        )
    elif provider == "OpenAI":
        return ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature
        )

def project_analysis_node(state: TaskPlanningState, llm) -> TaskPlanningState:
    """Analyze project goal using AI and intelligent analysis tools"""
    try:
        main_goal = state["main_goal"]
        project_context = state["execution_context"].get("requirements", "")

        # Use intelligent task analyzer tool
        project_analysis_result = intelligent_task_analyzer.invoke({
            "main_goal": main_goal,
            "project_context": project_context
        })

        # Get AI analysis for additional insights
        analysis_prompt = f"""
        Based on this project goal: "{main_goal}"

        And this context: "{project_context}"

        Provide strategic planning insights:
        1. Key success factors and potential roadblocks
        2. Stakeholder considerations and communication needs
        3. Quality standards and deliverable requirements
        4. Timeline risks and mitigation strategies
        5. Resource optimization recommendations

        Be specific and actionable in your recommendations.
        """

        ai_response = llm.invoke([
            SystemMessage(content="You are an expert project strategist and planning consultant."),
            HumanMessage(content=analysis_prompt)
        ])

        # Store both analysis results
        state["execution_context"]["project_analysis"] = json.loads(project_analysis_result)
        state["execution_context"]["ai_insights"] = ai_response.content
        state["planning_stage"] = "project_analyzed"
        state["next_action"] = "generate_tasks"

        return state

    except Exception as e:
        state["execution_context"]["errors"] = state["execution_context"].get("errors", []) + [f"Project analysis error: {str(e)}"]
        state["next_action"] = "error_handler"
        return state

def smart_task_generation_node(state: TaskPlanningState) -> TaskPlanningState:
    """Generate intelligent, realistic tasks based on project analysis"""
    try:
        main_goal = state["main_goal"]
        project_analysis = json.dumps(state["execution_context"]["project_analysis"])
        team_size = len(state["available_resources"].get("team_members", [{"id": "member_1"}, {"id": "member_2"}, {"id": "member_3"}]))

        # Use smart task generator tool
        task_generation_result = smart_task_generator.invoke({
            "main_goal": main_goal,
            "project_analysis": project_analysis,
            "team_size": team_size
        })

        task_plan_data = json.loads(task_generation_result)
        state["task_plan"] = task_plan_data
        state["planning_stage"] = "tasks_generated"
        state["next_action"] = "allocate_resources"

        return state

    except Exception as e:
        state["execution_context"]["errors"] = state["execution_context"].get("errors", []) + [f"Task generation error: {str(e)}"]
        state["next_action"] = "error_handler"
        return state

def dependency_analysis_node(state: TaskPlanningState) -> TaskPlanningState:
    """Analyze task dependencies and create execution sequence"""
    try:
        tasks_json = json.dumps(state["task_plan"])

        # Use dependency analyzer tool
        dependency_result = dependency_analyzer.invoke({"tasks": tasks_json})
        dependency_data = json.loads(dependency_result)

        state["execution_context"]["dependency_analysis"] = dependency_data
        state["planning_stage"] = "dependencies_analyzed"
        state["next_action"] = "allocate_resources"

        return state

    except Exception as e:
        state["execution_context"]["errors"] = state["execution_context"].get("errors", []) + [f"Dependency analysis error: {str(e)}"]
        state["next_action"] = "error_handler"
        return state

def intelligent_resource_allocation_node(state: TaskPlanningState) -> TaskPlanningState:
    """Intelligently allocate resources based on skills and availability"""
    try:
        task_plan_json = json.dumps(state["task_plan"])
        team_info_json = json.dumps(state["available_resources"])

        # Use intelligent resource allocator tool
        allocation_result = intelligent_resource_allocator.invoke({
            "task_plan": task_plan_json,
            "team_info": team_info_json
        })

        allocation_data = json.loads(allocation_result)
        state["execution_context"]["resource_allocation"] = allocation_data
        state["planning_stage"] = "resources_allocated"
        state["next_action"] = "create_execution_plan"

        return state

    except Exception as e:
        state["execution_context"]["errors"] = state["execution_context"].get("errors", []) + [f"Resource allocation error: {str(e)}"]
        state["next_action"] = "error_handler"
        return state

def execution_planner(state: TaskPlanningState, llm) -> TaskPlanningState:
    """Create final execution plan with timeline"""
    try:
        task_plan = state["task_plan"]
        resource_allocation = state["execution_context"].get("resource_allocation", {})
        project_analysis = state["execution_context"].get("project_analysis", {})
        ai_insights = state["execution_context"].get("ai_insights", "")

        # Extract key information for planning
        total_hours = task_plan.get("total_estimated_hours", 0)
        estimated_weeks = task_plan.get("estimated_weeks", 0)
        project_domain = task_plan.get("project_domain", "General")
        complexity = task_plan.get("complexity_level", "medium")
        phases = task_plan.get("phases", [])
        critical_path = task_plan.get("critical_path", [])
        risk_factors = task_plan.get("risk_factors", [])

        planning_prompt = f"""
        Create a comprehensive, actionable execution plan for this project:

        **Project Overview:**
        - Main Objective: {state["main_goal"]}
        - Domain: {project_domain}
        - Complexity: {complexity}
        - Estimated Duration: {estimated_weeks} weeks ({total_hours} hours)
        - Project Phases: {', '.join(phases)}

        **Key Project Data:**
        - Critical Path Tasks: {', '.join(critical_path)}
        - Risk Factors: {', '.join(risk_factors)}
        - Success Metrics: {project_analysis.get("success_metrics", [])}

        **Team & Resource Summary:**
        - Team Assignments: {len(resource_allocation.get("task_assignments", {}))} tasks assigned
        - Resource Utilization: {resource_allocation.get("workload_balance", {})}
        - Recommendations: {resource_allocation.get("recommendations", [])}

        **Additional Context:**
        {ai_insights}

        Please generate a detailed execution plan with:

        ## 📅 **Execution Timeline**
        - Phase-by-phase timeline with milestones
        - Critical path management strategy
        - Parallel work opportunities

        ## 🎯 **Key Milestones & Deliverables**
        - Major checkpoints and deliverables for each phase
        - Success criteria and acceptance gates

        ## ⚠️ **Risk Management**
        - Risk mitigation strategies for identified challenges
        - Contingency plans for critical path delays

        ## 📊 **Progress Tracking**
        - KPIs and metrics to monitor
        - Reporting frequency and stakeholder communication

        ## 🚀 **Success Strategy**
        - Priority focus areas
        - Team coordination approach
        - Quality assurance plan

        Make this practical and actionable for a real project team.
        """

        # Use synchronous invoke instead of async ainvoke
        response = llm.invoke([
            SystemMessage(content="You are an expert project manager and consultant creating detailed, actionable execution plans for real-world projects."),
            HumanMessage(content=planning_prompt)
        ])

        state["execution_context"]["execution_plan"] = response.content
        state["planning_stage"] = "plan_complete"
        state["next_action"] = "complete"

        return state

    except Exception as e:
        state["execution_context"]["errors"] = state["execution_context"].get("errors", []) + [f"Execution planning error: {str(e)}"]
        state["next_action"] = "error_handler"
        return state

def planning_router(state: TaskPlanningState) -> str:
    """Route planning workflow based on current state"""
    next_action = state.get("next_action", "generate_tasks")

    if next_action == "generate_tasks":
        return "smart_task_generation"
    elif next_action == "allocate_resources":
        return "intelligent_resource_allocation"
    elif next_action == "create_execution_plan":
        return "execution_planner"
    elif next_action == "complete":
        return END
    else:
        return "smart_task_generation"

def create_task_planning_graph(llm):
    """Create the intelligent task planning workflow graph"""

    # Create workflow
    workflow = StateGraph(TaskPlanningState)

    # Add nodes with improved functionality
    workflow.add_node("project_analysis", lambda state: project_analysis_node(state, llm))
    workflow.add_node("smart_task_generation", smart_task_generation_node)
    workflow.add_node("intelligent_resource_allocation", intelligent_resource_allocation_node)
    workflow.add_node("execution_planner", lambda state: execution_planner(state, llm))

    # Add edges for improved workflow
    workflow.add_edge(START, "project_analysis")
    workflow.add_conditional_edges("project_analysis", planning_router)
    workflow.add_conditional_edges("smart_task_generation", planning_router)
    workflow.add_conditional_edges("intelligent_resource_allocation", planning_router)
    workflow.add_edge("execution_planner", END)

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def render_task_planning_interface():
    """Render the Streamlit interface for Intelligent Task Planning System"""
    st.header("🧠 Intelligent Task Planning & Execution System")
    st.markdown("*AI-powered project planning with realistic task breakdown, smart resource allocation and timeline optimization*")

    # Sidebar configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        # LLM Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["Gemini", "Ollama", "Groq","Anthropic", "OpenAI"],
            key='planning_llm_provider'
        )

        model_options = {
            # Ollama: Free Open Models, runs on your local system (no API key required)
            "Ollama": ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "gemma2:2b", "gemma2:9b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "codestral:22b", "deepseek-coder:1.3b"],
            # Gemini: Google's Gemini models (requires API key)
            "Gemini": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
            # Groq: Open Models (requires API key)
            "Groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "openai/gpt-oss-120b"],
            # Anthropic: Claude AI models, good at reasoning & Coding (requires API key)
            "Anthropic": ["claude-sonnet-4-20250514", "claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"],
            # OpenAI: ChatGPT and GPT models, good at reasoning(requires API key)
            "OpenAI": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-5", "gpt-5-mini", "gpt-5-nano"]
        }

        model = st.selectbox(
            "Model",
            model_options[llm_provider],
            key='planning_model'
        )

        # Ollama-specific configuration
        ollama_base_url = None
        if llm_provider == "Ollama":
            st.markdown("**🔧 Ollama Configuration**")
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                key='planning_ollama_url',
                help="URL where Ollama server is running"
            )

            # Check Ollama status
            try:
                import requests
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    st.success("✅ Ollama server is running")
                else:
                    st.error("❌ Ollama server not accessible")
            except Exception as e:
                st.error("❌ Cannot connect to Ollama server")
                st.markdown("**Setup Instructions:**")
                st.code(f"1. Install Ollama from https://ollama.com\n2. Run: ollama serve\n3. Pull model: ollama pull {model}")


        st.markdown("### 📊 Quick Setup")
        use_default_team = st.checkbox("Use default team setup", value=True, key='use_default')


    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 🎯 Project Definition")

        main_goal = st.text_area(
            "Project Goal & Description",
            placeholder="e.g., Develop a mobile app for task management with user authentication, real-time sync, and offline capabilities...",
            height=100,
            key='main_goal'
        )

        project_context = st.text_area(
            "Additional Context & Requirements",
            placeholder="e.g., Target audience: small businesses, Budget: $50k, Timeline: 3 months, Must integrate with existing systems...",
            height=80,
            key='project_context'
        )

    with col2:
        st.markdown("#### ⚙️ Project Settings")

        project_type = st.selectbox(
            "Project Type",
            ["Auto-detect", "Software Development", "Marketing Campaign", "Research Project", "Event Management", "General"],
            key='project_type'
        )

        urgency = st.selectbox(
            "Urgency Level",
            ["Low", "Medium", "High", "Critical"],
            index=1,
            key='urgency'
        )

        budget_range = st.selectbox(
            "Budget Range",
            ["Under $10K", "$10K - $50K", "$50K - $100K", "$100K - $500K", "Over $500K", "Not specified"],
            index=5,
            key='budget_range'
        )

    # Team Configuration Section
    st.markdown("#### 👥 Team Configuration")

    if not use_default_team:
        st.markdown("**Custom Team Setup**")

        num_team_members = st.number_input("Number of team members", min_value=1, max_value=20, value=3, key='num_members')

        team_members = []
        for i in range(num_team_members):
            with st.expander(f"Team Member {i+1}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input(f"Name", value=f"Team Member {i+1}", key=f'member_name_{i}')
                    role = st.text_input(f"Role/Title", placeholder="e.g., Developer, Designer, Analyst", key=f'member_role_{i}')
                with col2:
                    availability = st.number_input(f"Hours/week", min_value=1, max_value=60, value=40, key=f'member_hours_{i}')

                skills = st.multiselect(
                    f"Skills",
                    ["programming", "testing", "design", "project management", "data analysis", "marketing",
                     "writing", "research", "communication", "leadership", "technical architecture",
                     "user experience", "quality assurance", "deployment", "database design", "documentation",
                     "monitoring", "devops", "system administration"],
                    key=f'member_skills_{i}'
                )

                team_members.append({
                    "id": f"member_{i+1}",
                    "name": name,
                    "role": role,
                    "skills": skills,
                    "availability": availability
                })
    else:
        # Default balanced team
        team_size = st.selectbox("Team Size", ["Small (3 people)", "Medium (5 people)", "Large (8 people)"], index=0, key='team_size')

        if team_size == "Small (3 people)":
            team_members = [
                {"id": "lead", "name": "Project Lead", "role": "Team Lead", "skills": ["project management", "leadership", "communication", "writing"], "availability": 40},
                {"id": "dev", "name": "Developer", "role": "Developer/Analyst", "skills": ["programming", "testing", "technical architecture", "deployment", "database design"], "availability": 40},
                {"id": "designer", "name": "Designer", "role": "Designer/Researcher", "skills": ["design", "research", "user experience", "writing"], "availability": 40}
            ]
        elif team_size == "Medium (5 people)":
            team_members = [
                {"id": "lead", "name": "Project Lead", "role": "Project Manager", "skills": ["project management", "leadership", "writing"], "availability": 40},
                {"id": "senior_dev", "name": "Senior Developer", "role": "Senior Developer", "skills": ["programming", "technical architecture", "deployment", "database design"], "availability": 40},
                {"id": "dev", "name": "Developer", "role": "Developer", "skills": ["programming", "testing", "deployment"], "availability": 40},
                {"id": "designer", "name": "Designer", "role": "UX/UI Designer", "skills": ["design", "user experience", "writing"], "availability": 40},
                {"id": "analyst", "name": "Analyst", "role": "Business Analyst", "skills": ["data analysis", "research", "writing"], "availability": 40}
            ]
        else:  # Large team
            team_members = [
                {"id": "pm", "name": "Project Manager", "role": "Project Manager", "skills": ["project management", "leadership", "writing"], "availability": 40},
                {"id": "tech_lead", "name": "Tech Lead", "role": "Technical Lead", "skills": ["programming", "technical architecture", "leadership", "deployment"], "availability": 40},
                {"id": "senior_dev", "name": "Senior Developer", "role": "Senior Developer", "skills": ["programming", "deployment", "database design"], "availability": 40},
                {"id": "dev1", "name": "Developer 1", "role": "Frontend Developer", "skills": ["programming", "design"], "availability": 40},
                {"id": "dev2", "name": "Developer 2", "role": "Backend Developer", "skills": ["programming", "database design", "deployment"], "availability": 40},
                {"id": "qa", "name": "QA Engineer", "role": "QA Engineer", "skills": ["testing", "quality assurance"], "availability": 40},
                {"id": "designer", "name": "UX Designer", "role": "UX/UI Designer", "skills": ["design", "user experience", "writing"], "availability": 40},
                {"id": "analyst", "name": "Business Analyst", "role": "Business Analyst", "skills": ["data analysis", "research", "writing"], "availability": 40}
            ]

    # Display team summary
    if team_members:
        with st.expander("👥 Team Summary", expanded=False):
            total_hours = sum(member["availability"] for member in team_members)
            all_skills = set()
            for member in team_members:
                all_skills.update(member["skills"])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Team Size", len(team_members))
            with col2:
                st.metric("Total Hours/Week", f"{total_hours}h")
            with col3:
                st.metric("Unique Skills", len(all_skills))

            st.markdown("**Available Skills:**")
            st.write(", ".join(sorted(all_skills)))

    # Advanced Configuration Section
    with st.expander("⚙️ Advanced Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🎛️ Model Settings**")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                key='planning_temperature',
                help="Controls planning creativity and variability"
            )

        with col2:
            st.markdown("**📋 Planning Settings**")
            planning_depth = st.selectbox(
                "Planning Depth",
                ["Basic", "Detailed", "Comprehensive"],
                index=1,
                key='planning_depth',
                help="Level of detail in task breakdown"
            )

            enable_dependencies = st.checkbox(
                "Track Dependencies",
                value=True,
                key='enable_dependencies',
                help="Track task dependencies and ordering"
            )

        with col3:
            st.markdown("**🔍 Analysis Options**")
            auto_scheduling = st.checkbox(
                "Auto-Scheduling",
                value=True,
                key='auto_scheduling',
                help="Automatically schedule tasks with timelines"
            )

            risk_assessment = st.checkbox(
                "Risk Assessment",
                value=False,
                key='risk_assessment',
                help="Include risk analysis in planning"
            )

    col1, col2 = st.columns(2)

    with col1:
        project_deadline = st.date_input("Target Completion Date", key='deadline')
        priority_level = st.selectbox("Priority Level", ["low", "medium", "high", "critical"], index=2, key='priority')

    with col2:
        budget_constraint = st.text_input("Budget Constraints (optional)", placeholder="e.g., $10,000, 100 hours", key='budget')
    # Execute task planning
    if st.button("Create Intelligent Task Plan", type="primary", key='create_plan'):
        if not main_goal:
            st.error("Please provide a project goal and description.")
            return

        if not team_members:
            st.error("Please configure your team members.")
            return

        # Prepare team and project resources
        available_resources = {
            "team_members": team_members,
            "project_settings": {
                "type": project_type,
                "urgency": urgency,
                "budget_range": budget_range
            }
        }

        # Initialize planning state
        initial_state = {
            "messages": [],
            "main_goal": main_goal,
            "task_plan": {},
            "current_task": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "available_resources": available_resources,
            "execution_context": {
                "requirements": project_context,
                "urgency": urgency,
                "budget_range": budget_range,
                "project_type": project_type
            },
            "planning_stage": "initialized",
            "next_action": "generate_tasks"
        }

        with st.spinner("🧠 Creating intelligent task plan with AI analysis..."):
            try:
                # Create LLM and graph with configuration
                llm_kwargs = {
                    'temperature': temperature
                }
                if llm_provider == "Ollama" and ollama_base_url:
                    llm_kwargs['base_url'] = ollama_base_url
                llm = create_planning_llm(llm_provider, model, **llm_kwargs)
                planning_graph = create_task_planning_graph(llm)

                # Execute planning workflow
                config = {"configurable": {"thread_id": f"plan_{datetime.now().timestamp()}"}}
                result = planning_graph.invoke(initial_state, config)

                # Display results
                st.success("✅ Intelligent task plan created successfully!")

                # Project Analysis Summary
                project_analysis = result["execution_context"].get("project_analysis", {})
                if project_analysis:
                    with st.expander("🔍 Project Analysis Summary", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Project Domain", project_analysis.get("domain", "General").title())
                        with col2:
                            st.metric("Complexity", project_analysis.get("complexity", "Medium").title())
                        with col3:
                            st.metric("Duration", project_analysis.get("estimated_duration", "Unknown"))

                        # Key insights
                        if project_analysis.get("key_skills_required"):
                            st.markdown("**🎯 Key Skills Required:**")
                            st.write(", ".join(project_analysis["key_skills_required"]))

                        if project_analysis.get("potential_challenges"):
                            st.markdown("**⚠️ Potential Challenges:**")
                            for challenge in project_analysis["potential_challenges"]:
                                st.write(f"• {challenge}")

                        if project_analysis.get("success_metrics"):
                            st.markdown("**📊 Success Metrics:**")
                            for metric in project_analysis["success_metrics"]:
                                st.write(f"• {metric}")

                # Results tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Tasks & Phases", "👥 Team Assignments", "📊 Resource Analysis", "⏰ Timeline & Critical Path", "🎯 Execution Strategy"])

                with tab1:
                    st.markdown("### 📋 Task Breakdown")

                    task_plan = result["task_plan"]
                    if task_plan and "tasks" in task_plan:
                        tasks = task_plan["tasks"]

                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Tasks", len(tasks))
                        with col2:
                            total_hours = sum(task.get("estimated_hours", 0) for task in tasks)
                            st.metric("Total Hours", f"{total_hours}h")
                        with col3:
                            estimated_weeks = task_plan.get("estimated_weeks", 0)
                            st.metric("Estimated Duration", f"{estimated_weeks} weeks")
                        with col4:
                            high_priority = len([t for t in tasks if t.get("priority") in ["high", "critical"]])
                            st.metric("High Priority Tasks", high_priority)

                        # Group tasks by phase
                        phases = {}
                        for task in tasks:
                            phase = task.get("phase", "General")
                            if phase not in phases:
                                phases[phase] = []
                            phases[phase].append(task)

                        # Display tasks by phase
                        for phase, phase_tasks in phases.items():
                            st.markdown(f"#### 📁 {phase}")
                            phase_hours = sum(task.get("estimated_hours", 0) for task in phase_tasks)
                            st.caption(f"{len(phase_tasks)} tasks • {phase_hours} hours estimated")

                            for task in phase_tasks:
                                with st.expander(f"🔹 {task['title']}", expanded=False):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Description:** {task['description']}")
                                        st.write(f"**Priority:** {task['priority'].title()}")
                                        if task.get('dependencies'):
                                            st.write(f"**Dependencies:** {', '.join(task['dependencies'])}")
                                    with col2:
                                        st.write(f"**Estimated Hours:** {task['estimated_hours']}h")
                                        st.write(f"**Required Skills:** {', '.join(task.get('resources_required', []))}")

                                        if task.get('deliverables'):
                                            st.write("**Deliverables:**")
                                            for deliverable in task['deliverables']:
                                                st.write(f"• {deliverable}")

                            st.markdown("---")

                with tab2:
                    st.markdown("### 👥 Team Assignments")
                    resource_allocation = result["execution_context"].get("resource_allocation", {})

                    if resource_allocation:
                        # Task assignments
                        assignments = resource_allocation.get("task_assignments", {})
                        utilization = resource_allocation.get("resource_utilization", {})

                        if assignments:
                            st.markdown("#### 📋 Task Assignments")

                            # Group assignments by team member
                            assignments_by_member = {}
                            for task_id, assignment in assignments.items():
                                member = assignment["assigned_to"]
                                if member not in assignments_by_member:
                                    assignments_by_member[member] = []
                                assignments_by_member[member].append((task_id, assignment))

                            for member, member_assignments in assignments_by_member.items():
                                member_info = utilization.get(member, {})
                                total_hours = member_info.get("total_hours", 0)
                                skills_used = member_info.get("skills_used", [])

                                with st.expander(f"👤 {member} ({total_hours}h assigned)", expanded=False):
                                    if skills_used:
                                        st.write(f"**Skills Used:** {', '.join(skills_used)}")

                                    st.markdown("**Assigned Tasks:**")
                                    for task_id, assignment in member_assignments:
                                        match_score = assignment.get("match_score", 0)
                                        hours = assignment.get("estimated_hours", 0)
                                        priority = assignment.get("priority", "medium")
                                        phase = assignment.get("phase", "Unknown")

                                        st.write(f"• **{task_id}** ({phase}) - {hours}h - {priority.title()} priority - Match: {match_score}")

                        # Workload balance analysis
                        workload_balance = resource_allocation.get("workload_balance", {})
                        if workload_balance:
                            st.markdown("#### ⚖️ Workload Balance")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average Hours", f"{workload_balance.get('average_hours', 0)}h")
                            with col2:
                                st.metric("Max Workload", f"{workload_balance.get('max_hours', 0)}h")
                            with col3:
                                balance_ratio = workload_balance.get('balance_ratio', 1)
                                st.metric("Balance Score", f"{balance_ratio:.2f}", help="1.0 = perfectly balanced")

                        # Skill gaps and recommendations
                        skill_gaps = resource_allocation.get("skill_gaps", [])
                        if skill_gaps:
                            st.markdown("#### ⚠️ Skill Gaps")
                            st.warning(f"Missing skills: {', '.join(skill_gaps)}")

                        recommendations = resource_allocation.get("recommendations", [])
                        if recommendations:
                            st.markdown("#### 💡 Recommendations")
                            for rec in recommendations:
                                st.info(f"• {rec}")

                        unassigned_tasks = resource_allocation.get("unassigned_tasks", [])
                        if unassigned_tasks:
                            st.markdown("#### 🚨 Unassigned Tasks")
                            for task in unassigned_tasks:
                                st.error(f"• **{task['task_id']}**: {task['title']} - {task['reason']}")

                with tab3:
                    st.markdown("### 👥 Resource Allocation")
                    resource_allocation = result["execution_context"].get("resource_allocation", {})

                    if resource_allocation:
                        allocations = resource_allocation.get("allocations", {})

                        # Resource utilization
                        utilization = resource_allocation.get("resource_utilization", {})
                        if utilization:
                            st.markdown("**Team Workload:**")
                            for agent, workload_info in utilization.items():
                                if isinstance(workload_info, dict):
                                    total_hours = workload_info.get("total_hours", 0)
                                    st.write(f"• {agent}: {total_hours}h")
                                else:
                                    # Handle old format if any
                                    hours = workload_info // 60 if isinstance(workload_info, (int, float)) else workload_info
                                    st.write(f"• {agent}: {hours}h")

                        # Task assignments
                        if allocations:
                            st.markdown("**Task Assignments:**")
                            for task_id, allocation in allocations.items():
                                agent = allocation.get("assigned_agent", "Unassigned")
                                priority = allocation.get("priority", "medium")
                                st.write(f"• {task_id}: {agent} (Priority: {priority})")

                        # Conflicts and recommendations
                        conflicts = resource_allocation.get("conflicts", [])
                        if conflicts:
                            st.markdown("**Resource Conflicts:**")
                            for conflict in conflicts:
                                st.warning(conflict)

                        recommendations = resource_allocation.get("recommendations", [])
                        if recommendations:
                            st.markdown("**Recommendations:**")
                            for rec in recommendations:
                                st.info(rec)

                with tab4:
                    st.markdown("### 📅 Execution Plan")
                    execution_plan = result["execution_context"].get("execution_plan", "")

                    if execution_plan:
                        st.markdown(execution_plan)
                    else:
                        st.info("Detailed execution plan not available.")

                    # Progress tracking simulation
                    if st.button("🔄 Simulate Progress Tracking", key='simulate_progress'):
                        completed_tasks = ["task_001", "task_002"]  # Simulate some completed tasks
                        tasks_json = json.dumps(result["task_plan"])

                        progress_result = progress_tracker.invoke({
                            "completed_tasks": json.dumps(completed_tasks),
                            "all_tasks": tasks_json
                        })

                        progress_data = json.loads(progress_result)

                        st.markdown("**Progress Update:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Progress", f"{progress_data['completion_percentage']:.1f}%")
                        with col2:
                            st.metric("Completed", progress_data['tasks_completed'])
                        with col3:
                            st.metric("Remaining", progress_data['tasks_remaining'])

                        # Next tasks
                        next_tasks = progress_data.get("next_tasks", [])
                        if next_tasks:
                            st.markdown("**Next Available Tasks:**")
                            for task in next_tasks:
                                st.write(f"• {task['title']} (Priority: {task['priority']})")

            except Exception as e:
                st.error(f"Task planning error: {str(e)}")

    # Task planning info
    with st.expander("ℹ️ Task Planning System Features", expanded=False):
        st.markdown("""
        **Intelligent Task Planning & Execution:**

        🧠 **Smart Decomposition**
        - Automatic goal breakdown into manageable tasks
        - Context-aware task generation based on project type
        - Intelligent priority and duration estimation

        🔗 **Dependency Management**
        - Automatic dependency detection and analysis
        - Critical path identification
        - Parallel execution optimization

        👥 **Resource Optimization**
        - Intelligent resource allocation and scheduling
        - Workload balancing across team members
        - Conflict detection and resolution

        📊 **Progress Tracking**
        - Real-time progress monitoring
        - Bottleneck identification
        - Adaptive replanning based on results

        📅 **Execution Planning**
        - Comprehensive timeline creation
        - Milestone and checkpoint definition
        - Risk assessment and mitigation strategies
        """)

if __name__ == "__main__":
    render_task_planning_interface()