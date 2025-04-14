"""
Medical Tools Module
This module provides a collection of medical tools for patient management,
including record updates, appointments, lab tests, and specialist referrals.
"""
import os
import inspect
import json
import logging
from datetime import datetime
from functools import wraps
from typing import Dict, Optional, Type, ClassVar, Callable, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOGGING_LEVEL"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_tool_execution(func: Callable) -> Callable:
    """
    Decorator to log tool execution details.
    
    Args:
        func (Callable): The function to be decorated
        
    Returns:
        Callable: Decorated function with logging
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = func.__name__
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool arguments: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"Tool {tool_name} executed successfully")
            logger.debug(f"Tool result: {result}")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {str(e)}")
            raise
            
    return wrapper

class MedicalToolError(Exception):
    """Custom exception for medical tool errors."""
    pass

# Basic Tool Functions
@log_tool_execution
def update_patient_record(patient_id: str, data: dict) -> str:
    """
    Update patient record with new data.
    
    Args:
        patient_id (str): Unique identifier for the patient
        data (dict): New data to update in the patient record
        
    Returns:
        str: Confirmation message
    """
    logger.info(f"Updating record for patient {patient_id}")
    return f"Updated patient {patient_id} with {data}"

@log_tool_execution
def schedule_appointment(patient_id: str, department: str, urgency: str) -> str:
    """
    Schedule patient appointment.
    
    Args:
        patient_id (str): Unique identifier for the patient
        department (str): Medical department for the appointment
        urgency (str): Urgency level of the appointment
        
    Returns:
        str: Confirmation message
    """
    logger.info(f"Scheduling {urgency} appointment for patient {patient_id}")
    return f"Scheduled {urgency} appointment for patient {patient_id} with {department}"

@log_tool_execution
def order_lab_test(patient_id: str, test_type: str) -> str:
    """
    Order laboratory test for patient.
    
    Args:
        patient_id (str): Unique identifier for the patient
        test_type (str): Type of laboratory test to order
        
    Returns:
        str: Confirmation message
    """
    logger.info(f"Ordering {test_type} test for patient {patient_id}")
    return f"Ordered {test_type} for patient {patient_id}"

@log_tool_execution
def refer_specialist(patient_id: str, specialty: str) -> str:
    """
    Refer patient to specialist.
    
    Args:
        patient_id (str): Unique identifier for the patient
        specialty (str): Medical specialty for referral
        
    Returns:
        str: Confirmation message
    """
    logger.info(f"Referring patient {patient_id} to {specialty}")
    return f"Referred patient {patient_id} to {specialty}"

def get_tool_map() -> Dict[str, Callable]:
    """
    Get mapping of tool names to functions.
    
    Returns:
        Dict[str, callable]: Dictionary mapping tool names to their functions
    """
    return {
        "update_patient_record": update_patient_record,
        "schedule_appointment": schedule_appointment,
        "order_lab_test": order_lab_test,
        "refer_specialist": refer_specialist
    }

def execute_medical_tool(tool_name: str, args: dict) -> str:
    """
    Execute selected medical tool function with validation.
    
    Args:
        tool_name (str): Name of the tool to execute
        args (dict): Arguments for the tool
        
    Returns:
        str: Tool execution result
        
    Raises:
        MedicalToolError: If tool validation or execution fails
    """
    logger.info(f"Attempting to execute tool: {tool_name}")
    tool_map = get_tool_map()
    
    try:
        # Validate tool existence
        if tool_name not in tool_map:
            raise MedicalToolError(f"Unknown tool: {tool_name}")
        
        # Validate patient_id presence
        if "patient_id" not in args:
            raise MedicalToolError("patient_id is required for all medical tools")
            
        # Get the selected function
        func = tool_map[tool_name]
        
        # Get the function's parameter names
        params = inspect.signature(func).parameters
        
        # Validate all required parameters are present
        missing_params = [param for param in params if param not in args]
        if missing_params:
            raise MedicalToolError(
                f"Missing required parameters for {tool_name}: {missing_params}"
            )
        
        # Execute the function with validated arguments
        logger.debug(f"Executing {tool_name} with args: {args}")
        result = func(**args)
        logger.info(f"Successfully executed {tool_name}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to execute {tool_name}: {str(e)}")
        raise MedicalToolError(f"Tool execution failed: {str(e)}")

# Pydantic Schemas
class UpdatePatientSchema(BaseModel):
    """Schema for updating patient records."""
    patient_id: str = Field(..., description="Patient identifier")
    state: str = Field(..., description="Current patient state")
    notes: str = Field(..., description="Clinical notes")
    vitals: Optional[Dict[str, float]] = Field(None, description="Vital signs")
    
    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "patient_id": "P12345",
                "state": "stable",
                "notes": "Patient responding well to treatment",
                "vitals": {"heart_rate": 72, "blood_pressure": 120/80}
            }
        }

class OrderTestSchema(BaseModel):
    """Schema for ordering medical tests."""
    patient_id: str = Field(..., description="Patient identifier")
    test_type: str = Field(..., description="Type of test to order")
    priority: str = Field(..., description="Test priority (routine/urgent/stat)")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "patient_id": "P12345",
                "test_type": "complete_blood_count",
                "priority": "urgent",
                "notes": "Check for infection markers"
            }
        }

class ReferralSchema(BaseModel):
    """Schema for specialist referrals."""
    patient_id: str = Field(..., description="Patient identifier")
    specialty: str = Field(..., description="Medical specialty")
    priority: str = Field(..., description="Referral priority")
    reason: str = Field(..., description="Reason for referral")
    
    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "patient_id": "P12345",
                "specialty": "cardiology",
                "priority": "urgent",
                "reason": "Abnormal ECG findings"
            }
        }

# LangChain Tool Implementations
class UpdatePatientTool(BaseTool):
    """Tool for updating patient records."""
    name: ClassVar[str] = "update_patient"
    description: ClassVar[str] = "Update patient status and medical record"
    args_schema: ClassVar[Type[BaseModel]] = UpdatePatientSchema
    
    def _run(self, patient_id: str, state: str, notes: str, 
             vitals: Optional[Dict[str, float]] = None) -> str:
        """Execute the update patient tool."""
        logger.info(f"Running UpdatePatientTool for patient {patient_id}")
        update = {
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "notes": notes,
            "vitals": vitals
        }
        return f"Updated patient {patient_id} record: {json.dumps(update)}"

class OrderTestTool(BaseTool):
    """Tool for ordering medical tests."""
    name: ClassVar[str] = "order_test"
    description: ClassVar[str] = "Order medical tests for patient"
    args_schema: ClassVar[Type[BaseModel]] = OrderTestSchema
    
    def _run(self, patient_id: str, test_type: str, priority: str, 
             notes: Optional[str] = None) -> str:
        """Execute the order test tool."""
        logger.info(f"Running OrderTestTool for patient {patient_id}")
        order = {
            "timestamp": datetime.now().isoformat(),
            "test_type": test_type,
            "priority": priority,
            "notes": notes
        }
        return f"Ordered test for patient {patient_id}: {json.dumps(order)}"
    

class ReferralTool(BaseTool):
    """Tool for specialist referrals."""
    name: ClassVar[str] = "refer_specialist"
    description: ClassVar[str] = "Refer patient to specialist"
    args_schema: ClassVar[Type[BaseModel]] = ReferralSchema
    
    def _run(self, patient_id: str, specialty: str, priority: str, reason: str) -> str:
        """Execute the referral tool."""
        logger.info(f"Running ReferralTool for patient {patient_id}")
        referral = {
            "timestamp": datetime.now().isoformat(),
            "specialty": specialty,
            "priority": priority,
            "reason": reason
        }
        return f"Created referral for patient {patient_id}: {json.dumps(referral)}"
