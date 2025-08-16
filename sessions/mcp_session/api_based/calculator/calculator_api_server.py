import math
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Calculator API", version="1.0.0")

class CalculationRequest(BaseModel):
    expression: str

class CalculationResponse(BaseModel):
    expression: str
    result: float
    success: bool
    error: str = None

@app.post("/calculate", response_model=CalculationResponse)
async def calculate(request: CalculationRequest):
    """Calculate mathematical expressions"""
    try:
        safe_dict = {
            "__builtins__": {},
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, 
            "tan": math.tan, "log": math.log, "exp": math.exp,
            "pi": math.pi, "e": math.e, "abs": abs, "round": round
        }
        
        result = eval(request.expression, safe_dict)
        
        # Store in history
        calc_entry = {
            "expression": request.expression,
            "result": result,
            "success": True
        }
            
        return CalculationResponse(
            expression=request.expression,
            result=result,
            success=True
        )
        
    except Exception as e:
        error_entry = {
            "expression": request.expression,
            "error": str(e),
            "success": False
        }
        return CalculationResponse(
            expression=request.expression,
            result=0,
            success=False,
            error=str(e)
        )

@app.get("/functions")
async def get_functions():
    """Get available mathematical functions and examples"""
    return {
        "operations": ["+", "-", "*", "/", "**", "%"],
        "functions": {
            "sqrt(x)": "Square root",
            "sin(x)": "Sine (radians)",
            "cos(x)": "Cosine (radians)",
            "tan(x)": "Tangent (radians)",
            "log(x)": "Natural logarithm",
            "exp(x)": "Exponential",
            "abs(x)": "Absolute value",
            "round(x, n)": "Round to n decimals"
        },
        "constants": {
            "pi": 3.141592653589793,
            "e": 2.718281828459045
        },
        "examples": [
            "sqrt(16)",
            "sin(pi/2)", 
            "2*pi*5",
            "log(e)",
            "exp(1)"
        ]
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Calculator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /calculate": "Calculate mathematical expressions",
            "GET /functions": "Get available functions",
            "GET /": "This information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Calculator REST API...")
    print("API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)