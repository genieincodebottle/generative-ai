from ollama_functions import OllamaFunctions

from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from connect import connect
from config import load_config


# Schema for structured response
class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text. \
                           If no sentiment is detected, the value is 'Neutral'")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10. \
            If no aggressiveness is detected, the value is 1"
    )

def fecth_data():
    try:
        config = load_config()
        connection = connect(config)
        cursor = connection.cursor()
        # Executing a SQL query to insert data into  table
        cursor.execute(f"SELECT * from customer_calls")
        record = cursor.fetchall()
        print("Result ", record)
        return record
    except Exception as e:
        print(e)
        raise e

def save_data(inp, res_dict):
    try:
        config = load_config()
        connection = connect(config)
        cursor = connection.cursor()
        # Executing a SQL query to insert data into  table
        insert_query = f""" INSERT INTO customer_tagging (call_details, sentiment, aggresiveness) VALUES ('{inp}', '{res_dict['sentiment']}', {res_dict['aggressiveness']})"""
        cursor.execute(insert_query)
        connection.commit()
        print("Record inserted successfully")
        # Fetch result
        cursor.execute("SELECT * from customer_tagging")
        record = cursor.fetchall()
        #print("Result ", record)
    except Exception as e:
        print(e)
        raise e


def main():

    try:
        #Prompt template
        tagging_prompt = PromptTemplate.from_template(
        """
        Extract the desired information from the  passage given in triple Backticks. 
        
        Only extract the properties mentioned in the 'Classification' function.

        Please follow the response format as given.

        Passage:
        '''{input}'''
        """
        )

        # Chain
        llm = OllamaFunctions(model="llama3", format="json", temperature=0)
        structured_llm = llm.with_structured_output(Classification)
        chain = tagging_prompt | structured_llm

        for item in fecth_data():
            res = chain.invoke({"input": item[2]})
            res_dict = res.dict()
            print(res_dict)
            save_data(item[2], res_dict)

    except Exception as e:
        print(e)
        raise e


if __name__ == "__main__":
    main()