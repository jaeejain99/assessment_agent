import os
import json
import sqlglot
from sqlglot.expressions import Select
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from langchain_databricks import ChatDatabricks
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from databricks.vector_search.client import VectorSearchClient 

import mlflow
mlflow.langchain.autolog()

# Initialize Databricks SDK client for SQL execution
def execute_sql_query(sql: str):
    """Execute SQL using Databricks SDK statement execution API."""
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
    print(f"TOOL DEBUG: Executing SQL via SDK...")
    w = WorkspaceClient()
    
    # Execute SQL statement
    print(f"TOOL DEBUG: Submitting SQL statement...")
    import time
    start_time = time.time()
    
    result = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="30s"
    )
    elapsed = time.time() - start_time
    print(f"TOOL DEBUG: Query execution took {elapsed:.2f}s")
    print(f"TOOL DEBUG: Statement executed successfully")
    return result

CATALOG = "rag"
SCHEMA = "wb_operations"
ALLOWED_TABLE = f"{CATALOG}.{SCHEMA}.gold_wb_projects".lower()

# -------------------------------------------------------------
# Tool 1: Structured Text-to-SQL with AST Safety Validation
# -------------------------------------------------------------
@tool
def query_wb_structured_projects(sql_query: str) -> str:
    """
    Execute read-only SQL queries on the World Bank projects Gold Delta table:
    Table: rag.wb_operations.gold_wb_projects
    """
    print(f"TOOL DEBUG: query_wb_structured_projects called with SQL: {sql_query}")
    try:
        # Parse AST to enforce SELECT-only policy
        parsed = sqlglot.parse_one(sql_query, read="databricks")
        if not isinstance(parsed, Select):
            return "Security Error: Only SELECT queries are permitted."
            
        # Verify table permissions
        for table in parsed.find_all(sqlglot.exp.Table):
            table_name = f"{table.db}.{table.name}" if table.db else table.name
            if table.name.lower() not in ["gold_wb_projects", ALLOW_TABLE := ALLOWED_TABLE.split('.')[-1]]:
                return f"Access Denied: Querying unauthorized table '{table_name}'."
                
        # Enforce deterministic output limit
        if not parsed.args.get("limit"):
            parsed = parsed.limit(50)
            
        executable_sql = parsed.sql(dialect="databricks")
        
        # Execute using Databricks SDK
        print(f"TOOL DEBUG: Executing SQL: {executable_sql}")
        result = execute_sql_query(executable_sql)
        
        # Convert SDK result to pandas DataFrame
        print(f"TOOL DEBUG: Result status state: {result.status.state if result.status else 'No status'}")
        
        if result.status and result.status.state == StatementState.SUCCEEDED:
            print(f"TOOL DEBUG: Query succeeded, processing results...")
            
            # Extract column names and data
            columns = [col.name for col in result.manifest.schema.columns] if result.manifest and result.manifest.schema else []
            print(f"TOOL DEBUG: Columns: {columns}")
            
            # Extract rows - data_array is already a list of rows
            rows = result.result.data_array if result.result and result.result.data_array else []
            print(f"TOOL DEBUG: Raw rows: {rows[:2] if rows else 'empty'}")
            
            print(f"TOOL DEBUG: Got {len(rows)} rows, {len(columns)} columns")
            result_pdf = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
            result_json = result_pdf.to_json(orient="records", date_format="iso")
            print(f"TOOL DEBUG: Returning result (length: {len(result_json)})")
            return result_json
        else:
            error_msg = f"Query failed: status={result.status.state if result.status else 'None'}, error={result.status.error if result.status and hasattr(result.status, 'error') else 'No error attribute'}"
            print(f"TOOL DEBUG ERROR: {error_msg}")
            return error_msg
    except Exception as e:
        error_msg = f"SQL Execution Failed: {str(e)}"
        print(f"TOOL DEBUG ERROR: {error_msg}")
        return error_msg

# -------------------------------------------------------------
# Tool 2: Unstructured Project Appraisal Document (PAD) Search
# -------------------------------------------------------------
@tool
def search_wb_appraisal_documents(query: str, project_id: str = "") -> str:
    """
    Perform semantic vector search across World Bank Project Appraisal Documents (PAD PDFs).
    
    Args:
        query: The search query or project name (e.g., "Sahel Regional Engagement for Learning")
        project_id: Optional project ID (e.g., 'P178942') to scope search to specific project
    
    Returns detailed excerpts with citations from PAD documents.
    """
    try:
        print(f"TOOL DEBUG: search_wb_appraisal_documents called with query: {query}")
        vsc = VectorSearchClient(
            workspace_url=os.getenv("DATABRICKS_WORKSPACE_URL", ""),
            personal_access_token=os.getenv("PERSONAL_ACCESS_TOKEN", ""),
            disable_notice=True)
        
        print(f"TOOL DEBUG: Getting vector search index...")
        index = vsc.get_index(
            endpoint_name="wb_ops_vector_endpoint",
            index_name=f"{CATALOG}.{SCHEMA}.gold_wb_pad_chunks_index"
        )
        print(f"TOOL DEBUG: Index retrieved: {index.name}")
        
        filters = {}
        if project_id and project_id.strip():
            filters = {"project_id": project_id.strip()}
            
        print(f"TOOL DEBUG: Performing similarity search...")
        results = index.similarity_search(
            query_text=query,
            columns=["project_id", "document_title", "page_number", "source_url", "chunk_text"],
            filters=filters,
            num_results=4
        )
        
        print(f"TOOL DEBUG: Search completed, processing results...")
        data_rows = results.get("result", {}).get("data_array", [])
        print(f"TOOL DEBUG: Found {len(data_rows)} results")
        
        if not data_rows:
            return "No matching appraisal document excerpts found."
            
        formatted_chunks = []
        for row in data_rows:
            formatted_chunks.append(
                f"[Project ID: {row[0]} | Doc: {row[1]} | Page: {row[2]}]\n"
                f"Excerpt: {row[4]}\n"
                f"Source: {row[3]}"
            )
        
        result_text = "\n\n---\n\n".join(formatted_chunks)
        print(f"TOOL DEBUG: Returning {len(formatted_chunks)} chunks (length: {len(result_text)})")
        return result_text
    except Exception as e:
        error_msg = f"Vector Retrieval Failed: {str(e)}"
        print(f"TOOL DEBUG ERROR: {error_msg}")
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}")
        return error_msg

# -------------------------------------------------------------
# Agent Core Definition
# -------------------------------------------------------------
SYSTEM_PROMPT = """You are the World Bank Operations & Project Appraisal Enterprise AI Assistant.

You have TWO tools available:

1. `query_wb_structured_projects` - SQL database with basic project info (country, funding, status, dates)
2. `search_wb_appraisal_documents` - Vector search over detailed Project Appraisal Documents (PADs)

DECISION TREE - Follow this exactly:

QUESTION: "What is [Project Name]?" or "Tell me about [Project Name]"
ACTION: IMMEDIATELY call `search_wb_appraisal_documents` tool with the project name as the query
DO NOT Try the SQL database first. DO NOT say you cannot find information without trying vector search.

QUESTION: "How many projects in [Country]?" or "Total funding for [Sector]?"
ACTION: Call `query_wb_structured_projects` with appropriate SQL query

QUESTION: Mix of both (e.g., "What are Mongolia projects and their objectives?")
ACTION: Call BOTH tools - SQL first for the list, then vector search for details

CRITICAL: If a user asks about a specific named project, you MUST call `search_wb_appraisal_documents`. 
Saying "I couldn't find information" without calling this tool is a failure.

When you get results from `search_wb_appraisal_documents`, synthesize them into a clear answer with citations. Adding Citations is CRITICAL for results from `search_wb_appraisal_documents` tool only.

If user asks a question that is not related to World Bank Projects, respond with "Please ask a question related to World Bank projects".

--- SQL GENERATION RULES (CRITICAL) ---
DO NOT give source details for SQL queries. For failed SQL queries, say "Information not available".

Table: rag.wb_operations.gold_wb_projects
Columns: project_id, project_name, country, region, total_commitment_usd, ibrd_commitment_usd, ida_commitment_usd, status, approval_date, primary_sector

DATA FORMAT RULES:
1. country column: Stored as JSON arrays like ["Republic of India"], ["Mongolia"], ["Republic of Kenya"]
   - ALWAYS use: WHERE country LIKE '%India%' (NOT country = 'India')
   - For multiple countries: WHERE (country LIKE '%India%' OR country LIKE '%Kenya%')

2. status column: Case-sensitive, exact values are "Active" or "Closed" (capital first letter)
   - ALWAYS use: WHERE status = 'Active' (NOT status = 'active')

3. Common query patterns:
   - Count active projects in India: SELECT COUNT(*) FROM rag.wb_operations.gold_wb_projects WHERE country LIKE '%India%' AND status = 'Active'
   - List projects by country: SELECT project_id, project_name, total_commitment_usd FROM rag.wb_operations.gold_wb_projects WHERE country LIKE '%[CountryName]%'
   - Sum funding by region: SELECT region, SUM(total_commitment_usd) as total_usd FROM rag.wb_operations.gold_wb_projects GROUP BY region
"""

def build_wb_agent():
    llm = ChatDatabricks(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        temperature=0.0, 
        max_tokens=1500
    )
    
    tools = [query_wb_structured_projects, search_wb_appraisal_documents]
    return create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)

class WorldBankAgentSession:
    def __init__(self):
        self.agent = build_wb_agent()
        self.history = []

    def ask(self, query: str) -> str:
        try:
            print(f"DEBUG: Starting query: {query}")
            self.history.append(HumanMessage(content=query))
            
            print(f"DEBUG: Invoking agent...")
            response = self.agent.invoke(
                {"messages": self.history},
                config={"recursion_limit": 20}  # Allow more steps to try multiple tools
            )
            
            print(f"DEBUG: Agent invoked successfully, processing response...")
            if not response or "messages" not in response:
                print(f"DEBUG: Invalid response structure: {response}")
                return "Error: No valid response from agent"
            
            final_answer = response["messages"][-1].content
            self.history.append(AIMessage(content=final_answer))
            print(f"FINAL RESPONSE: {final_answer}")
            return final_answer
        except TimeoutError as e:
            error_msg = f"Timeout Error: Query took too long to execute. {str(e)}"
            print(f"DEBUG TIMEOUT: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Agent Error: {str(e)}"
            print(f"DEBUG ERROR: {error_msg}")
            import traceback
            print(f"TRACEBACK: {traceback.format_exc()}")
            return error_msg