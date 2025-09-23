"""
Graph Builder for LangGraph SMART-LLM Pipeline.
Assembles the complete workflow with agents, routing, and validation.
"""

from langgraph.graph import StateGraph, END
from scripts.langgraph_pipeline.state.schema import PipelineState
from scripts.langgraph_pipeline.nodes.decomposition_node import DecompositionAgent
from scripts.langgraph_pipeline.nodes.allocation_node import AllocationAgent
from scripts.langgraph_pipeline.nodes.code_generation_node import CodeGenerationAgent
from scripts.langgraph_pipeline.edges.validators import validate_decomposition, validate_allocation, validate_code
from scripts.langgraph_pipeline.edges.routing_logic import create_conditional_edge_function, should_continue


class LangGraphSMARTLLM:
    """
    Main LangGraph-based SMART-LLM pipeline orchestrator.
    Builds and manages the complete workflow graph.
    """
    
    def __init__(self, api_key, decomp_model="gpt-4", alloc_model="gpt-4", codegen_model="gpt-4"):
        self.api_key = api_key
        
        # Initialize specialized agents
        self.decomp_agent = DecompositionAgent(decomp_model, api_key)
        self.alloc_agent = AllocationAgent(alloc_model, api_key)
        self.codegen_agent = CodeGenerationAgent(codegen_model, api_key)
        
        # Build the workflow graph
        self.workflow = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with conditional routing."""
        
        # Create the state graph
        workflow = StateGraph(PipelineState)
        
        # Add agent nodes
        workflow.add_node("decompose_task", self.decomp_agent)
        workflow.add_node("allocate_robots", self.alloc_agent)
        workflow.add_node("generate_code", self.codegen_agent)
        
        # Add validation nodes
        workflow.add_node("validate_decomposition", validate_decomposition)
        workflow.add_node("validate_allocation", validate_allocation)
        workflow.add_node("validate_code", validate_code)
        
        # Set entry point
        workflow.set_entry_point("decompose_task")
        
        # Create conditional routing function
        conditional_edge = create_conditional_edge_function()
        
        # Add conditional edges from each node
        workflow.add_conditional_edges(
            "decompose_task",
            lambda state: "validate_decomposition" if state['current_stage'] == 'allocation' else "decompose_task",
            {
                "validate_decomposition": "validate_decomposition",
                "decompose_task": "decompose_task"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_decomposition",
            lambda state: "allocate_robots" if state['current_stage'] == 'allocation' else "decompose_task",
            {
                "allocate_robots": "allocate_robots",
                "decompose_task": "decompose_task"
            }
        )
        
        workflow.add_conditional_edges(
            "allocate_robots",
            lambda state: "validate_allocation" if state['current_stage'] == 'code_generation' else "allocate_robots",
            {
                "validate_allocation": "validate_allocation",
                "allocate_robots": "allocate_robots"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_allocation",
            lambda state: "generate_code" if state['current_stage'] == 'code_generation' else ("decompose_task" if state['current_stage'] == 'decomposition' else "allocate_robots"),
            {
                "generate_code": "generate_code",
                "allocate_robots": "allocate_robots",
                "decompose_task": "decompose_task"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_code",
            lambda state: "validate_code" if state['current_stage'] == 'validation' else "generate_code",
            {
                "validate_code": "validate_code",
                "generate_code": "generate_code"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_code",
            lambda state: END if state['current_stage'] == 'complete' else ("generate_code" if state['current_stage'] == 'code_generation' else ("allocate_robots" if state['current_stage'] == 'allocation' else "decompose_task")),
            {
                END: END,
                "generate_code": "generate_code",
                "allocate_robots": "allocate_robots", 
                "decompose_task": "decompose_task"
            }
        )
        
        return workflow.compile()
    
    def process_single_task(self, task, available_robots, objects, ground_truth, trans, max_trans, 
                           floor_plan, examples):
        """Process a single task through the LangGraph pipeline."""
        
        print(f"\n📋 LangGraph Processing: {task}")
        print("=" * 60)
        
        # Initialize state with all required fields
        initial_state = PipelineState(
            # Original inputs (immutable)
            original_task=task,
            available_robots=available_robots,
            objects=objects,  # Now properly passed from main
            ground_truth=ground_truth,
            trans=trans,
            max_trans=max_trans,
            floor_plan=floor_plan,
            
            # Training examples (immutable)
            decomp_examples=examples[0],
            alloc_examples=examples[1],
            codegen_examples=examples[2],
            
            # Work products (mutable) 
            decomposed_plan="",
            allocation_plan="",
            generated_code="",
            
            # Process control
            current_stage="decomposition",
            iteration_count=0,
            max_iterations=9,
            
            # Agent iteration tracking
            decomposition_attempts=0,
            allocation_attempts=0,
            code_generation_attempts=0,
            max_attempts_per_agent=3,
            
            # Feedback system
            feedback_messages=[],
            errors=[],
            validation_results={},
            
            # Success tracking
            is_successful=False,
            final_stage_reached=""
        )
        
        try:
            # Execute the workflow
            print("🚀 Starting LangGraph workflow execution...")
            final_state = self.workflow.invoke(initial_state)
            
            # Log final results
            print(f"\n📊 LangGraph Results:")
            print(f"   Final Stage: {final_state.get('final_stage_reached', final_state.get('current_stage'))}")
            print(f"   Total Iterations: {final_state.get('iteration_count', 0)}")
            print(f"   Decomposition Attempts: {final_state.get('decomposition_attempts', 0)}")
            print(f"   Allocation Attempts: {final_state.get('allocation_attempts', 0)}")
            print(f"   Code Generation Attempts: {final_state.get('code_generation_attempts', 0)}")
            print(f"   Success: {final_state.get('is_successful', False)}")
            print(f"   Errors: {len(final_state.get('errors', []))}")
            
            return final_state
            
        except Exception as e:
            print(f"❌ LangGraph workflow error: {str(e)}")
            # Return failed state
            initial_state['current_stage'] = 'failed'
            initial_state['final_stage_reached'] = f'failed_with_exception: {str(e)}'
            initial_state['errors'].append(f"Workflow exception: {str(e)}")
            return initial_state


def create_langgraph_pipeline(api_key, decomp_model="gpt-4", alloc_model="gpt-4", codegen_model="gpt-4"):
    """Factory function to create the LangGraph pipeline."""
    return LangGraphSMARTLLM(api_key, decomp_model, alloc_model, codegen_model)
